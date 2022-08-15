import torch
import numpy as np
import AGENT_NET
import torch.nn.functional as FU
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.nn.utils as nn_utils
import rl_utils
from copy import deepcopy
'''
We recommend installing version >= 0.2.0 of the PyTorch Profiler TensorBoard plugin.
 Would you like to install the package'''
probs_softmax_add=1e-14

class PPO:
    ''' PPO算法,采用截断方式 '''
    def __init__(self,input_shape:tuple,num_subtasks,lr,weights,gamma,device,clip_grad,lmbda,epochs,eps):
        self.writer=SummaryWriter(comment='PPO')
        self.step=0
        self.agent=AGENT_NET.DoubleNet(input_shape,num_subtasks).to(device)
        self.agent_optimizer=torch.optim.NAdam(self.agent.parameters(),lr=lr,eps=1e-8)
        self.input_shape=input_shape
        self.num_processors=input_shape[0]
        self.num_subtasks=num_subtasks
        self.gamma=gamma
        self.weights=weights
        self.device=device
        self.num_subtasks=num_subtasks
        self.clip_grad=clip_grad
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device

    def take_action(self, state):
        F=lambda x:torch.tensor(x,dtype=torch.float).to(self.device)
        state=(F(state[0]),F(state[1]))
        u=state[0][:,:,:,:-self.num_subtasks]
        for i in u:
            i[:]=(i-i.mean())/i.std()
        for i in state[1]:
            i[:]=(i-i.mean())/i.std()
        (probs_subtasks_orginal,probs_prior_orginal),_=self.agent(state)
        action_subtasks=[torch.distributions.Categorical(x).sample().item()
            for x in probs_subtasks_orginal]

        action_prior=[]
        probs_prior_orginal=torch.cat(probs_prior_orginal,0)
        probs_prior_orginal=torch.tensor(
            np.concatenate((probs_prior_orginal.cpu().detach().numpy(),np.arange(self.num_subtasks).reshape(1,-1)),0),dtype=torch.float)
        for i in range(self.num_subtasks):
            x=torch.tensor(np.delete(probs_prior_orginal.numpy(),action_prior,1),dtype=torch.float)
            action_prior.append(x[-1,torch.distributions.Categorical(x[i]).sample().item()].int().item())
        
        action=np.zeros((2,self.num_subtasks),dtype='int')
        action[0]=action_subtasks
        action[1]=action_prior
        return action

    def update(self, transition_dict):
        F=lambda x:torch.tensor(x,dtype=torch.float).to(self.device)
        states=tuple(F(np.concatenate([x[i] for x in transition_dict['states']],0)) for i in range(len(transition_dict['states'][0])))
        u=states[0][:,:,:,:-self.num_subtasks]
        for i in u:
            i[:]=(i-i.mean())/i.std()
        for i in states[1]:
            i[:]=(i-i.mean())/i.std()
        actions=tuple(F(np.vstack([x[i] for x in transition_dict['actions']])).type(torch.int64) for i in range(len(transition_dict['actions'][0])))

        rewards=F(transition_dict['rewards']).view(-1,1)
        next_states=tuple(F(np.concatenate([x[i] for x in transition_dict['next_states']],0)) for i in range(len(transition_dict['states'][0])))
        u=next_states[0][:,:,:,:-self.num_subtasks]
        for i in u:
            i[:]=(i-i.mean())/i.std()
        for i in next_states[1]:
            i[:]=(i-i.mean())/i.std()
        overs=F(transition_dict['overs']).view(-1,1)

        td_target = rewards + self.gamma * self.agent(next_states)[1] * (1 - overs)
        td_delta = td_target - self.agent(states)[1]
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        #self.writer.add_scalar(tag='advantage',scalar_value=advantage,global_step=self.step//self.epochs)
        old_log_probs = torch.log(self.calculate_probs(self.agent(states)[0],actions)).detach()

        for _ in range(self.epochs):
            probs=self.agent(states)[0]
            log_probs = torch.log(self.calculate_probs(probs,actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps,
                                1 + self.eps) * advantage  # 截断
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
            critic_loss = torch.mean(
                FU.mse_loss(self.agent(states)[1], td_target.detach()))
            loss=actor_loss+self.weights*critic_loss
            if torch.isnan(loss)>0:
                print("here!")
            self.agent_optimizer.zero_grad()
            if not self.clip_grad=='max':
                nn_utils.clip_grad_norm_(self.agent.parameters(),self.clip_grad)
            loss.backward()
            self.agent_optimizer.step()

            self.writer.add_scalar(tag='cri_loss',scalar_value=critic_loss.item(),global_step=self.step)
            self.writer.add_scalar(tag='act_loss',scalar_value=actor_loss.item(),global_step=self.step)
            self.writer.add_scalar(tag='agent_loss',scalar_value=loss.item(),global_step=self.step)
            grad_max = 0.0
            grad_means = 0.0
            grad_count = 0
            for p in self.agent.parameters():
                grad_max = max(grad_max, p.grad.abs().max().item())
                grad_means += (p.grad ** 2).mean().sqrt().item()
                grad_count += 1
            self.writer.add_scalar("grad_l2", grad_means / grad_count, self.step)
            self.writer.add_scalar("grad_max", grad_max, self.step)
            probs_new=self.agent(states)[0]
            kl=0
            for i in range(2):
                for p_old,p_new in zip(probs[i],probs_new[i]):
                    kl+=-((p_new/p_old).log()*p_old).sum(dim=1).mean().item()
            self.writer.add_scalar("kl", kl, self.step)
            self.step+=1
    
    def calculate_probs(self,out_puts,actions):
        F=lambda i:torch.gather(out_puts[0][i],1,actions[0][:,[i]])*F(i+1)\
            if i<self.num_subtasks else 1.0
        probs=F(0)

        G=lambda i:((torch.gather(out_puts[1][i],1,actions[1][:,[i]])+1e-7)
            /(out_puts[1][i].sum(axis=1,keepdim=True)
                -torch.gather(out_puts[1][i],1,actions[1][:,:i]).sum(axis=1,keepdim=True)+1e-7)*G(i+1)
            if i<self.num_subtasks else 1.0)
        probs*=G(0)
        return probs

class PPO_softmax0:
    ''' PPO算法,采用截断方式 '''
    def __init__(self,input_shape:tuple,num_subtasks,lr,weights,gamma,device,clip_grad,lmbda,epochs,eps,beta,tanh):
        self.writer=SummaryWriter(comment='PPO')
        self.step=0
        self.agent=AGENT_NET.DoubleNet_softmax_simple(input_shape,num_subtasks,tanh).to(device)
        self.agent_optimizer=torch.optim.NAdam(self.agent.parameters(),lr=lr,eps=1e-8)
        self.input_shape=input_shape
        self.num_processors=input_shape[0]
        self.num_subtasks=num_subtasks
        self.gamma=gamma
        self.weights=weights
        self.device=device
        self.num_subtasks=num_subtasks
        self.clip_grad=clip_grad
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device
        self.beta=beta

    def take_action(self, state):
        F=lambda x:torch.tensor(x,dtype=torch.float).to(self.device)
        state=(F(state[0]),F(state[1]))
        u=state[0][:,:,:,:-self.num_subtasks]
        for i in u:
            i[:]=(i-i.mean())/i.std()
        for i in state[1]:
            i[:]=(i-i.mean())/i.std()
        (probs_subtasks_orginal,probs_prior_orginal),_=self.agent(state)
        action_subtasks=[torch.distributions.Categorical(logits=x).sample().item()
            for x in probs_subtasks_orginal]

        action_prior=[]
        probs_prior_orginal=torch.cat(probs_prior_orginal,0)
        probs_prior_orginal=torch.tensor(
            np.concatenate((probs_prior_orginal.cpu().detach().numpy(),np.arange(self.num_subtasks).reshape(1,-1)),0),dtype=torch.float)
        for i in range(self.num_subtasks):
            x=torch.tensor(np.delete(probs_prior_orginal.numpy(),action_prior,1),dtype=torch.float)
            action_prior.append(x[-1,torch.distributions.Categorical(logits=x[i]).sample().item()].int().item())
        
        action=np.zeros((2,self.num_subtasks),dtype='int')
        action[0]=action_subtasks
        action[1]=action_prior
        return action

    def update(self, transition_dict):
        F=lambda x:torch.tensor(x,dtype=torch.float).to(self.device)
        states=tuple(F(np.concatenate([x[i] for x in transition_dict['states']],0)) for i in range(len(transition_dict['states'][0])))
        u=states[0][:,:,:,:-self.num_subtasks]
        for i in u:
            i[:]=(i-i.mean())/i.std()
        for i in states[1]:
            i[:]=(i-i.mean())/i.std()
        actions=tuple(F(np.vstack([x[i] for x in transition_dict['actions']])).type(torch.int64) for i in range(len(transition_dict['actions'][0])))

        rewards=F(transition_dict['rewards']).view(-1,1)
        next_states=tuple(F(np.concatenate([x[i] for x in transition_dict['next_states']],0)) for i in range(len(transition_dict['states'][0])))
        u=next_states[0][:,:,:,:-self.num_subtasks]
        for i in u:
            i[:]=(i-i.mean())/i.std()
        for i in next_states[1]:
            i[:]=(i-i.mean())/i.std()
        overs=F(transition_dict['overs']).view(-1,1)

        all_states=tuple(torch.concat((states[i][0:1],next_states[i])) for i in range(2))
        temp=self.agent(all_states)[1]
        td_target = rewards + self.gamma * temp[1:] * (1 - overs)
        
        td_delta = td_target - temp[:-1]
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.calculate_probs(self.agent(states)[0],actions)).detach()

        for _ in range(self.epochs):
            #probs不是概率，而是神经网络的输出
            probs=self.agent(states)[0]
            log_probs = torch.log(self.calculate_probs(probs,actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps,
                                1 + self.eps) * advantage  # 截断
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
            critic_loss = torch.mean(
                FU.mse_loss(self.agent(states)[1], td_target.detach()))

            '''s=0
            for prob in probs[0]:
                s+=((FU.softmax(prob,dim=1)+probs_softmax_add)*FU.log_softmax(prob,dim=1)).sum(dim=1)
            t=0
            for prob in probs[1]:
                t+=((FU.softmax(prob,dim=1)+probs_softmax_add)*FU.log_softmax(prob,dim=1)).sum(dim=1)
            epo_loss=self.beta*(s.mean()+t.mean())'''

            s=0
            for i in range(2):
                for prob in probs[0]:
                    u=FU.softmax(prob,dim=1)+probs_softmax_add
                    s+=(u*u.log()).sum(dim=1)
            epo_loss=self.beta*(s.mean())
            '''s=0
            for i in range(2):
                for prob in probs[0]:
                    s+=((FU.softmax(prob,dim=1)+probs_softmax_add)*(FU.softmax(prob,dim=1)+probs_softmax_add).log()).sum(dim=1)
            epo_loss=self.beta*(s.mean())'''

            loss=actor_loss+self.weights*critic_loss
            if self.beta:
                loss+=epo_loss
            
            #loss=actor_loss+self.weights*critic_loss
            if torch.isnan(loss)>0:
                print("here!")
            self.agent_optimizer.zero_grad()
            if not self.clip_grad=='max':
                nn_utils.clip_grad_norm_(self.agent.parameters(),self.clip_grad)
            loss.backward()
            self.agent_optimizer.step()

            self.writer.add_scalar(tag='cri_loss',scalar_value=critic_loss.item(),global_step=self.step)
            self.writer.add_scalar(tag='act_loss',scalar_value=actor_loss.item(),global_step=self.step)
            self.writer.add_scalar(tag='agent_loss',scalar_value=loss.item(),global_step=self.step)
            grad_max = 0.0
            grad_means = 0.0
            grad_count = 0
            for p in self.agent.parameters():
                grad_max = max(grad_max, p.grad.abs().max().item())
                grad_means += (p.grad ** 2).mean().sqrt().item()
                grad_count += 1
            self.writer.add_scalar("grad_l2", grad_means / grad_count, self.step)
            self.writer.add_scalar("grad_max", grad_max, self.step)

            probs_new=self.agent(states)[0]
            probs_new=tuple([FU.softmax(x,dim=1)+probs_softmax_add for x in probs_new[i]] for i in range(2))
            probs=tuple([FU.softmax(x,dim=1)+probs_softmax_add for x in probs[i]] for i in range(2))
            kl=0
            for i in range(2):
                for p_old,p_new in zip(probs[i],probs_new[i]):
                    kl+=-((p_new/p_old).log()*p_old).sum(dim=1).mean().item()
            self.writer.add_scalar(tag="kl", scalar_value=kl, global_step=self.step)
            self.step+=1
    
    def calculate_probs(self,out_puts,actions):
        out_puts=tuple([FU.softmax(x,dim=1) for x in out_puts[i]] for i in range(2))
        probs=1
        for i in range(self.num_subtasks):
            t=torch.gather(out_puts[0][i],1,actions[0][:,[i]])
            for k,tt in enumerate(t):
                if tt.item()==0:
                    print(k,':prob_sub is zero')
            probs*=t
        for i in range(self.num_subtasks-1):
            t=torch.gather(out_puts[1][i],1,actions[1][:,[i]])
            u=out_puts[1][i].sum(axis=1,keepdim=True)
            s=torch.gather(out_puts[1][i],1,actions[1][:,:i]).sum(axis=1,keepdim=True)
            for k,tt in enumerate(t):
                if tt.item()==0:
                    print(k,':prob_prior_fz is zero')
            for k,tt in enumerate(u-s):
                if tt.item()==0:
                    print(k,':prob_prior_fm is zero')
            probs*=t/(u-s)
        return probs

class PPO_softmax_back:
    ''' PPO算法,采用截断方式 '''
    def __init__(self,input_shape:tuple,num_subtasks,weights,gamma,device,clip_grad,lmbda,epochs,eps,beta,net,optim):
        self.writer=SummaryWriter(comment='PPO')
        self.step=0
        self.agent=net
        self.agent_optimizer=optim
        self.input_shape=input_shape
        self.num_processors=input_shape[0]
        self.num_subtasks=num_subtasks
        self.gamma=gamma
        self.weights=weights
        self.device=device
        self.num_subtasks=num_subtasks
        self.clip_grad=clip_grad
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device
        self.beta=beta

    def take_action(self, state):
        F=lambda x:torch.tensor(x,dtype=torch.float).to(self.device)
        state=(F(state[0]),F(state[1]))
        u=state[0][:,:,:,:-self.num_subtasks]
        for i in u:
            i[:]=(i-i.mean())/i.std()
        for i in state[1]:
            i[:]=(i-i.mean())/i.std()
        (probs_subtasks_orginal,probs_prior_orginal),_=self.agent(state)
        action_subtasks=[torch.distributions.Categorical(logits=x).sample().item()
            for x in probs_subtasks_orginal]

        action_prior=[]
        probs_prior_orginal=torch.cat(probs_prior_orginal,0)
        probs_prior_orginal=torch.tensor(
            np.concatenate((probs_prior_orginal.cpu().detach().numpy(),np.arange(self.num_subtasks).reshape(1,-1)),0),dtype=torch.float)
        for i in range(self.num_subtasks):
            x=torch.tensor(np.delete(probs_prior_orginal.numpy(),action_prior,1),dtype=torch.float)
            action_prior.append(x[-1,torch.distributions.Categorical(logits=x[i]).sample().item()].int().item())
        
        action=np.zeros((2,self.num_subtasks),dtype='int')
        action[0]=action_subtasks
        action[1]=action_prior
        return action

    def update(self, transition_dict):
        F=lambda x:torch.tensor(x,dtype=torch.float).to(self.device)
        states=tuple(F(np.concatenate([x[i] for x in transition_dict['states']],0)) for i in range(len(transition_dict['states'][0])))
        u=states[0][:,:,:,:-self.num_subtasks]
        for i in u:
            i[:]=(i-i.mean())/i.std()
        for i in states[1]:
            i[:]=(i-i.mean())/i.std()
        actions=tuple(F(np.vstack([x[i] for x in transition_dict['actions']])).type(torch.int64) for i in range(len(transition_dict['actions'][0])))

        rewards=F(transition_dict['rewards']).view(-1,1)
        next_states=tuple(F(np.concatenate([x[i] for x in transition_dict['next_states']],0)) for i in range(len(transition_dict['states'][0])))
        u=next_states[0][:,:,:,:-self.num_subtasks]
        for i in u:
            i[:]=(i-i.mean())/i.std()
        for i in next_states[1]:
            i[:]=(i-i.mean())/i.std()
        overs=F(transition_dict['overs']).view(-1,1)

        all_states=tuple(torch.concat((states[i][0:1],next_states[i])) for i in range(2))
        temp=self.agent(all_states)[1]
        td_target = rewards + self.gamma * temp[1:] * (1 - overs)
        
        td_delta = td_target - temp[:-1]
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        #advantage=(advantage-advantage.mean())/advantage.std()
        #td_target=advantage.view(-1,1)+temp[:-1]                                               
        old_log_probs = self.calculate_probs_log(self.agent(states)[0],actions).detach()

        #probs不是概率，而是神经网络的输出
        probs=self.agent(states)[0]
        log_probs = self.calculate_probs_log(probs,actions)
        ratio = torch.exp(log_probs - old_log_probs)

        for j in range(self.epochs):

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps,
                                1 + self.eps) * advantage  # 截断
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
            critic_loss = torch.mean(
                FU.mse_loss(self.agent(states)[1], td_target.detach()))

            s=0
            for i in range(2):
                for prob in probs[0]:
                    s+=((FU.softmax(prob,dim=1))*FU.log_softmax(prob,dim=1)).sum(dim=1)
            epo_loss=self.beta*(s.mean())

            loss=actor_loss+self.weights*critic_loss
            if self.beta:
                loss+=epo_loss
            
            if torch.isnan(loss)>0:
                print("here!")

            net_old=deepcopy(self.agent)
            optim_old_state=deepcopy(self.agent_optimizer.state)
            
            self.agent_optimizer.zero_grad()
            if not self.clip_grad=='max':
                nn_utils.clip_grad_norm_(self.agent.parameters(),self.clip_grad)
            loss.backward()
            self.agent_optimizer.step()

            probs=self.agent(states)[0]
            log_probs = self.calculate_probs_log(probs,actions)
            ratio = torch.exp(log_probs - old_log_probs)

            flag=0
            for r in ratio:
                if r>1+self.eps or r<1-self.eps:
                    '''self.agent=net_old
                    for param in net_old.parameters():
                        optim_old.add_param_group(param)
                    self.agent_optimizer=optim_old'''
                    for data,data_old in zip(self.agent.parameters(),net_old.parameters()):
                        data.data=data_old.data
                    self.agent_optimizer.state=optim_old_state
                    print('ratio_over:',r)
                    flag=1
                    break
            if flag:
                break

            self.writer.add_scalar(tag='cri_loss',scalar_value=critic_loss.item(),global_step=self.step)
            self.writer.add_scalar(tag='act_loss',scalar_value=actor_loss.item(),global_step=self.step)
            self.writer.add_scalar(tag='epo_loss',scalar_value=epo_loss.item(),global_step=self.step)
            self.writer.add_scalar(tag='agent_loss',scalar_value=loss.item(),global_step=self.step)
            if j>0:
                self.writer.add_scalar(tag='ratio',scalar_value=ratio.mean().item(),global_step=self.step)
            grad_max = 0.0
            grad_means = 0.0
            grad_count = 0
            for p in self.agent.parameters():
                grad_max = max(grad_max, p.grad.abs().max().item())
                grad_means += (p.grad ** 2).mean().sqrt().item()
                grad_count += 1
            self.writer.add_scalar("grad_l2", grad_means / grad_count, self.step)
            self.writer.add_scalar("grad_max", grad_max, self.step)

            probs_new=self.agent(states)[0]
            kl=0
            for i in range(2):
                for p_old,p_new in zip(probs[i],probs_new[i]):
                    kl+=-((p_new-p_old)*FU.softmax(p_old,dim=1)).sum(dim=1).mean().item()
            self.writer.add_scalar(tag="kl", scalar_value=kl, global_step=self.step)
            self.step+=1
        
        probs=self.agent(states)[0]
        log_probs = self.calculate_probs_log(probs,actions)
        ratio = torch.exp(log_probs - old_log_probs)
        for r in ratio:
            if r>1+self.eps or r<1-self.eps:
                print('ratio_last_over:',r)
        self.writer.add_scalar(tag='ratio',scalar_value=ratio.mean().item(),global_step=self.step)
    
    def calculate_probs_log(self,out_puts,actions):
        probs_log=0
        for i in range(self.num_subtasks):
            t=torch.gather(FU.log_softmax(out_puts[0][i],dim=1),1,actions[0][:,[i]])
            probs_log+=t
        for i in range(self.num_subtasks-1):
            t=torch.gather(out_puts[1][i],1,actions[1][:,[i]])
            u=out_puts[1][i].exp().sum(axis=1,keepdim=True)
            s=torch.gather(out_puts[1][i].exp(),1,actions[1][:,:i]).sum(axis=1,keepdim=True)
            probs_log+=t-(u-s).log()
        return probs_log


class PPO_softmax1:
    ''' PPO算法,采用截断方式 '''
    def __init__(self,input_shape:tuple,num_subtasks,weights,gamma,device,clip_grad,lmbda,epochs,eps,beta,net,optim,cut):
        self.writer=SummaryWriter(comment='PPO')
        self.step=0
        self.agent=net
        self.agent_optimizer=optim
        self.input_shape=input_shape
        self.num_processors=input_shape[0]
        self.num_subtasks=num_subtasks
        self.gamma=gamma
        self.weights=weights
        self.device=device
        self.num_subtasks=num_subtasks
        self.clip_grad=clip_grad
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device
        self.beta=beta
        self.cut=cut

    def take_action(self, state):
        F=lambda x:torch.tensor(x,dtype=torch.float).to(self.device)
        state=(F(state[0]),F(state[1]))
        u=state[0][:,:,:,:-self.num_subtasks]
        for i in u:
            i[:]=(i-i.mean())/i.std()
        for i in state[1]:
            i[:]=(i-i.mean())/i.std()
        (probs_subtasks_orginal,probs_prior_orginal),_=self.agent(state)
        action_subtasks=[torch.distributions.Categorical(logits=x).sample().item()
            for x in probs_subtasks_orginal]

        action_prior=[]
        probs_prior_orginal=torch.cat(probs_prior_orginal,0)
        probs_prior_orginal=torch.tensor(
            np.concatenate((probs_prior_orginal.cpu().detach().numpy(),np.arange(self.num_subtasks).reshape(1,-1)),0),dtype=torch.float)
        for i in range(self.num_subtasks):
            x=torch.tensor(np.delete(probs_prior_orginal.numpy(),action_prior,1),dtype=torch.float)
            action_prior.append(x[-1,torch.distributions.Categorical(logits=x[i]).sample().item()].int().item())
        
        action=np.zeros((2,self.num_subtasks),dtype='int')
        action[0]=action_subtasks
        action[1]=action_prior
        return action

    def update(self, transition_dict):
        F=lambda x:torch.tensor(x,dtype=torch.float).to(self.device)
        states=tuple(F(np.concatenate([x[i] for x in transition_dict['states']],0)) for i in range(len(transition_dict['states'][0])))
        u=states[0][:,:,:,:-self.num_subtasks]
        for i in u:
            i[:]=(i-i.mean())/i.std()
        for i in states[1]:
            i[:]=(i-i.mean())/i.std()
        actions=tuple(F(np.vstack([x[i] for x in transition_dict['actions']])).type(torch.int64) for i in range(len(transition_dict['actions'][0])))

        rewards=F(transition_dict['rewards']).view(-1,1)
        next_states=tuple(F(np.concatenate([x[i] for x in transition_dict['next_states']],0)) for i in range(len(transition_dict['states'][0])))
        u=next_states[0][:,:,:,:-self.num_subtasks]
        for i in u:
            i[:]=(i-i.mean())/i.std()
        for i in next_states[1]:
            i[:]=(i-i.mean())/i.std()
        overs=F(transition_dict['overs']).view(-1,1)

        all_states=tuple(torch.concat((states[i][0:1],next_states[i])) for i in range(2))
        temp=self.agent(all_states)[1]
        td_target = rewards + self.gamma * temp[1:] * (1 - overs)
        
        td_delta = td_target - temp[:-1]
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        #advantage=(advantage-advantage.mean())/advantage.std()
        #td_target=advantage.view(-1,1)+temp[:-1]                                               
        old_log_probs = self.calculate_probs_log(self.agent(states)[0],actions).detach()

        for j in range(self.epochs):
            #probs不是概率，而是神经网络的输出
            probs=self.agent(states)[0]
            log_probs = self.calculate_probs_log(probs,actions)
            ratio = torch.exp(log_probs - old_log_probs)

            flag=0
            for r in ratio:
                if r>1+self.eps or r<1-self.eps:
                    print('ratio_pre_over:',r)
                    flag=1
            if self.cut and flag:
                break

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps,
                                1 + self.eps) * advantage  # 截断
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
            critic_loss = torch.mean(
                FU.mse_loss(self.agent(states)[1], td_target.detach()))

            s=0
            for i in range(2):
                for prob in probs[0]:
                    s+=((FU.softmax(prob,dim=1))*FU.log_softmax(prob,dim=1)).sum(dim=1)
            epo_loss=self.beta*(s.mean())

            loss=actor_loss+self.weights*critic_loss
            if self.beta:
                loss+=epo_loss
            
            if torch.isnan(loss)>0:
                print("here!")
            self.agent_optimizer.zero_grad()
            #self.agent.zero_grad()
            loss.backward()
            if not self.clip_grad=='max':
                nn_utils.clip_grad_norm_(self.agent.parameters(),self.clip_grad)
            self.agent_optimizer.step()

            self.writer.add_scalar(tag='cri_loss',scalar_value=critic_loss.item(),global_step=self.step)
            self.writer.add_scalar(tag='ac',scalar_value=actor_loss.item(),global_step=self.step)
            self.writer.add_scalar(tag='epo',scalar_value=epo_loss.item(),global_step=self.step)
            self.writer.add_scalar(tag='agent_loss',scalar_value=loss.item(),global_step=self.step)
            if j>0:
                self.writer.add_scalar(tag='ratio',scalar_value=ratio.mean().item(),global_step=self.step)
            grad_max = 0.0
            grad_means = 0.0
            grad_count = 0
            for p in self.agent.parameters():
                grad_max = max(grad_max, p.grad.abs().max().item())
                grad_means += (p.grad ** 2).mean().sqrt().item()
                grad_count += 1
            self.writer.add_scalar("grad_l2", grad_means / grad_count, self.step)
            self.writer.add_scalar("grad_max", grad_max, self.step)

            probs_new=self.agent(states)[0]
            kl=0
            for i in range(2):
                for p_old,p_new in zip(probs[i],probs_new[i]):
                    kl+=-((p_new-p_old)*FU.softmax(p_old,dim=1)).sum(dim=1).mean().item()
            self.writer.add_scalar(tag="kl", scalar_value=kl, global_step=self.step)
            self.step+=1
        
        probs=self.agent(states)[0]
        log_probs = self.calculate_probs_log(probs,actions)
        ratio = torch.exp(log_probs - old_log_probs)
        for r in ratio:
            if r>1+self.eps or r<1-self.eps:
                print('ratio_last_over:',r)
        self.writer.add_scalar(tag='ratio',scalar_value=ratio.mean().item(),global_step=self.step)
    
    def calculate_probs_log(self,out_puts,actions):
        probs_log=0
        for i in range(self.num_subtasks):
            t=torch.gather(FU.log_softmax(out_puts[0][i],dim=1),1,actions[0][:,[i]])
            probs_log+=t
        for i in range(self.num_subtasks-1):
            t=torch.gather(out_puts[1][i],1,actions[1][:,[i]])
            u=out_puts[1][i].exp().sum(axis=1,keepdim=True)
            s=torch.gather(out_puts[1][i].exp(),1,actions[1][:,:i]).sum(axis=1,keepdim=True)
            probs_log+=t-(u-s).log()
        return probs_log

class PPO_softmax:
    ''' PPO算法,采用截断方式 '''
    def __init__(self,input_shape:tuple,num_subtasks,weights,gamma,device,clip_grad,lmbda,epochs,eps,beta,net,optim,cut,norm,std=[1,1],mean=[0,0],state_beta=0.9,reward_one=False,fm_eps=1e-8):
        self.writer=SummaryWriter(comment='PPO')
        self.step=0
        self.agent=net
        self.agent_optimizer=optim
        self.input_shape=input_shape
        self.num_processors=input_shape[0]
        self.num_subtasks=num_subtasks
        self.gamma=gamma
        self.weights=weights
        self.device=device
        self.num_subtasks=num_subtasks
        self.clip_grad=clip_grad
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device
        self.beta=beta
        self.cut=cut
        self.norm=self.F_norm(norm)
        self.std=std
        self.mean=mean
        self.reward_one=reward_one
        self.state_beta=state_beta
        self.fm_eps=fm_eps
    
    def F_norm(self,norm):
        def mean_std_all(state):
            u=state[0][:,:,:,:-self.num_subtasks]
            for i in u:
                i[:]=(i-i.mean())/i.std()
            for i in state[1]:
                i[:]=(i-i.mean())/i.std()
        
        def mean_std_single(state):
            u=state[0][:,:,:,:-self.num_subtasks]
            for i in u:
                for j in i[0]:
                    j[:]=(j-j.mean())/j.std()
            for i in state[1]:
                i[:]=(i-i.mean())/i.std()
        
        def low_high_all(state):
            u=state[0][:,:,:,:-self.num_subtasks]
            for i in u:
                i[:]=(i-torch.min(i))/(torch.max(i)-torch.min(i))
            for i in state[1]:
                i[:]=(i-torch.min(i))/(torch.max(i)-torch.min(i))
        
        def low_high_single(state):
            u=state[0][:,:,:,:-self.num_subtasks]
            for i in u:
                for j in i[0]:
                    j[:]=(j-torch.min(i))/(torch.max(j)-torch.min(j))
            for i in state[1]:
                i[:]=(i-torch.min(i))/(torch.max(i)-torch.min(i))
        
        def state_one(state):
            u=state[0][:,:,:,:-self.num_subtasks]
            u[:]=(u-self.mean[0])/(self.std[0]+self.fm_eps)
            state[1][:]=(state[1]-self.mean[1])/(self.std[1]+self.fm_eps)
        
        if norm=='msa':
            return mean_std_all
        if norm=='mss':
            return mean_std_single
        if norm=='lha':
            return low_high_all
        if norm=='lhs':
            return low_high_single
        if norm=='sto':
            return state_one
        return lambda x:x

    def take_action(self, state):
        F=lambda x:torch.tensor(x,dtype=torch.float).to(self.device)
        state=(F(state[0]),F(state[1]))
        self.norm(state)
        (probs_subtasks_orginal,probs_prior_orginal),_=self.agent(state)
        action_subtasks=[torch.distributions.Categorical(logits=x).sample().item()
            for x in probs_subtasks_orginal]

        action_prior=[]
        probs_prior_orginal=torch.cat(probs_prior_orginal,0)
        probs_prior_orginal=torch.tensor(
            np.concatenate((probs_prior_orginal.cpu().detach().numpy(),np.arange(self.num_subtasks).reshape(1,-1)),0),dtype=torch.float)
        for i in range(self.num_subtasks):
            x=torch.tensor(np.delete(probs_prior_orginal.numpy(),action_prior,1),dtype=torch.float)
            action_prior.append(x[-1,torch.distributions.Categorical(logits=x[i]).sample().item()].int().item())
        
        action=np.zeros((2,self.num_subtasks),dtype='int')
        action[0]=action_subtasks
        action[1]=action_prior
        return action

    def update(self, transition_dict):
        F=lambda x:torch.tensor(x,dtype=torch.float).to(self.device)
        states=tuple(F(np.concatenate([x[i] for x in transition_dict['states']],0)) for i in range(len(transition_dict['states'][0])))
        if self.norm=='sto':
            self.mean[0][:]=self.state_beta*self.mean[0]+(1-self.state_beta)*states[0][:,:,:,:-self.num_subtasks].mean(dim=0)
            self.std[0][:]=self.state_beta*self.std[0]+(1-self.state_beta)*states[0][:,:,:,:-self.num_subtasks].std(dim=0)
            self.mean[1][:]=self.state_beta*self.mean+(1-self.state_beta)*states[1].mean(dim=0)
            self.std[1][:]=self.state_beta*self.std+(1-self.state_beta)*states[1].std(dim=0)
        self.norm(states)
        actions=tuple(F(np.vstack([x[i] for x in transition_dict['actions']])).type(torch.int64) for i in range(len(transition_dict['actions'][0])))

        rewards=F(transition_dict['rewards']).view(-1,1)
        next_states=tuple(F(np.concatenate([x[i] for x in transition_dict['next_states']],0)) for i in range(len(transition_dict['states'][0])))
        self.norm(next_states)
        overs=F(transition_dict['overs']).view(-1,1)
        dones=transition_dict['dones']

        all_states=tuple(torch.concat((states[i][0:1],next_states[i])) for i in range(2))
        temp=self.agent(all_states)[1]
        td_target = rewards + self.gamma * temp[1:] * (1 - overs)
        
        td_delta = td_target - temp[:-1]
        #advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,td_delta.cpu()).to(self.device)
        advantage = rl_utils.compute_advantage_batch(self.gamma, self.lmbda,td_delta.cpu(),dones).to(self.device)
        if self.reward_one:
            advantage=(advantage)/(advantage.std()+self.fm_eps)
        #td_target=advantage.view(-1,1)+temp[:-1]                                               
        old_log_probs = self.calculate_probs_log(self.agent(states)[0],actions).detach()
        flag=0
        for j in range(self.epochs):
            #probs不是概率，而是神经网络的输出
            loss=0
            if not flag or not self.cut:
                probs=self.agent(states)[0]
                log_probs = self.calculate_probs_log(probs,actions)
                ratio = torch.exp(log_probs - old_log_probs)
                for r in ratio:
                    if r>1+self.eps or r<1-self.eps:
                        #print('ratio_pre_over:',r)
                        flag=1
                if self.cut and not self.agent.depart and flag:
                    break
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.eps,
                                    1 + self.eps) * advantage  # 截断
                actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
                
                s=0
                for i in range(2):
                    for prob in probs[0]:
                        s+=((FU.softmax(prob,dim=1))*FU.log_softmax(prob,dim=1)).sum(dim=1)
                epo_loss=self.beta*(s.mean())
                loss=actor_loss
                if self.beta:
                    loss+=epo_loss
            critic_loss = torch.mean(FU.mse_loss(self.agent(states)[1], td_target.detach()))
            loss+=self.weights*critic_loss
            self.agent_optimizer.zero_grad()
            #self.agent.zero_grad()
            loss.backward()
            #critic_loss.backward()
            #loss+=critic_loss
            if torch.isnan(loss)>0:
                print("here!")
            if not self.clip_grad=='max':
                nn_utils.clip_grad_norm_(self.agent.parameters(),self.clip_grad)
            self.agent_optimizer.step()
            self.writer.add_scalar(tag='cri_loss',scalar_value=critic_loss.item(),global_step=self.step)
            self.writer.add_scalar(tag='ac',scalar_value=actor_loss.item(),global_step=self.step)
            self.writer.add_scalar(tag='epo_loss',scalar_value=epo_loss.item(),global_step=self.step)
            if not self.agent.depart:
                self.writer.add_scalar(tag='agent_loss',scalar_value=loss.item(),global_step=self.step)
            if j>0:
                self.writer.add_scalar(tag='ratio',scalar_value=ratio.mean().item(),global_step=self.step)
            '''if ratio.mean().item()>3:
                print('ratio_too_big')'''
            grad_max = 0.0
            grad_means = 0.0
            grad_count = 0
            for p in self.agent.parameters():
                grad_max = max(grad_max, p.grad.abs().max().item())
                grad_means += (p.grad ** 2).mean().sqrt().item()
                grad_count += 1
            self.writer.add_scalar("grad_l2", grad_means / grad_count, self.step)
            self.writer.add_scalar("grad_max", grad_max, self.step)

            probs_new=self.agent(states)[0]
            kl=0
            for i in range(2):
                for p_old,p_new in zip(probs[i],probs_new[i]):
                    kl+=-((p_new-p_old)*FU.softmax(p_old,dim=1)).sum(dim=1).mean().item()
            self.writer.add_scalar(tag="kl", scalar_value=kl, global_step=self.step)
            self.step+=1
        
        probs=self.agent(states)[0]
        log_probs = self.calculate_probs_log(probs,actions)
        ratio = torch.exp(log_probs - old_log_probs)
        '''for r in ratio:
            if r>1+self.eps or r<1-self.eps:
                print('ratio_last_over:',r)'''
        self.writer.add_scalar(tag='ratio',scalar_value=ratio.mean().item(),global_step=self.step)
    
    def calculate_probs_log(self,out_puts,actions):
        probs_log=0
        for i in range(self.num_subtasks):
            t=torch.gather(FU.log_softmax(out_puts[0][i],dim=1),1,actions[0][:,[i]])
            probs_log+=t
        for i in range(self.num_subtasks-1):
            t=torch.gather(out_puts[1][i],1,actions[1][:,[i]])
            u=out_puts[1][i].exp().sum(axis=1,keepdim=True)
            s=torch.gather(out_puts[1][i].exp(),1,actions[1][:,:i]).sum(axis=1,keepdim=True)
            probs_log+=t-(u-s).log()
        return probs_log
import numpy as np
import torch
import collections
import ENV_AGENT
import CS_ENV
import random
from TEST import model_test
import time
from torch.utils.tensorboard import SummaryWriter

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) 

    def add(self, state, action, reward, next_state, done): 
        self.buffer.append((state, action, reward, next_state, done)) 

    def sample(self, batch_size): 
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done 

    def size(self): 
        return len(self.buffer)

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

def train_on_policy_agent(env, agent, num_episodes,max_steps,cycles,T_cycles=torch.inf,T_max=0,print_steps=True):
    writer=agent.writer
    frame_idx=0
    ts_time=time.time()
    return_list = []
    done=False
    state = env.reset()
    episode_return = 0
    i_episode=0
    k=0
    while i_episode < num_episodes:
        transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [], 'overs': []}
        step=0
        k+=1
        while not done and step<max_steps:
            step+=1
            frame_idx+=1
            action = agent.take_action(state)
            next_state, reward, done, over, _ = env.step(action)
            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['next_states'].append(next_state)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)
            transition_dict['overs'].append(over)
            state = next_state
            episode_return += reward
        if done:
            return_list.append(episode_return)
            writer.add_scalar(tag='return',scalar_value=episode_return,global_step=i_episode)
            i_episode+=1
            if i_episode % cycles == 0:
                print('speed:{}'.format(frame_idx/(time.time()-ts_time)))
                frame_idx,ts_time=0,time.time()
                test_reward=model_test(env,agent,3,recored=False)
                print('episode:{}, test_reward:{}'.format(i_episode,test_reward[0]))
                writer.add_scalar('test_reward',test_reward[0],i_episode)
                print('episode:{}, reward:{}'.format(i_episode,np.mean(return_list[-cycles:])))
            state = env.reset()
            done = False
            episode_return = 0
        if k%T_cycles==0 and max_steps<T_max:
            max_steps+=1
        if print_steps:
            print('env_steps:{}'.format(env.num_steps))
        agent.update(transition_dict)
    writer.close()
    return return_list

def train_on_policy_agent_batch(env:CS_ENV.CSENV, agent, num_episodes,max_steps,cycles,T_cycles=torch.inf,T_max=0,print_steps=False,pre_epochs=100,device='cuda',reps=1e-1,change_optim=10000,lr=1e-4,step_base=1,save_name=None):
    #states_orgin=[]
    done=False
    state = env.reset()
    t=0
    if pre_epochs<env.maxnum_episode:
        print('pre_epochs too little')
    for _ in range(pre_epochs):
        for pro in env.processors.pros:
            t+=pro.pro_dic['twe']+pro.pro_dic['ler']
        action = agent.take_action(state)
        next_state, _, done, _, _ = env.step(action)
        #states_orgin.append(state)
        state = next_state
        if done:
            state=env.reset()
    env.time_break=t/(3*env.time_steps*pre_epochs*env.num_pros)
    print('break_time:',env.time_break)
    '''F=lambda x:torch.tensor(x,dtype=torch.float).to(device)
    states=tuple(F(np.concatenate([x[i] for x in states_orgin],0)) for i in range(len(states_orgin[0])))
    mean,std=[],[]
    mean.append(states[0][:,:,:,:-agent.num_subtasks].mean(dim=0))
    std.append(states[0][:,:,:,:-agent.num_subtasks].std(dim=0))
    mean.append(states[1].mean(dim=0))
    std.append(states[1].std(dim=0))
    agent.mean,agent.std=mean,std'''

    writer=agent.writer
    frame_idx=0
    all_steps=0
    num_changes=0
    change_finish=False
    ts_time=time.time()
    return_list = []
    return_cycle_list=[]
    done=False
    state = env.reset()
    episode_return = 0
    i_episode=0
    a_max=-1e100
    kk=0
    while i_episode < num_episodes:
        transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [], 'overs': []}
        step=0
        kk+=1
        while step<max_steps:
            step+=1
            frame_idx+=1
            all_steps+=1
            action = agent.take_action(state)
            for i,pro in enumerate(env.processors.pros):
                writer.add_scalar(tag='pro_twe-'+str(i),scalar_value=pro.pro_dic['twe'],global_step=all_steps)
                writer.add_scalar(tag='pro_ler-'+str(i),scalar_value=pro.pro_dic['ler'],global_step=all_steps)
                writer.add_scalar(tag='pro_loc-'+str(i),scalar_value=(pro.cal_squard_d(pro.t))**0.5,global_step=all_steps)
                writer.add_scalar(tag='pro_num_tasks-'+str(i),scalar_value=(action[0]==i).sum(),global_step=all_steps)

            #print(state[1])
            #print(action[0])

            t1=[pro.pro_dic['twe'] for pro in env.processors.pros]
            next_state, reward, done, over, _ = env.step(action)
            t2=[pro.pro_dic['twe'] for pro in env.processors.pros]
            for a,b in zip(t1,t2):
                if a and a==b:
                    print('wrong_env')

            '''for pro in env.processors.pros:
                print('twe',pro.pro_dic['twe'])
            for pro in env.processors.pros:
                print('ler',pro.pro_dic['ler'])'''

            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['next_states'].append(next_state)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)
            transition_dict['overs'].append(over)
            state = next_state
            episode_return += reward
            writer.add_scalar(tag='step_rewards_all',scalar_value=reward,global_step=all_steps)
            for key,value in env.tarb_dic.items():
                writer.add_scalar(tag='step_rewards_'+key,scalar_value=value[-1]*env.lams[key[0]],global_step=all_steps)
        if done:
            return_list.append(episode_return)
            writer.add_scalar(tag='return',scalar_value=episode_return,global_step=i_episode)
            i_episode+=1
            if i_episode % cycles == 0:
                print('speed:{}'.format(frame_idx/(time.time()-ts_time)))
                frame_idx,ts_time=0,time.time()
                test_reward=model_test(env,agent,5,recored=False)
                print('episode:{}, test_reward:{}'.format(i_episode,test_reward[0]))
                writer.add_scalar('test_reward',test_reward[0],i_episode)
                a=np.mean(return_list[-cycles:])
                print('episode:{}, reward:{}'.format(i_episode,a))
                if test_reward[0]>a_max:
                    a_max=test_reward[0]
                    torch.save(agent.agent.state_dict(), '../data/CS_'+save_name+'_model_parameter.pkl')
                return_cycle_list.append(a)
                if len(return_cycle_list)>2 and (return_cycle_list[-1]-return_cycle_list[-2])>0 and (return_cycle_list[-1]-return_cycle_list[-2])/(return_cycle_list[-2]-return_cycle_list[-3])<reps:
                    agent.beta/=step_base
                    try:
                        agent.eps*=step_base
                    except:
                        pass
                    reps/=step_base
                    num_changes+=1
                    if not change_finish and num_changes==change_optim:
                        agent.optimizer=torch.optim.SGD(agent.agent.parameters(),lr=lr,momentum=0.9)
                        change_finish=True

            state = env.reset()
            #done = False
            episode_return = 0
        if kk%T_cycles==0 and max_steps<T_max:
            max_steps+=1
        if print_steps:
            print('env_steps:{}'.format(env.num_steps))
        agent.update(transition_dict)
    writer.close()
    return return_list

def print_state(env:ENV_AGENT.ENV_AGENT):
    print('pro_index: ',env.pro_index)
    print('speed: ',env.processor_speed)
    print('loc: ',env.processor_location)
    print('sub: \n',env.subtask_location)

def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    for i_episode in range(num_episodes):
        episode_return = 0
        state = env.reset()
        done = False
        while not done:
            action = agent.take_action(state)
            next_state, reward, done, _ = env.step(action)
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            episode_return += reward
            if replay_buffer.size() > minimal_size:
                b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                agent.update(transition_dict)
        return_list.append(episode_return)
        if (i_episode+1) % 10 == 0:
            print('episode:{}, reward:{}'.format(i_episode+1,np.mean(return_list[-10])))
    return return_list


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(np.concatenate(advantage_list,axis=0), dtype=torch.float)

def compute_advantage_batch(gamma, lmbda, td_delta,dones):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta,done in zip(td_delta[::-1],dones[::-1]):
        if done:
            advantage=0.0
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(np.concatenate(advantage_list,axis=0), dtype=torch.float)
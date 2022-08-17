import AC
import torch
import rl_utils
import time
from PUBLIC_ENV import *

load_net=False
agent=AC.ActorCritic_Double_softmax(W,maxnum_tasks,5,gamma,device,clip_grad=1e-1,beta=1e-4,n_steps=0,mode='gce',labda=0.95,
proc_name='0',optimizer=optim,net=net,norm='u',reward_one=False,state_beta=0.99,cri_type='u')
if load_net:
    agent.agent.load_state_dict(torch.load("../data/CS_AC_model_parameter_last.pkl"))
    agent.agent_optimizer=torch.optim.SGD(params=agent.agent.parameters(),lr=lr,momentum=0.9)

if __name__=='__main__':
    t_start=time.time()
    rl_utils.train_on_policy_agent_batch(env_c,agent,num_episodes,max_steps,cycles=10,T_cycles=50,T_max=0,print_steps=False,reps=1e-1,change_optim=30000,save_name='AC',pre_epochs=100)
    torch.save(agent.agent.state_dict(), "../data/CS_AC_model_parameter_last.pkl")
    agent.writer.close()
    public_test(agent)
    print(time.time()-t_start)

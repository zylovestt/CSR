import AC
import torch
import rl_utils
import time
from PUBLIC_ENV import *

#net.load_state_dict(torch.load("../data/CS_AC_model_parameter.pkl"))
agent=AC.ActorCritic_Double_softmax(W,maxnum_tasks,5,gamma,device,
    clip_grad=1e-1,beta=1e-6,n_steps=0,mode='gce',labda=0.95,proc_name='0',optimizer=optim,net=net,norm='sto',reward_one=True)

if __name__=='__main__':
    t_start=time.time()
    rl_utils.train_on_policy_agent_batch(env_c,agent,num_episodes,max_steps,cycles=10,T_cycles=50,T_max=0,print_steps=False,reps=1e-1,change_optim=30000)
    torch.save(agent.agent.state_dict(), "../data/CS_AC_model_parameter.pkl")
    agent.writer.close()
    public_test(agent)
    print(time.time()-t_start)

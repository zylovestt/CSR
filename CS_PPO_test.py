import torch
import rl_utils
import PPO 
import time
from PUBLIC_ENV import *

lmbda = 0.95
epochs = 3
eps = 0.2

net.load_state_dict(torch.load("../data/CS_PPO_model_parameter.pkl")) 
#print(1/(maxnum_tasks*math.log(maxnum_tasks*num_pros)))ws
#beta=1e-2/(maxnum_tasks*math.log(maxnum_tasks*num_pros))
#print('beta:',beta)
beta=1e-5
agent = PPO.PPO_softmax(W,maxnum_tasks,weights=5,gamma=gamma,device=device,clip_grad='max',lmbda=lmbda,epochs=epochs,eps=eps,beta=beta,net=net,optim=optim,cut=False,norm='sto',reward_one=True)

if __name__=='__main__': 
    t_start=time.time()
    rl_utils.train_on_policy_agent_batch(env_c, agent, num_episodes,max_steps,cycles=10,T_cycles=50,T_max=0,print_steps=False,reps=1e-1,change_optim=30000)
    torch.save(agent.agent.state_dict(), "../data/CS_PPO_model_parameter.pkl")
    agent.writer.close()
    public_test(agent)
    print(time.time()-t_start)
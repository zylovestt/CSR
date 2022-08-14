from cProfile import label
from pickle import TRUE
import numpy as np
import CS_ENV
import torch
import rl_utils
import PPO 
import math
from TEST import model_test
import AGENT_NET
import time

np.random.seed(1)
torch.manual_seed(0)
lr = 1e-4
num_episodes = 100
gamma = 0.98
num_pros=10
maxnum_tasks=10
env_steps=100
max_steps=10
tanh=True
device = torch.device("cuda")
iseed=1
tseed=[np.random.randint(0,1000) for _ in range(1000)]
seed=[np.random.randint(0,1000) for _ in range(1000)]
#tseed=seed
'''F,Q,er,econs,rcons,B,p,g,d,w,alpha,twe,ler'''
np.set_printoptions(2)
pro_dic={}
pro_dic['F']=(0.7,0.99)
pro_dic['Q']=(0.7,0.99)
pro_dic['er']=(0.1,0.9)
pro_dic['econs']=(0.1,0.9)
pro_dic['rcons']=(0.1,0.9)
pro_dic['B']=(0.1,0.9)
pro_dic['p']=(0.1,0.9)
pro_dic['g']=(0.1,0.9)
def fx():
    h=np.random.random()
    #h=1
    def g(x):
        t=100*h*math.sin(h*x/10)+10
        return t
    return g
def fy():
    h=np.random.random()
    #h=1
    def g(x):
        t=50*h*math.sin(h*x/5)-10
        return t
    return g
pro_dic['x']=fx
pro_dic['y']=fy
pro_dic['w']=1
pro_dic['alpha']=2
pro_dic['twe']=(0,0)
pro_dic['ler']=(0,0)
pro_dics=[CS_ENV.fpro_config(pro_dic) for _ in range(num_pros)]
task_dic={}
task_dic['ez']=(0.5,1)
task_dic['rz']=(0.5,1)
task_dics=[CS_ENV.ftask_config(task_dic) for _ in range(maxnum_tasks)]
job_d={}
job_d['time']=(0.1,0.3)
job_d['womiga']=(0.5,1)
job_d['sigma']=(0.5,1)
job_d['num']=(1,maxnum_tasks)
job_dic=CS_ENV.fjob_config(job_d)
loc_config=CS_ENV.floc_config()
z=['Q','T','C','F']
lams={}
lams['T']=1*1e-1
lams['Q']=-1*1e-1
lams['F']=-1*1e-1
lams['C']=1*1e-1
bases={x:1 for x in z}

env_c=CS_ENV.CSENV(pro_dics,maxnum_tasks,task_dics,
        job_dic,loc_config,lams,env_steps,bases,bases,seed,tseed,reset_states=False,cut_states=True,init_seed=iseed,reset_step=False)
    
state=env_c.reset()
W=(state[0].shape,state[1].shape)
r_agent=CS_ENV.RANDOM_AGENT(maxnum_tasks)
model_test(env_c,r_agent,10)

for key in env_c.bases:
    env_c.tar_dic[key].sort()
    g=np.array(env_c.tar_dic[key],dtype='float32')
    l=len(g)
    env_c.bases[key]=g[l//2]
    env_c.bases_fm[key]=g[l*3//4]-g[l//4]+1
for key in env_c.bases:
    env_c.tar_dic[key]=[]
    env_c.tarb_dic[key+'b']=[]
bases_fm=env_c.bases_fm

lmbda = 0.95
epochs = 3
eps = 0.2

net=AGENT_NET.DoubleNet_softmax_simple(W,maxnum_tasks,tanh,depart=True).to(device)
#net.load_state_dict(torch.load("../data/CS_PPO_model_parameter.pkl"))
optim=torch.optim.NAdam(net.parameters(),lr=lr,eps=1e-8)
print(1/(maxnum_tasks*math.log(maxnum_tasks*num_pros)))
agent = PPO.PPO_softmax(W,maxnum_tasks,1e-1,  gamma, device,'max',lmbda,epochs, eps,1/(maxnum_tasks*math.log(maxnum_tasks*num_pros)),net,optim,cut=False,norm='mss')

if __name__=='__main__':
    t_start=time.time()
    return_list = rl_utils.train_on_policy_agent_batch(env_c, agent, num_episodes,max_steps,cycles=10,T_cycles=50,T_max=0,print_steps=False)
    #return_list = rl_utils.train_on_policy_agent(env_c, agent, num_episodes,max_steps,cycles=10,T_cycles=50,T_max=0,print_steps=False)
    torch.save(agent.agent.state_dict(), "../data/CS_PPO_model_parameter.pkl")
    agent.writer.close()
 
    #env_c.reset_step=False
    #env_c.reset_states=True
    #env_c.reset_step=False
    tl_0,step_rewards0=model_test(env_c,agent,20)
    #tl_0=return_list[0]
    print('#'*20)
    env_c.cut_states=False
    r_agent=CS_ENV.OTHER_AGENT(CS_ENV.random_choice,maxnum_tasks)
    tl_1,step_rewards1=model_test(env_c,r_agent,20)
    print('#'*20)
    s_agent=CS_ENV.OTHER_AGENT(CS_ENV.short_twe_choice,maxnum_tasks)
    tl_2,step_rewards2=model_test(env_c,s_agent,20)
    print('agent_choice:{},r_choice:{},short_wait_choice:{}'.format(tl_0,tl_1,tl_2))
    print(time.time()-t_start)
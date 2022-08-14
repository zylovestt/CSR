import numpy as np
import math
import CS_ENV
import torch
from TEST import model_test
from torch.utils.tensorboard import SummaryWriter

np.random.seed(1)
torch.manual_seed(0)
lr = 1e-4
num_episodes = 10
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
        job_dic,loc_config,lams,env_steps,bases,bases,seed,tseed,reset_states=False,cut_states=False,init_seed=iseed,reset_step=False)
    
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

import time

def train_on_policy_agent_batch(env, agent, num_episodes,max_steps,cycles,T_cycles=torch.inf,T_max=0,print_steps=False):
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
                test_reward=model_test(env,agent,10,recored=False)
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
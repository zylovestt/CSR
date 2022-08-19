import numpy as np
import math
import CS_ENV
import torch
from TEST import model_test
import AGENT_NET

np.random.seed(1)
torch.manual_seed(0)

lr = 1*1e-4 #PPO:1e-5 AC:1e-4 maybe
num_episodes = 500
gamma = 0.95 #0.95
num_pros=5
maxnum_tasks=7
env_steps=50
max_steps=10
tanh=False #False
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#device=torch.device("cpu")
iseed=1
tseed=[np.random.randint(0,1000) for _ in range(1000)]
tseed_init=[np.random.randint(0,1000) for _ in range(1000)]
seed=[1]
seed_init=[2]
'''F,Q,er,econs,rcons,B,p,g,d,w,alpha,twe,ler'''
np.set_printoptions(2)
pro_dic={}
pro_dic['F']=(0.71,0.9)
pro_dic['Q']=(0.76,0.9)
pro_dic['er']=(0.73*100,0.9*100)
pro_dic['econs']=(0.74*10,0.9*10)
pro_dic['rcons']=(0.72*10,0.9*10)
pro_dic['B']=(0.73*100,0.9*100)
pro_dic['p']=(0.75,0.9)
pro_dic['g']=(0.78,0.9)
def fx():
    h=np.random.uniform(0.8,1)
    def g(x):
        t=10*h*math.sin(h*x/10)+10
        return t
    return g
def fy():
    h=np.random.uniform(0.65,1)
    def g(x):
        t=5*h*math.sin(h*x/5)-10
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
task_dic['ez']=(1*10,1*10)
task_dic['rz']=(1*1e-2,1e-2)
task_dics=[CS_ENV.ftask_config(task_dic) for _ in range(maxnum_tasks)]
job_d={}
job_d['time']=(0.7,1)
job_d['womiga']=(0.6,1)
job_d['sigma']=(0.8,1)
job_d['num']=(1,maxnum_tasks)
job_dic=CS_ENV.fjob_config(job_d)
loc_config=CS_ENV.floc_config()
z=['Q','T','C','F','B']
lams={}
lams['T']=1*1e-1
lams['Q']=-1*1e-1
lams['F']=-1*1e-1
lams['C']=1*1e-1
lams['B']=-0*1e-1
bases={x:0 for x in z}
bases_fm={x:0 for x in z}


env_c=CS_ENV.CSENV(pro_dics,maxnum_tasks,task_dics,
        job_dic,loc_config,lams,env_steps,bases,bases_fm,seed,tseed,
        change_prob=0.0,send_type=1,reward_one=False,state_one=False,
        set_break_time=False,state_beta=1,reward_one_type='std',
        train_init_seed=seed_init,test_init_seed=tseed_init)

'''r_agent=CS_ENV.RANDOM_AGENT(maxnum_tasks)
model_test(env_c,r_agent,1,recored=False)
for key in env_c.bases:
    env_c.tar_dic[key].sort()
    g=np.array(env_c.tar_dic[key],dtype='float32')
    l=len(g)
    env_c.bases[key]=g[l//2]
    env_c.bases_fm[key]=g[l*3//4]-g[l//4]+1
for key in env_c.bases:
    env_c.tar_dic[key]=[]
    env_c.tarb_dic[key+'b']=[]'''

'''if env_c.reset_states:
    env_c.cut_states=False
else:
    env_c.cut_states=True'''
state=env_c.reset() #use seed
W=(state[0].shape,state[1].shape)
net=AGENT_NET.DoubleNet_softmax_simple(W,maxnum_tasks,tanh,depart=True,fc=False).to(device)  #change
optim=torch.optim.NAdam(net.parameters(),lr=lr,eps=1e-8)

def public_test(agent):
    print('start_test'+'#'*60)
    tl_0=model_test(env_c,agent,10)
    print('#'*20)

    env_c.cut_states=False
    env_c.state_one=False
    print(env_c.test_seed[:10])
    r_agent=CS_ENV.OTHER_AGENT(CS_ENV.random_choice,maxnum_tasks)
    tl_1=model_test(env_c,r_agent,10)
    print('#'*20)
    s_agent=CS_ENV.OTHER_AGENT(CS_ENV.short_twe_choice,maxnum_tasks)
    tl_2=model_test(env_c,s_agent,10)
    print('agent_choice:{},r_choice:{},short_wait_choice:{}'.format(tl_0[0],tl_1[0],tl_2[0]))
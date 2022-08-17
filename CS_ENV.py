import math
import numpy as np
import pandas as pd
from collections import OrderedDict,defaultdict
from TEST import model_test

EPS=1e-8
rui=lambda u:(lambda:float(np.random.randint(u[0],u[1])))
ruf=lambda u:(lambda:float(np.random.uniform(u[0],u[1])))
PRO_STATE_NAMES=['twe', 'ler', 'er', 'econs', 'rcons', 'B', 'p', 'g',  'w', 'alpha','PF','Aq', 'x', 'y', 'vx','vy']

def fpro_config(dic):
    config={}
    i=['er','econs','rcons','B','p','g']
    f=['F','Q','twe','ler']
    for item in i:
        config[item]=ruf(dic[item]) #change
    for item in f:
        config[item]=ruf(dic[item])
    config['w']=float(dic['w'])
    config['alpha']=float(dic['alpha'])
    config['x']=dic['x']
    config['y']=dic['y']
    return config

def ftask_config(dic):
    config={}
    f=['rz','ez']
    for item in f:
        config[item]=ruf(dic[item])
    return config

def fjob_config(dic):
    config={}
    f=['time','womiga','sigma']
    for item in f:
        config[item]=ruf(dic[item])
    config['num']=lambda:int(np.random.randint(dic['num'][0],dic['num'][1]+1))
    return config

def floc_config():
    def generate(num_pros,maxnum_tasks):
        num_pro_choices=np.random.randint(1,num_pros+1,maxnum_tasks)
        loc=np.zeros((num_pros,maxnum_tasks),dtype='float')
        loc[:]=EPS
        for i in range(maxnum_tasks):
            num_pro_choice=num_pro_choices[i]
            pro_choice=np.random.choice(np.arange(num_pros,dtype='int'),num_pro_choice,False)
            loc[pro_choice,i]=1
        return loc
    return generate

class PROCESSOR:
    def __init__(self,config:dict):
        '''F,Q,er,econs,rcons,B,p,g,d,w,alpha,twe,ler'''
        self.pro_dic=OrderedDict()
        for k in config:
            if callable(config[k]) and not k=='Q':
                self.pro_dic[k]=config[k]()
            else:
                self.pro_dic[k]=config[k]
        self.Exe=0
        self.UExe=0
        self.cal_PF()
        self.sum_Aq=0
        self.Nk=0
        self.cal_Aq()
        self.t=0
    
    def cal_PF(self):
        self.PF=(self.Exe+1)/(self.Exe+self.UExe+2)
    
    def cal_Aq(self):
        self.Aq=(self.sum_Aq+1)/(self.Nk+2)
    
    def cal_squard_d(self,t):
        self.d=self.pro_dic['x'](t)**2+self.pro_dic['y'](t)**2
        return self.d
    
    def cal_v(self,t,tp):
        return self.pro_dic['x'](t)-self.pro_dic['x'](tp),self.pro_dic['y'](t)-self.pro_dic['y'](tp)
    
    def __call__(self,tin:float,task:dict,sigma:float):
        te=[x/self.pro_dic['er'] for x in task['ez']]
        tp=tin-self.t
        self.t=tin
        twe=self.pro_dic['twe']
        ler=self.pro_dic['ler']
        ler=min(max(ler+twe-tp,0),ler)
        twe=max(twe-tp,0)
        Q,finish=0,True
        for i in range(len(te)):
            if task['ez'][0]:
                if np.random.rand()>self.pro_dic['F']:
                    self.UExe+=1
                    finish=False
                else:
                    Q+=self.pro_dic['Q']()
                    self.Nk+=1
                    self.Exe+=1
            twe+=te[i]
            twr=max(ler-te[i],0)
            t=tin+twe+twr
            tr=self.cal_tr(task['rz'][i],t)
            ler=twr+tr
        self.pro_dic['twe']=twe
        self.pro_dic['ler']=ler
        if task['ez'][0]:
            self.cal_PF()
            Q*=sigma
            self.sum_Aq+=Q
            self.cal_Aq()
            if twe+ler==0:
                print('t_here!')
        return Q,twe+ler,np.sum(te)*self.pro_dic['econs']+np.sum(tr)*self.pro_dic['rcons'],finish
    
    def cal_tr(self,rz,t):
        r=self.pro_dic['B']*np.log2(
                1+self.pro_dic['p']*self.pro_dic['g']/
                (self.cal_squard_d(t)**(self.pro_dic['alpha']/2)
                *self.pro_dic['w']**2))
        return rz/r
    
    def cal_avg_rr(self):
        r=self.pro_dic['B']*np.log2(
                1+self.pro_dic['p']*self.pro_dic['g']/
                (2**(self.pro_dic['alpha']/2)
                *self.pro_dic['w']**2))
        return r

class PROCESSORS:
    def __init__(self,pro_configs:list,avg_reward=False):
        self.num_pros=len(pro_configs)
        self.pros=[PROCESSOR(pro_config) for pro_config in pro_configs]
        self.avg_reward=avg_reward
        if avg_reward:
            self.pros_er=1/np.array([pro.pro_dic['er'] for pro in self.pros]).sum().item()
            self.pros_rr=1/np.array([pro.cal_avg_rr() for pro in self.pros]).sum().item()
            self.pros_econs=np.array([pro.pro_dic['econs'] for pro in self.pros]).sum().item()
            self.pros_rcons=np.array([pro.pro_dic['rcons'] for pro in self.pros]).sum().item()
            self.pros_Q=1/np.array([pro.pro_dic['Q']() for pro in self.pros]).sum().item()    #see,mybe change
            self.pros_F=1/np.array([pro.pro_dic['F'] for pro in self.pros]).sum().item()

    def __call__(self,tin:float,tasks:dict,action:np.ndarray,womiga:float,sigma:float):
        for i,rz in enumerate(tasks['rz']):
            if not rz:
                break
        num_tasks=i if not rz else i+1
        tasks['ez']=tasks['ez'][:num_tasks]
        tasks['rz']=tasks['rz'][:num_tasks]
        act_list=[(i,action[0][i],action[1][i]) for i in range(num_tasks)]
        act_list=sorted(act_list,key=lambda x:x[-1])
        Q,task_time,cons,finish=0,0,0,True
        for i,pro in enumerate(self.pros):
            task={}
            task['ez'],task['rz']=[],[]
            for item in act_list:
                if item[1]==i:
                    task['ez'].append(tasks['ez'][item[0]])
                    task['rz'].append(tasks['rz'][item[0]])
            if len(task['ez']):
                Q1,task_time1,cons1,finish1=pro(tin,task,sigma)
                if not finish1:
                    finish=finish1
                Q+=Q1
                task_time=max(task_time,task_time1)
                cons+=cons1
            else:
                task['ez'],task['rz']=[0],[0]
                pro(tin,task,sigma)
        if task_time==0:
            print('ta_here!')
        dic={}
        dic['Q'],dic['T'],dic['C'],dic['F']=Q,task_time*womiga,cons,finish
        if self.avg_reward:
            tasks_ez=np.array(tasks['ez']).sum()
            tasks_rz=np.array(tasks['rz']).sum()
            dic['Q']*=self.pros_Q
            dic['F']*=self.pros_F
            dic['C']*=(tasks_ez*self.pros_econs+tasks_rz*self.pros_rcons)
            dic['T']*=(tasks_ez*self.pros_er+tasks_rz*self.pros_rr)
        l_t=[]
        for pro in self.pros:
            l_t.append(pro.pro_dic['twe']+pro.pro_dic['ler'])
        dic['B']=np.array(l_t).std()
        return dic

class JOB:
    def __init__(self,maxnum_tasks:int,task_configs:list,job_config:dict):
        self.maxnum_tasks=maxnum_tasks
        self.task_configs=task_configs
        self.job_config=job_config
        self.job_index=0
        self.tin=0

    def __call__(self):
        self.job_index+=1
        num_tasks=self.job_config['num']()
        tasks=defaultdict(list)
        for i,config in enumerate(self.task_configs):
            if i<num_tasks:
                for k in config:
                    tasks[k].append(config[k]())
            else:
                for k in tasks:
                    tasks[k].append(0)
        self.tin+=self.job_config['time']()
        womiga=self.job_config['womiga']()
        sigma=self.job_config['sigma']()
        #print(tasks)
        return self.tin,tasks,womiga,sigma

class CSENV:
    name=0
    def __init__(self,pro_configs:list,maxnum_tasks:int,task_configs:list,job_config:dict,loc_config,
        lams:dict,maxnum_episode:int,bases:dict,bases_fm:dict,seed:list,test_seed:list,reset_states=False,
        cut_states=False,init_seed=1,reset_step=False,change_prob=0,send_type=1,time_steps=5,time_break=2,
        state_one=True,reward_one=True):
        '''lams:Q,T,C,F'''
        self.name+=1
        self.init_seed=init_seed
        self.pro_configs=pro_configs
        self.task_configs=task_configs
        self.job_config=job_config
        self.loc_config=loc_config
        self.maxnum_tasks=maxnum_tasks
        self.num_pros=len(pro_configs)
        self.lams=lams
        self.bases=bases
        self.bases_fm=bases_fm
        self.tar_dic,self.tarb_dic={},{}
        self.tar_dic['Q']=[]
        self.tar_dic['T']=[]
        self.tar_dic['C']=[]
        self.tar_dic['F']=[]
        self.tar_dic['B']=[]
        self.tarb_dic['Qb']=[]
        self.tarb_dic['Tb']=[]
        self.tarb_dic['Cb']=[]
        self.tarb_dic['Fb']=[]
        self.tarb_dic['Bb']=[]
        self.sum_tar=[]
        self.sum_test_tar=[]
        self.sum_tarb=[]
        self.maxnum_episode=maxnum_episode
        self.set_random_const=False
        self.train=True
        self.seed=seed
        self.seedid=0
        self.test_seed=test_seed
        self.test_id=0
        np.random.seed(init_seed)
        self.processors=PROCESSORS(self.pro_configs)
        self.job=JOB(self.maxnum_tasks,self.task_configs,self.job_config)
        self.reset_states=reset_states
        self.cut_states=cut_states
        self.reset_step=reset_step
        self.change_prob=change_prob
        self.send_type=send_type
        self.time_steps=time_steps
        self.time_break=time_break
        self.state_one=state_one
        self.reward_one=reward_one
    
    def send(self):
        if self.send_type==0:
            return self.send0()
        if self.send_type==1:
            return self.send1()
    
    def send0(self):
        self.tin,self.tasks,self.womiga,self.sigma=self.job()
        task_loc=self.loc_config(self.processors.num_pros,self.job.maxnum_tasks)
        self.task_loc=task_loc
        pro_status=[]
        if not self.cut_states:
            names=['twe', 'ler','er', 'econs', 'rcons', 'B', 'p', 'g',  'w', 'alpha']
        else:
            names=['twe', 'ler']
        for pro in self.processors.pros:
            #items=[value for k,value in pro.pro_dic.items() if not callable(value) and not k=='F']
            items=[pro.pro_dic[key] for key in names]
            items.extend([pro.PF,pro.Aq,pro.pro_dic['x'](self.job.tin),pro.pro_dic['y'](self.job.tin)])
            items.extend(pro.cal_v(self.job.tin,10))
            pro_status.append(items)
        pro_status=np.concatenate((np.array(pro_status),task_loc),1).reshape(1,1,self.processors.num_pros,-1)
        task_status=[]
        for item in self.tasks.values():
            task_status.extend(item)
        task_status.extend([self.womiga,self.sigma])
        task_status=np.array(task_status).reshape(1,-1)
        return pro_status,task_status
    
    def send1(self):
        self.tin,self.tasks,self.womiga,self.sigma=self.job()
        task_loc=self.loc_config(self.processors.num_pros,self.job.maxnum_tasks)
        self.task_loc=task_loc
        F=lambda:np.empty((self.num_pros,self.maxnum_tasks))
        ez_div_er=F()
        ez_mul_econs=F()
        rz_mul_rcons=F()
        rz_div_B=F()
        tr_t=np.empty((self.num_pros,self.time_steps))
        for i in range(self.num_pros):
            pro=self.processors.pros[i]
            ez_div_er[i]=np.array(self.tasks['ez'])/pro.pro_dic['er']
            ez_mul_econs[i]=np.array(self.tasks['ez'])/pro.pro_dic['econs']
            rz_mul_rcons[i]=np.array(self.tasks['rz'])/pro.pro_dic['rcons']
            rz_div_B[i]=np.array(self.tasks['rz'])/pro.pro_dic['B']
            for j in range(tr_t.shape[1]):
                tr_t[i,j]=pro.cal_tr(1,self.tin-self.time_break*j)

        pro_status=[]
        for pro in self.processors.pros:
            items=[pro.pro_dic['twe'],pro.pro_dic['ler'],pro.PF,pro.Aq]
            pro_status.append(items)
        pro_status=np.concatenate((np.array(pro_status),ez_div_er,ez_mul_econs,rz_mul_rcons,rz_div_B,tr_t,task_loc),1).reshape(1,1,self.processors.num_pros,-1)
        task_status=np.array([self.womiga,self.sigma]).reshape(1,-1)
        return pro_status,task_status

    def accept(self,action:np.ndarray):
        choice_prob=np.prod(self.task_loc[action[0],range(self.maxnum_tasks)])
        if choice_prob<0.5:
            print(str(self.name)+' wrong_choice')
        R=self.processors(self.tin,self.tasks,action,self.womiga,self.sigma)
        t,s,s_t=0,0,0
        for key,value in self.tar_dic.items():
            value.append(R[key])
            r=(self.bases[key]-R[key])/self.bases_fm[key]
            self.tarb_dic[key+'b'].append(r)
            s+=self.lams[key]*r
            if not key=='B':
                t+=self.lams[key]*R[key]
                s_t+=self.lams[key]*r
        self.sum_tar.append(t)
        self.sum_tarb.append(s)
        self.sum_test_tar.append(s_t)

    def set_test_mode(self):
        self.train=False
        self.test_id=0
    
    def set_train_mode(self):
        self.train=True

    def set_one(self):
        if self.reward_one or self.state_one:
            agent=RANDOM_AGENT(self.maxnum_tasks)

            if self.reward_one:
                for key in self.bases:
                    self.bases[key]=1
                    self.bases_fm[key]=1
            
            if self.state_one:
                self.states_mean=[0,0]
                self.states_std=[1,1]
                states_orgin=[]

            state=self.send()
            while not self.done:
                if self.state_one:
                    states_orgin.append(state)
                action = agent.take_action(state)
                next_state,*_ = self.step(action)
                state = next_state
            
            if self.state_one:
                states=tuple(np.concatenate([x[i] for x in states_orgin],0) for i in range(len(states_orgin[0])))
                self.states_mean[0]=(states[0][:,:,:,:-self.maxnum_tasks].mean(axis=0))
                self.states_std[0]=(states[0][:,:,:,:-self.maxnum_tasks].std(axis=0))
                self.states_mean[1]=(states[1].mean(axis=0))
                self.states_std[1]=(states[1].std(axis=0))

            if self.reward_one:
                for key in self.bases:
                    self.tar_dic[key].sort()
                    g=np.array(self.tar_dic[key],dtype='float32')
                    l=len(g)
                    self.bases[key]=g[l//2]
                    self.bases_fm[key]=g[l*3//4]-g[l//4]+1
                for key in self.bases:
                    self.tar_dic[key]=[]
                    self.tarb_dic[key+'b']=[]

    def reset(self):
        self.over=0
        self.done=0
        self.num_steps=0

        

        if self.reset_states:
            if self.train:
                np.random.seed(self.seed[self.seedid%len(self.seed)])  
            else:
                np.random.seed(self.test_seed[self.test_id%len(self.test_seed)])
            self.processors=PROCESSORS(self.pro_configs)
            self.job.job_index=0
            self.job.tin=0

            self.set_one()
            
            if self.train:
                np.random.seed(self.seed[self.seedid%len(self.seed)])
                self.seedid+=1
            else:
                np.random.seed(self.test_seed[self.test_id%len(self.test_seed)])
                self.test_id+=1 
            self.processors=PROCESSORS(self.pro_configs)
            self.job.job_index=0
            self.job.tin=0

        else:
            np.random.seed(self.init_seed)
            self.processors=PROCESSORS(self.pro_configs)
            self.job.job_index=0
            self.job.tin=0
            if self.train:
                np.random.seed(self.seed[self.seedid%len(self.seed)])
            else:
                np.random.seed(self.test_seed[self.test_id%len(self.test_seed)])

            self.set_one()

            np.random.seed(self.init_seed)
            self.processors=PROCESSORS(self.pro_configs)
            self.job.job_index=0
            self.job.tin=0
            if self.train:
                np.random.seed(self.seed[self.seedid%len(self.seed)])
                self.seedid+=1
            else:
                np.random.seed(self.test_seed[self.test_id%len(self.test_seed)])
                self.test_id+=1

        return self.send()
    
    def step(self,action:np.ndarray):
        self.accept(action)
        if self.train:
            reward=self.sum_tarb[-1]
        else:
            reward=self.sum_test_tar[-1]        #change
        self.num_steps+=1
        if self.num_steps>=self.maxnum_episode:
            self.done=1
            print(str(self.name)+' done')
        if self.reset_step:
            l=['er','econs','rcons','B','p','g']
            for pro,pro_conf in zip(self.processors.pros,self.pro_configs):
                for key in l:
                    pro.pro_dic[key]=pro_conf[key]()
        if self.change_prob:
            l=['er','econs','rcons','B','p','g']
            for pro,pro_conf in zip(self.processors.pros,self.pro_configs):
                if not pro.pro_dic['ler'] and not pro.pro_dic['twe'] and np.random.random()<self.change_prob and pro.Exe+pro.UExe:
                    print('change')
                    for key in l:
                        pro.pro_dic[key]=pro_conf[key]()
                        pro.Exe=0
                        pro.UExe=0
                        pro.cal_PF()
                        pro.sum_Aq=0
                        pro.Nk=0
                        pro.cal_Aq()
        return self.send(),reward,self.done,self.over,None

class RANDOM_AGENT:
    def __init__(self,maxnum_tasks):
        self.maxnum_tasks=maxnum_tasks
    
    def take_action(self,state):
        action=np.zeros((2,self.maxnum_tasks),dtype='int')
        action[1]=np.random.permutation(np.arange(self.maxnum_tasks))
        sub_loc=state[0][0,0,:,-self.maxnum_tasks:]
        num_pros=sub_loc.shape[0]
        for j,col in enumerate(sub_loc.T):
            action[0][j]=np.random.choice(np.arange(num_pros),p=col/col.sum())
        return action

class OTHER_AGENT:
    def __init__(self,choice,maxnum_tasks):
        self.choice=choice
        self.maxnum_tasks=maxnum_tasks
    
    def take_action(self,state):
        action=np.zeros((2,self.maxnum_tasks),dtype='int')
        action[1]=np.random.permutation(np.arange(self.maxnum_tasks))
        sub_loc=state[0][0,0,:,-self.maxnum_tasks:]
        pro_status=state[0][0,0,:,:-self.maxnum_tasks]
        action[0]=self.choice(sub_loc,pro_status)
        return action

def random_choice(sub_loc,_):
    num_pros=sub_loc.shape[0]
    return [np.random.choice(np.arange(num_pros),p=col/col.sum()) for col in sub_loc.T]

def short_twe_choice(sub_loc,pro_status):
    num_pros=sub_loc.shape[0]
    num_tasks=sub_loc.shape[1]
    l_pros=[(i,pro_status[i,PRO_STATE_NAMES.index('twe')]+pro_status[i,PRO_STATE_NAMES.index('ler')]) for i in range(num_pros)]
    l_pros.sort(key=lambda x:x[1])
    l_pros_index=[x[0] for x in l_pros]
    act=[-1 for _ in range(num_tasks)]
    task_visited=[0 for _ in range(num_tasks)]
    i=j=k=count=0
    while count<num_tasks:
        while task_visited[j]:
            j=(j+1)%num_tasks
        pro=l_pros_index[i]
        if sub_loc[pro,j]==1:
            count+=1
            act[j]=pro
            task_visited[j]=1
            i=(i+1)%num_pros
            k=0
        j=(j+1)%num_tasks
        k+=1
        if k>num_tasks-count:
            i=(i+1)%num_pros
            k=0
    return act
    
if __name__=='__main__':
    '''F,Q,er,econs,rcons,B,p,g,d,w,alpha,twe,ler'''
    #np.random.seed(1)
    np.set_printoptions(2)
    pro_dic={}
    pro_dic['F']=(0.9,0.99)
    pro_dic['Q']=(0.7,1)
    pro_dic['er']=(10,20)
    pro_dic['econs']=(1,5)
    pro_dic['rcons']=(1,5)
    pro_dic['B']=(10,20)
    pro_dic['p']=(10,20)
    pro_dic['g']=(10,20)
    def fx():
        h=np.random.random()
        def g(x):
            t=100*h*math.sin(h*x/10)+10
            return t
        return g
    def fy():
        h=np.random.random()
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
    num_pros=3
    pro_dics=[fpro_config(pro_dic) for _ in range(num_pros)]
    task_dic={}
    task_dic['ez']=(10,20)
    task_dic['rz']=(10,20)
    maxnum_tasks=4
    task_dics=[ftask_config(task_dic) for _ in range(maxnum_tasks)]
    job_d={}
    job_d['time']=(1,9)
    job_d['womiga']=(0.5,1)
    job_d['sigma']=(0.5,1)
    job_d['num']=(1,maxnum_tasks)
    job_dic=fjob_config(job_d)
    loc_config=floc_config()
    z=['Q','T','C','F']
    lams={x:1 for x in z}
    bases={x:1 for x in z}
    job_pros=CSENV(pro_dics,maxnum_tasks,task_dics,job_dic,loc_config,lams,100,bases,bases,[0],[0],cut_states=False)
    state=job_pros.reset()
    A=state[0].reshape(num_pros,-1)
    A=np.around(A,2)
    l=list(np.arange(maxnum_tasks))
    #ls=['er', 'econs', 'rcons', 'B', 'p', 'g', 'twe', 'ler', 'w', 'alpha','PF','Aq', 'x', 'y', 'vx','vy']
    names=['ez_div_er','ez_mul_econs','rz_mul_rcons','rz_div_B']
    ls=['twe', 'ler','PF','Aq']
    for name in names:
        for i in range(maxnum_tasks):
            ls.append(name+str(i))
    for i in range(job_pros.time_steps):
        ls.append('tr_t'+str(i))
    ls.extend(l)
    pd.DataFrame(A,columns=ls,index=['pro_1','pro_2','pro_3']).to_csv('sample.csv')
    rand_agent=RANDOM_AGENT(maxnum_tasks)
    done=0
    while not done:
        action=rand_agent.take_action(state)
        state,_,done,_,_=job_pros.step(action)
    print(job_pros.tar_dic)
    print(job_pros.sum_tar)
    print(job_pros.tarb_dic)
    print(job_pros.sum_tarb)
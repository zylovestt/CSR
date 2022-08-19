import numpy as np
import CS_ENV
import AC
import torch
from TEST import model_test
import torch.multiprocessing as mp
import time
import os
from PUBLIC_ENV import *

LR=1e-4
NUM_EPISODES=0
ENV_STEPS=100
MAX_STEPS=10
NUM_PROCESSINGS=4
NUM_ENVS=1
QUEUE_SIZE=NUM_PROCESSINGS
TRAIN_BATCH=2
BATCH_SIZE=NUM_ENVS
CYCLSES=10

f_worker=AC.ActorCritic_Double_softmax(W,maxnum_tasks,5,gamma,device,clip_grad=1e-1,beta=1e-4,n_steps=0,mode='gce',labda=0.95,
        proc_name='finally',optimizer=optim,net=net,norm='u',reward_one=False,state_beta=0.99,cri_type='u')
            
def data_func(proc_name,net,train_queue,id):
    np.random.seed(1)
    torch.manual_seed(0)
    '''F,Q,er,econs,rcons,B,p,g,d,w,alpha,twe,ler'''
    frame_idx=0
    ts_time=time.time()

    f_env=lambda x:CS_ENV.CSENV(pro_dics,maxnum_tasks,task_dics,
        job_dic,loc_config,lams,env_steps,bases,bases_fm,seed,tseed,reset_states=True,
        change_prob=0.0,send_type=1,reward_one=False,state_one=False,
        set_break_time=False,state_beta=1,reward_one_type='std',
        train_init_seed=seed_init,test_init_seed=tseed_init)  #need to change seed

    '''input_shape,num_subtasks,weights,gamma,device,clip_grad,beta,n_steps,mode,labda,proc_name'''
    worker=AC.ActorCritic_Double_softmax(W,maxnum_tasks,5,gamma,device,clip_grad=1e-1,beta=1e-4,n_steps=0,mode='gce',labda=0.95,
        proc_name='proc_name',optimizer=optim,net=net,norm='u',reward_one=False,state_beta=0.99,cri_type='u')

    worker.agent=net
    envs=[f_env(i) for i in range(NUM_ENVS)]
    for i,env in enumerate(envs):
        env.name=proc_name+' '+str(i)


    done=False
    state=[env.reset() for env in envs]
    episode_return=[0 for _ in range(NUM_ENVS)]
    return_list=[]
    i_episode=0
    grads_l=[]            #break time
    while i_episode<NUM_EPISODES:
        for i,env in enumerate(envs):
            transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [], 'overs': []}
            step=0
            while not done and step<MAX_STEPS:
                step+=1
                frame_idx+=1
                action = worker.take_action(state[i])
                next_state, reward, done, over, _ = env.step(action)
                transition_dict['states'].append(state[i])
                transition_dict['actions'].append(action)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)
                transition_dict['overs'].append(over)
                state[i] = next_state
                episode_return[i] += reward
            if done:
                return_list.append(episode_return[i])
                worker.writer.add_scalar(tag='return',scalar_value=episode_return[i],global_step=i_episode)
                i_episode+=1
                if i_episode%CYCLSES==0:
                    s=frame_idx/(time.time()-ts_time)
                    print('{}: speed:{}'.format(proc_name,s))
                    worker.writer.add_scalar(tag='speed',scalar_value=s,global_step=i_episode)
                    frame_idx,ts_time=0,time.time()
                    test_reward=model_test(env,worker,10)
                    print('{}: episode:{} test_reward:{}'.format(proc_name,i_episode,test_reward[0]))
                    worker.writer.add_scalar('test_reward',test_reward[0],i_episode)
                    print('{}: episode:{} reward:{}'.format(proc_name,i_episode,np.mean(return_list[-CYCLSES:])))
                state[i] = env.reset()
                done = False
                episode_return[i] = 0
            grads_l.append(worker.update(transition_dict))
        if i_episode%BATCH_SIZE==0:
            for k in range(1,len(grads_l)):
                for grad0,gradk in zip(grads_l[0],grads_l[k]):
                    grad0+=gradk
            train_queue.put(grads_l[0])
            grads_l.clear()
    worker.writer.close()
    train_queue.put(None)

if __name__=='__main__':
    mp.set_start_method('spawn',force=True)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    '''F,Q,er,econs,rcons,B,p,g,d,w,alpha,twe,ler'''
    train_queue=mp.Queue(QUEUE_SIZE)
    #net=AGENT_NET.DoubleNet_softmax(W,MAXNUM_TASKS).to(DEVICE)
    
    #net.load_state_dict(torch.load("../data/CS_A3C_model_parameter.pkl"))
    net.share_memory()
    
    
    data_proc_list = []
    for proc_idx in range(NUM_PROCESSINGS):
        args=(str(proc_idx), net, train_queue,proc_idx)
        p = mp.Process(target=data_func,args=args)
        p.start()
        data_proc_list.append(p)

    batch = []
    step_idx = 0
    grad_buffer = None

    try:
        while True:
            train_entry = train_queue.get()
            if train_entry is None:
                break

            step_idx += 1

            if grad_buffer is None:
                grad_buffer = train_entry
            else:
                for tgt_grad, grad in zip(grad_buffer,
                                            train_entry):
                    tgt_grad += grad

            if step_idx % TRAIN_BATCH == 0:
                for param, grad in zip(net.parameters(),
                                        grad_buffer):
                    param.grad = torch.FloatTensor(grad/(TRAIN_BATCH*NUM_ENVS*BATCH_SIZE)).to(device)
                optim.step()
                grad_buffer = None
            if step_idx%100==0:
                torch.save(net.state_dict(), "../data/CS_A3C_model_parameter.pkl")
    finally:
        for p in data_proc_list:
            p.terminate()
            p.join()

    f_worker.agent=net
    public_test(f_worker)
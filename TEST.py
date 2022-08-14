import numpy as np
from torch.utils.tensorboard import SummaryWriter

def model_test(env,agent,num_episodes,recored=True):
    #env.train=False
    if recored:
        writer=SummaryWriter(comment='TEST')
    env.set_test_mode()
    return_list = []
    step_rewards=[[]]
    #num_subtasks=env.num_subtasks
    #np.random.seed(seed)
    for i_episode in range(num_episodes):
        episode_return = 0
        state = env.reset()
        done = 0
        k=0
        while not done:
            #print('sub_loc:\n{}'.format(state[0][0,0,:,-num_subtasks:]))
            action = agent.take_action(state)
            #print('action0:{}\naction1:{}'.format(action[0],action[1]))
            next_state, reward, done,*_ = env.step(action)
            state = next_state
            episode_return += reward
            step_rewards[-1].append(reward)
            if recored:
                writer.add_scalar(tag='step_rewards:'+str(i_episode),scalar_value=reward,global_step=k)
            k+=1
        return_list.append(episode_return)
        step_rewards.append([])
        '''if (i+1)%cycles==0:
            print('episode:{}, reward:{}'.format(i+1,np.mean(return_list[-cycles])))'''
    env.set_train_mode()
    #env.reset()
    return np.mean(return_list),step_rewards
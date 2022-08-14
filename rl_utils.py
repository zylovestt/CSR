import numpy as np
import torch
import collections
import ENV_AGENT
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
        while step<max_steps:
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
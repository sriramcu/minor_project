import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np
import shelve

# ~ MAX_ACCOUNT_BALANCE = 2147483647
# ~ MAX_NUM_SHARES = 2147483647
# ~ MAX_SHARE_PRICE = 5000
# ~ MAX_OPEN_POSITIONS = 5
# ~ MAX_STEPS = 20000
# ~ INITIAL_ACCOUNT_BALANCE = 10000

MAX_STEPS = 2000
INITIAL_AVERAGE_DELAY = 50
INITIAL_BUFFER_SIZE = 10
MIN_BUFFER_SIZE = 5
MAX_BUFFER_SIZE = 100 #packets
MIN_DELAY = 10
MAX_DELAY = 1000
MAX_REWARD = 400
DEQUEUE_RATE = 1 #Processing time of switch/ dequeue rate is 1 packet per second
ENQUEUE_RATE = 3
INITIAL_PACKET_DROP = 0.1
MAX_TRAVERSE_TIME = 10
MIN_REWARD = -100
REWARD_TOLERANCE = 0.90
PACKET_DROP_REWARD = 1000
DELAY_REWARD = 20


steps = []
delays = []
packetdrops = []
buffer_sizes = []
rewards = []

class SwitchEnv(gym.Env):
    """A switch environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(SwitchEnv, self).__init__()

        self.df = df
        self.reward_range = (0, MIN_REWARD)

        # Actions of the format Increase x%, Decrease x%, Hold, etc.
        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(2,3), dtype=np.float16) #2 arrays of 3 elements each

    def _next_observation(self):
        # Get the stock data points for the last 5 days and scale to between 0-1
        frame = np.array([
            self.df.loc[self.current_step: self.current_step +
                        2, 'traverse_time'].values / MAX_TRAVERSE_TIME,
        ]) #actual dataframe has sent and received timestamp also
        # ~ print("FRAMESHAPE")
        # ~ print(frame.shape)

        # Append additional data and scale each value to between 0-1
        params = np.array([[
            self.avg_delay/MAX_DELAY,
            self.packet_drop, #already probability, 0-1
            self.buffer_size/MAX_BUFFER_SIZE
        ]])
        # ~ print(params)
        # ~ print("PARAMSSHAPE")
        # ~ print(params.shape)
                
        obs = np.append(frame, params, axis=0)
        # ~ print("OBSSSHAPE")
        # ~ print(obs.shape)
        return obs

    def _take_action(self, action):


        action_type = action[0]
        amount = action[1]

        if action_type <= 1:
            # Increase buffer size by x%
            # ~ print("Buffer size increased by {}%".format(amount*100))
            tmp = self.buffer_size
            self.buffer_size = int(max(self.buffer_size + self.buffer_size * amount, MAX_BUFFER_SIZE))
            self.buffer_remaining = self.buffer_remaining + self.buffer_size-tmp

        elif action_type <= 2:
            # Decrease buffer size by x%
            # ~ print("Buffer size decreased by {}%".format(amount*100))
            tmp1 = self.buffer_size
            tmp2 = self.buffer_remaining
            delta1 = self.buffer_size * amount
            delta2= tmp1-MIN_BUFFER_SIZE
            delta3 = tmp2-0
            delta = int(min(delta1,delta2,delta3))
            
            self.buffer_size = tmp1-delta
            self.buffer_remaining = tmp2-delta
            
        else:
            pass
            # ~ print("Buffer size stayed the same")
            

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)

        self.current_step += 1

        if self.current_step > len(self.df.loc[:, 'traverse_time'].values) - 3:
            done = 1
            self.current_step = 0
        else:
            done = 0
        
        
        
        #current_delay,packet drop = calc_from dataframe and lookahead dequeue based on enqueue, dequeue rates
        #FIRST PROCESS PACKET RECEIVED AT BUFFER ENTRY BEFORE PROCESSING PACKET TO BE DEPARTED
        if self.buffer_remaining <= 0:
            self.packet_drop = (self.packet_drop * self.current_step + 1)/ (self.current_step +1)
            current_delay = self.avg_delay
        
        else:
            self.packet_drop = (self.packet_drop * self.current_step)/ (self.current_step +1)
            self.buffer_remaining -= 1
            current_delay = self.df.loc[self.current_step,'traverse_time'] + ENQUEUE_RATE + DEQUEUE_RATE * (self.buffer_size-self.buffer_remaining)
            key = self.current_step + DEQUEUE_RATE * (self.buffer_size-self.buffer_remaining)
            if key not in self.dequeue_times.keys():
                self.dequeue_times[key] = 1
            self.dequeue_times[key] += 1 
            
        
        
        self.buffer_remaining += self.dequeue_times[self.current_step] # we remove all packets that were shceduled to be dequeued for this timing
        self.dequeue_times[self.current_step] = 0
        self.avg_delay = (self.avg_delay*self.current_step+current_delay)/(self.current_step+1)
        
        #calculate and update delay and packet drop probability here
        #we will compare received time and current step (we are taking current step as a timestamp)
        
        delay_modifier = (self.current_step / MAX_STEPS)
        logical_reward = (self.prev_avg_delay-self.avg_delay) * DELAY_REWARD + (self.prev_packet_drop-self.packet_drop) * PACKET_DROP_REWARD
        reward = logical_reward * delay_modifier
        self.prev_avg_delay = self.avg_delay
        self.prev_packet_drop = self.packet_drop
        
        done = done or (reward <= (MIN_REWARD * REWARD_TOLERANCE)) or (reward >= (MAX_REWARD * REWARD_TOLERANCE))
        obs = self._next_observation()

        return obs, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.prev_avg_delay = INITIAL_AVERAGE_DELAY
        self.avg_delay = INITIAL_AVERAGE_DELAY
        self.prev_packet_drop = INITIAL_PACKET_DROP
        self.packet_drop = INITIAL_PACKET_DROP   #probability
        self.buffer_size = INITIAL_BUFFER_SIZE
        self.buffer_remaining = INITIAL_BUFFER_SIZE
        my_dict = {}
        for i in range(MAX_STEPS):
            my_dict[i] = 0        
        self.dequeue_times = my_dict

        # Set the current step to a random point within the data frame
        self.current_step = 0
        #assuming each step represents one second
        
        return self._next_observation()

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        #profit = self.net_worth - INITIAL_ACCOUNT_BALANCE
        sf = shelve.open('rewards.sf')
        reward = sf['reward']
        sf.close()
        
        sf = shelve.open("data.sf")
        steps = sf['steps']
        delay = sf['delay']
        packet_drop = sf['packet_drop']
        buffer_sizes = sf['buffer_sizes']
        rewards = sf['rewards']
        
        steps.append(self.current_step)
        delay.append(self.avg_delay)
        packet_drop.append(self.packet_drop)
        buffer_sizes.append(self.buffer_size)
        rewards.append(reward[0])
        
        sf['steps'] = steps
        sf['delay'] = delay
        sf['packet_drop'] = packet_drop
        sf['buffer_sizes'] = buffer_sizes
        sf['rewards'] = rewards
        sf.close()
        print("Step : {}".format(self.current_step))
        print("Average Delay : {}".format(self.avg_delay))
        print("Packet drop probability : {}".format(self.packet_drop))
        print("Buffer size : {}".format(self.buffer_size))
        print("Buffer remaining : {}".format(self.buffer_remaining))
        print("Reward : {}".format(reward))
        
        
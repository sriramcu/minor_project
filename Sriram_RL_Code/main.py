import gym
import json
import datetime as dt
import shelve
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
import matplotlib.pyplot as plt
from SwitchEnv import SwitchEnv

import pandas as pd


sf = shelve.open("data.sf")
sf['steps'] = []
sf['delay'] = []
sf['packet_drop'] = []
sf['buffer_sizes'] = []
sf['rewards'] = []
sf.close()
        
        
        
        
df = pd.read_csv('switch_dump.csv')
# ~ df = df.sort_values('Date')

# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: SwitchEnv(df)])

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=2000)

obs = env.reset()
sf = shelve.open('rewards.sf')
sf['reward'] = 0
sf.close()
for i in range(2000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    sf = shelve.open('rewards.sf')
    sf['reward'] = rewards
    sf.close()
    env.render()


sf = shelve.open("data.sf")
steps = sf['steps']
delay = sf['delay']
packet_drop = sf['packet_drop']
buffer_sizes = sf['buffer_sizes']
rewards = sf['rewards']

sf.close()


myvalue = 0
mylist = steps.copy()

# ~ last_idx = len(mylist) - mylist[::-1].index(myvalue) - 1 - 2000
# ~ print(last_idx)
# ~ print(len(mylist))

last_idx = 0
steps = steps[last_idx:last_idx+2000]
mi = steps.index(max(steps))
steps = steps[:mi]
delay = delay[:mi]
packet_drop = packet_drop[:mi]
buffer_sizes = buffer_sizes[:mi]
rewards = rewards[:mi]

print(packet_drop)
plt.plot(steps,delay, 'b', label='delay')
plt.plot(steps,packet_drop, 'y',label='packet_drop')
plt.plot(steps,buffer_sizes,'r', label='buffer size')
plt.plot(steps,rewards,'k', label='reward')
plt.xlabel('steps')
plt.ylabel('parameter value')
plt.legend()
plt.show()


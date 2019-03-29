import gym
import gym_2d_nav
import time
from armenv import ArmEnv

env = gym.make('Nav_2D-v0')
#env = ArmEnv()
for i in range(1,100):
    env.reset()
    env.render()
    time.sleep(0.1)


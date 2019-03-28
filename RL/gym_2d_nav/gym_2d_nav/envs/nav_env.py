import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
import math

class NavEnv2D(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    # Parameters
    self.min_theta = -0.00875  # -0.175 rad/s
    self.max_theta = 0.00875   # 0.175 rad/s
    self.min_r = 0             # 0 m/s
    self.max_r = 0.07          # 1.4 m/s
    self.max_grid_x = 20       # 20 m
    self.max_grid_y = 20       # 20 m
    self.dt = 0.05             #20 Hz | 1/20 sec

    #Initialisation
    self.init_pos = np.zeros(2)
    self.init_theta = 0
    self.goal_pos = np.zeros(2)
    self.goal_theta = 0

    # r, theta : Action vector
    actions_low = np.array([-self.max_r, self.min_theta])
    actions_high = np.array([self.max_r, self.max_theta])

    self.action_space = spaces.Box(low=actions_low, high=actions_high,dtype=np.float32)

    # goal (x,y) - robot position (x,y), goal_theta - present_theta, lin_vel, ang_vel : State vector
    min_state = np.array([-self.max_grid_x,-self.max_grid_y,-np.pi,0,-0.4])
    max_state = np.array([self.max_grid_x,self.max_grid_y,np.pi,1.4,0.4])

    self.observation_space = spaces.Box(low=min_state,high=max_state,dtype=np.float32)

    self.seed()

  def seed(self,seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def step(self, a):
    r = abs(a[0])
    self.v = r/self.dt
    self.w = a[1]/self.dt

    self.theta += a[1]
    if(self.theta < -np.pi):
      self.theta = self.theta%(2*np.pi)
    elif(self.theta > np.pi):
      self.theta = self.theta%(-2*np.pi)

    self.pos = self.pos + np.array(r*np.cos(a[1]),r*np.sin(a[1]))
    done = bool(np.linalg.norm(self.goal_pos-self.pos)<=0.05 and abs(self.goal_theta - self.theta) <= 0.05)

    reward = self.get_reward(a)
    if done:
      reward += 10

    return self.get_state(), reward, done, {}

  def reset(self):
    self.init_pos = self.np_random.uniform(low=0, high=20, size=2)
    self.init_theta = self.np_random.uniform(low=-np.pi, high=np.pi)
    self.goal_theta = self.np_random.uniform(low=-np.pi, high=np.pi)
    while True:
      self.goal_pos = self.np_random.uniform(low=0, high=20, size=2)
      dist = np.linalg.norm(self.goal_pos-self.init_pos)
      if( dist <= 2 and dist >= 0.5):
        break
    self.pos = self.init_pos.copy()
    self.theta = self.init_theta
    self.v = 0
    self.w = 0
    print(self.get_state())
    return self.get_state()

  def get_state(self):
    return np.concatenate((self.goal_pos - self.pos,
            self.goal_theta - self.theta,
            self.v,
            self.w), axis=None)

  def get_reward(self,a):
    dist_rew = -np.linalg.norm(self.goal_pos-self.init_pos)
    theta_rew = -abs(self.goal_theta - self.theta)
    ctrl_rew = -np.square(a).sum()

    reward = dist_rew + theta_rew + ctrl_rew
    return reward

  def render(self, mode='human', close=False):
    if mode is 'human':
        ... # pop up a window and render
    else:
        super(NavEnv2D, self).render(mode=mode) # just raise an exception
        #dadadadada
        #test comment

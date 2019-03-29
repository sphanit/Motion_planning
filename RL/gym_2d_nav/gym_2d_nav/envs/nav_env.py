import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
import math
import pyglet
pyglet.clock.set_fps_limit(10000)

class NavEnv2D(gym.Env):
  metadata = {'render.modes': ['human']}
  viewer = None
  viewer_xy = (400, 400)
  scale=20
  arm_info = np.zeros((2, 4))
  point_info = np.array([200, 100])
  point_l = 15
  mouse_in = np.array([False])

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
    done = bool(np.linalg.norm(self.goal_pos-self.pos)<=0.05)

    reward = self.get_reward(a)
    if done:
      reward += 10
    #print(a)
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
        if self.viewer is None:
            self.viewer = Viewer(*self.viewer_xy, self.arm_info, self.point_info, self.point_l, self.mouse_in, self.scale)
        self.viewer.render()
    else:
        super(NavEnv2D, self).render(mode=mode) # just raise an exception
        #dadadadada
        #test comment


class Viewer(pyglet.window.Window):
    color = {
        'background': [1]*3 + [1]
    }
    fps_display = pyglet.clock.ClockDisplay()
    bar_thc = 5

    def __init__(self, width, height, arm_info, point_info, point_l, mouse_in, scale):
        super(Viewer, self).__init__(width, height, resizable=False, caption='nav', vsync=False)  # vsync=False to not use the monitor FPS
        self.set_location(x=80, y=10)
        pyglet.gl.glClearColor(*self.color['background'])

        self.arm_info = arm_info
        self.point_info = point_info
        self.mouse_in = mouse_in
        self.point_l = point_l

        self.center_coord = np.array((min(width, height)/2, ) * 2)
        self.batch = pyglet.graphics.Batch()

        arm1_box, arm2_box, point_box = [0]*8, [0]*8, [0]*8
        c1, c2, c3 = (249, 86, 86)*4, (86, 109, 249)*4, (249, 39, 65)*4
        self.point = self.batch.add(4, pyglet.gl.GL_QUADS, None, ('v2f', point_box), ('c3B', c2))
        self.arm1 = self.batch.add(4, pyglet.gl.GL_QUADS, None, ('v2f', arm1_box), ('c3B', c1))
        self.arm2 = self.batch.add(4, pyglet.gl.GL_QUADS, None, ('v2f', arm2_box), ('c3B', c1))

    def render(self):
        pyglet.clock.tick()
        self._update_arm()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def on_draw(self):
        self.clear()
        self.batch.draw()
        # self.fps_display.draw()

    def _update_arm(self):
        point_l = self.point_l
        point_box = (self.point_info[0] - point_l, self.point_info[1] - point_l,
                     self.point_info[0] + point_l, self.point_info[1] - point_l,
                     self.point_info[0] + point_l, self.point_info[1] + point_l,
                     self.point_info[0] - point_l, self.point_info[1] + point_l)
        self.point.vertices = point_box

        arm1_coord = (*self.center_coord, *(self.arm_info[0, 2:4]))  # (x0, y0, x1, y1)
        arm2_coord = (*(self.arm_info[0, 2:4]), *(self.arm_info[1, 2:4]))  # (x1, y1, x2, y2)
        arm1_thick_rad = np.pi / 2 - self.arm_info[0, 1]
        x01, y01 = arm1_coord[0] - np.cos(arm1_thick_rad) * self.bar_thc, arm1_coord[1] + np.sin(
            arm1_thick_rad) * self.bar_thc
        x02, y02 = arm1_coord[0] + np.cos(arm1_thick_rad) * self.bar_thc, arm1_coord[1] - np.sin(
            arm1_thick_rad) * self.bar_thc
        x11, y11 = arm1_coord[2] + np.cos(arm1_thick_rad) * self.bar_thc, arm1_coord[3] - np.sin(
            arm1_thick_rad) * self.bar_thc
        x12, y12 = arm1_coord[2] - np.cos(arm1_thick_rad) * self.bar_thc, arm1_coord[3] + np.sin(
            arm1_thick_rad) * self.bar_thc
        arm1_box = (x01, y01, x02, y02, x11, y11, x12, y12)
        arm2_thick_rad = np.pi / 2 - self.arm_info[1, 1]
        x11_, y11_ = arm2_coord[0] + np.cos(arm2_thick_rad) * self.bar_thc, arm2_coord[1] - np.sin(
            arm2_thick_rad) * self.bar_thc
        x12_, y12_ = arm2_coord[0] - np.cos(arm2_thick_rad) * self.bar_thc, arm2_coord[1] + np.sin(
            arm2_thick_rad) * self.bar_thc
        x21, y21 = arm2_coord[2] - np.cos(arm2_thick_rad) * self.bar_thc, arm2_coord[3] + np.sin(
            arm2_thick_rad) * self.bar_thc
        x22, y22 = arm2_coord[2] + np.cos(arm2_thick_rad) * self.bar_thc, arm2_coord[3] - np.sin(
            arm2_thick_rad) * self.bar_thc
        arm2_box = (x11_, y11_, x12_, y12_, x21, y21, x22, y22)
        self.arm1.vertices = arm1_box
        self.arm2.vertices = arm2_box

    def on_key_press(self, symbol, modifiers):
        if symbol == pyglet.window.key.UP:
            self.arm_info[0, 1] += .1
            print(self.arm_info[:, 2:4] - self.point_info)
        elif symbol == pyglet.window.key.DOWN:
            self.arm_info[0, 1] -= .1
            print(self.arm_info[:, 2:4] - self.point_info)
        elif symbol == pyglet.window.key.LEFT:
            self.arm_info[1, 1] += .1
            print(self.arm_info[:, 2:4] - self.point_info)
        elif symbol == pyglet.window.key.RIGHT:
            self.arm_info[1, 1] -= .1
            print(self.arm_info[:, 2:4] - self.point_info)
        elif symbol == pyglet.window.key.Q:
            pyglet.clock.set_fps_limit(1000)
        elif symbol == pyglet.window.key.A:
            pyglet.clock.set_fps_limit(30)

    def on_mouse_motion(self, x, y, dx, dy):
        self.point_info[:] = [x, y]

    def on_mouse_enter(self, x, y):
        self.mouse_in[0] = True

    def on_mouse_leave(self, x, y):
        self.mouse_in[0] = False

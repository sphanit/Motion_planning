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
  viewer_xy = (500, 500)
  scale=20
  length = 1
  bredth = 0.6
  goal_pos = np.zeros(2)
  goal_theta = 0
  pos = np.zeros(2)
  theta = 0

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

    return self.get_state(), reward, done, {}

  def reset(self):
    self.init_pos = self.np_random.uniform(low=0, high=20, size=2)
    self.init_theta = self.np_random.uniform(low=-np.pi, high=np.pi)
    self.goal_theta = self.np_random.uniform(low=-np.pi, high=np.pi)
    while True:
      self.goal_pos = self.np_random.uniform(low=0, high=20, size=2)
      dist = np.linalg.norm(self.goal_pos-self.init_pos)
      if( dist <= 10 and dist >= 0.5):
        break
    self.pos = self.init_pos.copy()
    self.theta = self.init_theta
    self.v = 0
    self.w = 0
    print(self.pos)
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

  def set_fps(self, fps=30):
        pyglet.clock.set_fps_limit(fps)

  def render(self, mode='human', close=False):
    if mode is 'human':
        if self.viewer is None:
            self.viewer = Viewer(*self.viewer_xy, self.length, self.bredth, self.scale, self.pos, self.theta, self.goal_pos, self.goal_theta, self)
        self.viewer.render()
    else:
        super(NavEnv2D, self).render(mode=mode) # just raise an exception


class Viewer(pyglet.window.Window):
    color = {
        'background': [1]*3 + [1]
    }
    fps_display = pyglet.clock.ClockDisplay()

    def __init__(self, width, height, length, bredth, scale, pos, theta, goal_pos, goal_theta, nav2d):
        super(Viewer, self).__init__(width, height, resizable=False, caption='nav', vsync=False)  # vsync=False to not use the monitor FPS
        self.set_location(x=80, y=10)
        pyglet.gl.glClearColor(*self.color['background'])
        self.nav2d = nav2d
        self.scale = self.nav2d.scale
        self.length = self.nav2d.length * self.scale
        self.bredth = self.nav2d.bredth * self.scale

        self.batch = pyglet.graphics.Batch()

        robot_box, goal_box = [0]*10, [0]*10
        c1, c2, c3 = (249, 86, 86)*5, (86, 109, 249)*5, (249, 39, 65)*5
        self.robot = self.batch.add(5, pyglet.gl.GL_POLYGON, pyglet.graphics.OrderedGroup(0), ('v2f', robot_box), ('c3B', c2))
        self.goal  = self.batch.add(5, pyglet.gl.GL_POLYGON, None, ('v2f', goal_box), ('c3B', c1))

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
        leng = self.length
        brd = self.bredth
        print(self.nav2d.pos)


        rx1,ry1 = self.rotate_points(-leng/2, -brd/2,self.nav2d.theta) + self.nav2d.pos*self.scale
        rx2,ry2 = self.rotate_points(+leng/2, -brd/2,self.nav2d.theta) + self.nav2d.pos*self.scale
        rx3,ry3 = self.rotate_points(+leng/2 + 6, 0, self.nav2d.theta) + self.nav2d.pos*self.scale
        rx4,ry4 = self.rotate_points(+leng/2, +brd/2,self.nav2d.theta) + self.nav2d.pos*self.scale
        rx5,ry5 = self.rotate_points(-leng/2, +brd/2,self.nav2d.theta) + self.nav2d.pos*self.scale

        robot_box = (rx1,ry1, rx2,ry2, rx3,ry3, rx4,ry4, rx5,ry5)
        self.robot.vertices = robot_box


        gx1,gy1 = self.rotate_points( -leng/2,  -brd/2, self.nav2d.goal_theta) + self.nav2d.goal_pos*self.scale
        gx2,gy2 = self.rotate_points( +leng/2, -brd/2,self.nav2d.goal_theta)+ self.nav2d.goal_pos*self.scale
        gx3,gy3 = self.rotate_points( +leng/2 + 6, 0,self.nav2d.goal_theta)+ self.nav2d.goal_pos*self.scale
        gx4,gy4 = self.rotate_points( +leng/2,  +brd/2,self.nav2d.goal_theta)+ self.nav2d.goal_pos*self.scale
        gx5,gy5 = self.rotate_points( -leng/2,  +brd/2,self.nav2d.goal_theta)+ self.nav2d.goal_pos*self.scale

        goal_box = (gx1,gy1, gx2,gy2, gx3,gy3, gx4,gy4, gx5,gy5)
        self.goal.vertices = goal_box

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

    # def on_mouse_motion(self, x, y, dx, dy):
    #     self.point_info[:] = [x, y]
    #
    # def on_mouse_enter(self, x, y):
    #     self.mouse_in[0] = True
    #
    # def on_mouse_leave(self, x, y):
    #     self.mouse_in[0] = False

    def rotate_points(self,x,y,theta):
        x_new = x*np.cos(theta)-y*np.sin(theta)
        y_new = x*np.sin(theta)+y*np.cos(theta)
        return x_new, y_new

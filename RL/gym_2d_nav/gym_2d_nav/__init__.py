from gym.envs.registration import register

register(
    id='Nav_2D-v0',
    entry_point='gym_2d_nav.envs:NavEnv2D',
)


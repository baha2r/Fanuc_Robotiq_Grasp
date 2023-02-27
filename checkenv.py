from pickle import TRUE
from gym.utils.env_checker import check_env
from robotiqGymEnv import robotiqGymEnv

env = robotiqGymEnv(renders=False)

check_env(env) #skip_render_check=False
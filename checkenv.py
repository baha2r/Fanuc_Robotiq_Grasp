import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
import gymnasium
sys.modules["gym"] = gymnasium
import pybullet as p
from pickle import TRUE
from gym.utils.env_checker import check_env
from robotiqGymEnv import robotiqGymEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import SAC

env = robotiqGymEnv(renders=False)
model = SAC.load("./tensorboard/20230301-08:19PM_SAC_M1000/model")

# check_env(env) #skip_render_check=False

episode_reward, episode_length = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True, render=False)
print(episode_reward, episode_length)
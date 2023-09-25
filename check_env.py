import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
import gymnasium
sys.modules["gym"] = gymnasium
import pybullet as p
from pickle import TRUE
from gym.utils.env_checker import check_env
from robotiq_gym_env import robotiqGymEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import SAC

env = robotiqGymEnv()
# model = SAC.load("models/trained_agent/best_model.zip")

# obs, info = env.reset(seed=123)

check_env(env, skip_render_check=True) #skip_render_check=False

# episode_reward, episode_length = evaluate_policy(model, env, n_eval_episodes=20, deterministic=True)
# print(episode_reward, episode_length)
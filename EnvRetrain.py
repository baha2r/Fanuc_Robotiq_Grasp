import os
import gym
import pybullet_envs
from datetime import datetime
import sys
import gymnasium
sys.modules["gym"] = gymnasium

from stable_baselines3 import A2C, DDPG, PPO, TD3, SAC
# from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.env_util import make_vec_env
import multiprocessing as mp
from robotiqGymEnv import robotiqGymEnv


date = datetime. now(). strftime("%Y%m%d-%I:%M%p")
NAME = f"{date}_SAC"
numberofenv = 4

def make_my_env():
    env = robotiqGymEnv()
    return env
# Create and wrap the environment
env = robotiqGymEnv()
mass = env.targetmass
distance_threshold = env.distance_threshold
multienv = make_vec_env(lambda:make_my_env(), n_envs=numberofenv)



dir = "./tensorboard/20230312-06:53PM_SAC_M10000_0.04_WTS/model.zip"

model = SAC.load(dir, env=multienv)

# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
# print(mean_reward)

# env = DummyVecEnv([lambda: robotiqGymEnv(renders=True, isDiscrete=False)])
# # Automatically normalize the input features and reward
# env = VecNormalize(env, norm_obs=True, norm_reward=True)
evalenv = Monitor(env)

NAME = f"{date}_SAC_M{mass}_{distance_threshold}_WTS"

callback = EvalCallback(evalenv, best_model_save_path=f"./models/{NAME}/", log_path=f"./logs/{NAME}/", eval_freq=20000, deterministic=True, render=False)

model.learn(total_timesteps=1e7,tb_log_name="second_run", reset_num_timesteps=False, log_interval=10, callback=callback)
model.save(f"./tensorboard/{NAME}/model")



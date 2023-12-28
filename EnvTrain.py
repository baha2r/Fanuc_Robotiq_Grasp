import sys
import gymnasium
sys.modules["gym"] = gymnasium
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
from stable_baselines3 import A2C, DDPG, PPO, TD3, SAC, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from robotiqGymEnv import robotiqGymEnv
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes, StopTrainingOnNoModelImprovement, StopTrainingOnRewardThreshold, EveryNTimesteps, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
import multiprocessing as mp

date = datetime. now(). strftime("%Y%m%d-%I:%M%p")

numberofenv = 4

# callable function to create the environment
def make_my_env():
    env = robotiqGymEnv()
    return env
# Create and wrap the environment
env = robotiqGymEnv()
mass = env.targetmass
distance_threshold = env.distance_threshold
multienv = make_vec_env(lambda:make_my_env(), n_envs=numberofenv)

# The noise objects
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
evalenv = Monitor(env)
NAME = f"{date}_SAC_M{mass}_{distance_threshold}_39"
stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=10, min_evals=50, verbose=1)
callback = EvalCallback(evalenv, best_model_save_path=f"./models/{NAME}/", log_path=f"./logs/{NAME}/", eval_freq=20000, deterministic=True, render=False)#, callback_after_eval=stop_train_callback, n_eval_episodes=10)
model = SAC("MlpPolicy", multienv, verbose=1, tensorboard_log=f"./tensorboard/{NAME}/" , batch_size=1024, train_freq=numberofenv)#, action_noise=action_noise


model.learn(total_timesteps=3e7, log_interval=10, callback=callback)
model.save(f"./tensorboard/{NAME}/model")
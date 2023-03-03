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
from stable_baselines3.common.callbacks import BaseCallback
from robotiqGymEnv import robotiqGymEnv
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.callbacks import EveryNTimesteps
from stable_baselines3.common.callbacks import StopTrainingOnRewardThreshold
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes
from stable_baselines3.common.env_util import make_vec_env
import multiprocessing as mp

date = datetime. now(). strftime("%Y%m%d-%I:%M%p")
NAME = f"{date}_SAC_M10000_0.03"
numberofenv = 4

# callable function to create the environment
def make_my_env():
    env = robotiqGymEnv()
    return env
# Create and wrap the environment
env = robotiqGymEnv()
multienv = make_vec_env(lambda:make_my_env(), n_envs=numberofenv)

# The noise objects for TD3
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# env.reset()
# obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
callback = EvalCallback(env, best_model_save_path=f"./models/{NAME}/", log_path=f"./logs/{NAME}/", eval_freq=20000, deterministic=True, render=False)#, n_eval_episodes=10)
# model = SAC("MlpPolicy", multienv, learning_rate=1e-5, action_noise=action_noise, 
#             verbose=1, tensorboard_log=f"./tensorboard/{NAME}/" , batch_size=1024, 
#             policy_kwargs = dict(net_arch=dict(pi=[512, 1024, 512], qf=[512, 1024, 512]))) #, train_freq=numberofenv
model = SAC("MlpPolicy", multienv, action_noise=action_noise, verbose=1, tensorboard_log=f"./tensorboard/{NAME}/" , train_freq=numberofenv, batch_size=1024)

# add policy_kwargs to change the neural network architecture

model.learn(total_timesteps=1e7, log_interval=10, callback=callback)
model.save(f"./tensorboard/{NAME}/model")
# env = Monitor(env, log_dir)
# env = DummyVecEnv([lambda: env])

# model = TD3("MultiInputPolicy", env, verbose=1, tensorboard_log=f"./tensorboard/{NAME}/")
# model.learn(total_timesteps=10000000)
# model.save(f"./tensorboard/{NAME}/model")

# time_steps = 1e3
# results_plotter.plot_results([log_dir], time_steps, results_plotter.X_TIMESTEPS, "DDPG robotiq")
# plt.show()
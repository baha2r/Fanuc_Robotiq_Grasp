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
from robotiq_gym_env import robotiqGymEnv
import numpy as np

# date = datetime. now(). strftime("%Y%m%d-%I:%M%p")
# NAME = f"{date}_SAC"
# numberofenv = 4

# def make_my_env():
#     env = robotiqGymEnv()
#     env = Monitor(env)
#     return env
# Create and wrap the environment
env = robotiqGymEnv()
# multienv = make_vec_env(lambda:make_my_env(), n_envs=numberofenv)



dir = "models/trained_agent/best_model.zip"

model = SAC.load(dir, env=env)

# mean_reward, std_reward = evaluate_policy(model, Monitor(env), n_eval_episodes=100)
# print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

# Collect success values over 100 episodes
successes = []
rewards = []
for _ in range(100):
    obs = env.reset()
    done = False
    rew = 0
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        rew += reward
    # Assuming your environment has an attribute `success` that indicates whether the episode was successful
    success = env._is_success()
    successes.append(success)
    rewards.append(rew)

# Calculate mean and std of success rate
mean_success_rate = np.mean(successes)
std_success_rate = np.std(successes)
mean_reward = np.mean(rewards)
std_reward = np.std(rewards)

print(f"Mean success rate: {mean_success_rate:.2f} +/- {std_success_rate:.2f}")
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")


# env = DummyVecEnv([lambda: robotiqGymEnv(renders=True, isDiscrete=False)])
# # Automatically normalize the input features and reward
# env = VecNormalize(env, norm_obs=True, norm_reward=True)
# evalenv = Monitor(env)

# NAME = f"{date}_SAC_M{mass}_{distance_threshold}_WTS"

# callback = EvalCallback(evalenv, best_model_save_path=f"./models/{NAME}/", log_path=f"./logs/{NAME}/", eval_freq=20000, deterministic=True, render=False)

# model.learn(total_timesteps=1e7,tb_log_name="second_run", reset_num_timesteps=False, log_interval=10, callback=callback)
# model.save(f"./tensorboard/{NAME}/model")



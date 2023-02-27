import os
import gym
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime

from stable_baselines3 import A2C, DDPG, PPO, TD3, SAC
# from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from robotiqGymEnv import robotiqGymEnv
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.env_util import make_vec_env


def main():

  env = robotiqGymEnv(records=True, renders=True)
  # env = make_vec_env(lambda: env, n_envs=4)

  # rewa = evaluate_policy(model, env, deterministic=True, return_episode_rewards = True)
  dir = "models/20230207-12:41PM_SAC_rtop1/best_model.zip"
  # dir = "tensorboard/20230127-03:21PM_SAC/model.zip"
  model = SAC.load(dir)


  # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
  # print(mean_reward)

  dones = False
  obs = env.reset()
 
  while not dones:
      action, _states = model.predict(obs)
      obs, rewards, dones, info = env.step(action)

      # print(f"block position: {obs[19:22]}")
      # print(f"gripper psoition action: {action[0:3]}")
      # print(f"gripper orientation norm action: {np.linalg.norm(action[3:6])/ np.sqrt(3)}")
      # print(f"total normal force: {env._contactinfo()[4]}")
      # print(f"reward: {rewards}")
      # print(env._contactinfo()[4])
      # print(np.linalg.norm(env._p.getBaseVelocity(env._robotiq.robotiqUid)[1]))
      if dones:
          env.reset()
      env.render()

if __name__ == "__main__":
  main()
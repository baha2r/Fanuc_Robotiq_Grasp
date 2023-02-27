import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

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


def main():

  env = robotiqGymEnv(records=False, renders=True, isDiscrete=False)

  # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
  # print(mean_reward)

  dones = False
  obs = env.reset()
 
  while not dones:
    # if env._envStepCounter < 100:
    #   action = [0 , 0 , 0 , 0 , 0 , 0 , 0.1]
    # else:
    #   action = [0 , 0 , 0 , 1 , 0 , 1 , 1]
    # action = env.action_space.sample()
    action = [0 , 0 , 0 , 0 , 0 , 0 , 0]
    obs, rewards, dones, info = env.step(action)
    # print(len(env._robotiq.linkpos))
    # print(env._contactinfo()[4])
    # env.render()



if __name__ == "__main__":
  main()
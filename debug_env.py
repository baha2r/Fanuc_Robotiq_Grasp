import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
import gymnasium
sys.modules["gym"] = gymnasium
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
import pybullet as p

from stable_baselines3 import A2C, DDPG, PPO, TD3, SAC
# from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from robotiq_gym_env import robotiqGymEnv
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from scipy.spatial.transform import Rotation as R


def main():

  env = robotiqGymEnv(records=False, renders=True)

  # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
  # print(mean_reward)

  dones = False
  obs = env.reset()
 
  while not dones:
    # xtargetvel = p.getBaseVelocity(env.blockUid)[0][0]
    # ytargetvel = p.getBaseVelocity(env.blockUid)[0][1]
    # ztargetvel = p.getBaseVelocity(env.blockUid)[0][2]
    # print("xvel: ", xtargetvel)
    # print("yvel: ", ytargetvel)
    # print("zvel: ", ztargetvel)
    if env._env_step_counter < 300:
      action = [1 , 0 , 0 , 0 , 0 , 0 ]
    else:
      action = [0 , 0 , 0 , 0 , 0 , 0 ]
    # action = env.action_space.sample()
    # action = [0 , 0 , 0 , 0 , 0 , 0]
    obs, rewards, dones, info = env.step(action)
    # targetspeed = p.getBaseVelocity(env.blockUid)
    # print(p.getAABB(env.blockUid))
    # print((p.getBasePositionAndOrientation(env._robotiq.robotiqUid)[1]))
    # print(p.getBasePositionAndOrientation(env.blockUid)[0])
    # print(p.getBaseVelocity(env.blockUid)[0])
    print(p.getBaseVelocity(env._robotiq.robotiq_uid)[0])
    # print(targetspeed)
    # print(len(env._robotiq.linkpos))
    # print(env._contactinfo()[4])
    # env.render()


if __name__ == "__main__":
  main()
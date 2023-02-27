# import sys
# import gymnasium
# sys.modules["gym"] = gymnasium

from robotiqGymEnv import robotiqGymEnv
# from robotiq import robotiq
import numpy as np
import torch as th
import pybullet as p
from sklearn.preprocessing import normalize
import mujoco
from gym.spaces import Discrete, MultiDiscrete
import time
# from gym.utils.env_checker import check_env
import gymnasium
import multiprocessing as mp
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import A2C, DDPG, PPO, TD3, SAC, DQN

# env = robotiqGymEnv()
# print(env.action_space)
# print(len(env.observation_space.sample()))

env = gymnasium.make('AdroitHandDoor-v1', max_episode_steps=400)
print(env.action_space)
print(env.observation_space)
multienv = make_vec_env(lambda: env, n_envs=4)
model = SAC("MlpPolicy", multienv, verbose=1, batch_size=1024)



# check_env(env)
# robot = robotiq()

# x = np.load("./logs/20221121-02:54PM_TD3_3fgrip/evaluations.npz")
# print(x.files)
# for k in x.files:
#     print(k)


# fingerpose = p.getLinkStates(robot.robotiqUid, robot.thirdjointidx)
# result = [tup[0] for tup in fingerpose]
# print(result[0])
# print(np.array(p.getBasePositionAndOrientation(env._robotiq.robotiqUid)[0]))
# action  = env.action_space.sample()
# poscommand = np.array(action[0:3]) * 0.2
# oricommand = np.array(p.getQuaternionFromEuler(np.array(action[3:6]) * 0.5))

# vec1 = np.array([1,1,1])
# vec2 = np.array([0.5,0.5,0.5])
# vec3 = np.array([0,0,-1])

# diff = np.subtract(np.array(vec1), np.array(vec2))

# dotvec = np.dot(diff/np.linalg.norm(diff),vec3/np.linalg.norm(vec3))
# print(dotvec)
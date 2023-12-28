import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
import gymnasium
sys.modules["gym"] = gymnasium
import pybullet as p
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from robotiqGymEnv import robotiqGymEnv
import numpy as np
import csv
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from datetime import datetime

def make_my_env():
    env = robotiqGymEnv()
    return env

# Load your pre-trained SAC agent
agent = SAC.load("models/20230316-03:42PM_SAC/best_model.zip")
date = datetime. now(). strftime("%Y%m%d-%I:%M%p")
NAME = f"{date}"
# Initialize your environment
env = robotiqGymEnv(records=False, renders=False)
evalenv = Monitor(env)
multienv = make_vec_env(lambda:make_my_env(), n_envs=4)

# Define the curriculum
max_tilt_angle = np.pi/4  # rad
orientation_levels = np.linspace(0, max_tilt_angle, num=10)

class CurriculumLearningCallback(BaseCallback):
    def __init__(self, env, agent, orientation_levels, eval_freq, verbose=1):
        super(CurriculumLearningCallback, self).__init__(verbose)
        self.env = env
        self.agent = agent
        self.orientation_levels = iter(orientation_levels)
        self.current_level = next(self.orientation_levels)
        self.env.target_yaw = self.current_level
        self.eval_freq = eval_freq
        self.step_counter = 0

    def evaluate_agent(self, num_episodes=100):
        """ Evaluate the agent over a specified number of episodes """
        success_count = 0
        for _ in range(num_episodes):
            obs = self.env.reset()
            done = False
            while not done:
                action, _states = self.agent.predict(obs, deterministic=True)
                obs, reward, done, info = self.env.step(action)
                if done and info.get("is_success", False):
                    success_count += 1
        success_rate = success_count / num_episodes
        return success_rate

    def on_step(self) -> bool:
        self.step_counter += 1
        # Perform evaluation at specified frequency
        if self.step_counter % self.eval_freq == 0:
            success_rate = self.evaluate_agent()
            print(f"Evaluation at step {self.step_counter}, success rate: {success_rate * 100:.2f}%")

            # Advance level if success rate threshold is met
            if success_rate >= 0.80:
                print("Success rate threshold met, advancing level.")
                try:
                    self.current_level = next(self.orientation_levels)
                    self.env.target_yaw = self.current_level
                except StopIteration:
                    print("All levels completed")
                    return False  # Stop training

            self.step_counter = 0  # Reset the counter after evaluation

        return True

"""
class CurriculumLearningCallback(BaseCallback):
    def __init__(self, env, agent, orientation_levels, verbose=1):
        super(CurriculumLearningCallback, self).__init__(verbose)
        self.env = env
        self.agent = agent
        self.orientation_levels = iter(orientation_levels)
        self.current_level = next(self.orientation_levels)
        self.env.target_yaw = self.current_level

    def evaluate_agent(self, num_episodes=100):
        # Evaluate the agent over a specified number of episodes
        success_count = 0
        for _ in range(num_episodes):
            obs = self.env.reset()
            done = False
            while not done:
                action, _states = self.agent.predict(obs, deterministic=True)
                obs, reward, done, info = self.env.step(action)
                if done and info["is_success"]:
                    success_count += 1
        success_rate = success_count / num_episodes
        return success_rate

    def on_step(self) -> bool:
        # Check agent performance and adjust orientation level
        success_rate = self.evaluate_agent(num_episodes=100)
        if success_rate >= 0.80:
            print(f"Success rate of {success_rate * 100:.2f}% achieved, advancing level.")
            try:
                self.current_level = next(self.orientation_levels)
                self.env.target_yaw = self.current_level
            except StopIteration:
                print("All levels completed")
                return False  # Stop training
        return True
"""

# Initialize callback
curriculum_callback = CurriculumLearningCallback(env, agent, orientation_levels, eval_freq=20000)
eval_callback = EvalCallback(evalenv, best_model_save_path=f"./models/{NAME}/", log_path=f"./logs/{NAME}/", eval_freq=20000, deterministic=True, render=False, n_eval_episodes=10)
callback_list = CallbackList([curriculum_callback, eval_callback])

# Training with callback
agent.set_env(multienv)
agent.learn(total_timesteps=1000000, callback=callback_list, log_interval=10)
agent.save(f"./tensorboard/{NAME}/model")
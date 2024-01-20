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
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import Schedule
from typing import Callable


def make_my_env():
    env = robotiqGymEnv()
    return env

env = robotiqGymEnv(records=False, renders=False)
evalenv = Monitor(env)
multienv = make_vec_env(lambda:make_my_env(), n_envs=4)
# Load your pre-trained SAC agent
agent = SAC.load("tensorboard/20230316-03:42PM_SAC/model.zip", env=env)
print(f"The loaded_model has {agent.replay_buffer.size()} transitions in its buffer")

agent.collect_rollouts
date = datetime. now(). strftime("%Y%m%d-%I:%M%p")
NAME = f"{date}"
# Initialize your environment


# Define the curriculum
max_tilt_angle = np.pi/4  # rad
orientation_levels = np.linspace(0, max_tilt_angle, num=10)

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

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

    def _on_step(self) -> bool:
        self.step_counter += 1
        print(f"Replay Buffer Size: {len(self.agent.replay_buffer)}")

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

def populate_replay_buffer(agent, env, replay_buffer, num_steps):
    obs = env.reset()
    for _ in range(num_steps):
        action, _states = agent.predict(obs, deterministic=True)
        new_obs, reward, done, info = env.step(action)
        replay_buffer.add(obs, new_obs, action, reward, done, info)
        obs = new_obs
        if done:
            obs = env.reset()

# Initialize replay buffer (ensure the buffer size is appropriate for your environment)
replay_buffer_size = 100000  # Adjust this size based on your environment and requirements
replay_buffer = ReplayBuffer(replay_buffer_size, agent.policy.observation_space, agent.policy.action_space, device=agent.device)

# Populate the replay buffer with experiences from the pre-trained agent
populate_replay_buffer_steps = 100000  # Adjust this number based on your requirements
populate_replay_buffer(agent, env, replay_buffer, populate_replay_buffer_steps)

# Set the agent's replay buffer
agent.replay_buffer = replay_buffer

# Initialize callback
curriculum_callback = CurriculumLearningCallback(env, agent, orientation_levels, eval_freq=20000)
eval_callback = EvalCallback(evalenv, best_model_save_path=f"./models/{NAME}/", log_path=f"./logs/{NAME}/", eval_freq=20000, deterministic=True, render=False, n_eval_episodes=10)
callback_list = CallbackList([curriculum_callback, eval_callback])

# Training with callback
# agent.set_env(multienv)

# Define the new learning rate
new_learning_rate = 1e-6
# agent.policy.lr_schedule = new_learning_rate

# Update the learning rate in the optimizer
# for param_group in agent.policy.optimizer.param_groups:
#     param_group['lr'] = new_learning_rate

# If your SAC model has separate optimizers for actor and critic, update both
# if hasattr(agent.policy, 'actor_optimizer') and hasattr(agent.policy, 'critic_optimizer'):
#     for param_group in agent.policy.actor_optimizer.param_groups:
#         param_group['lr'] = new_learning_rate
#     for param_group in agent.policy.critic_optimizer.param_groups:
#         param_group['lr'] = new_learning_rate

# agent.learning_starts = 50000
agent.set_env(env)
agent.lr_schedule = linear_schedule(1e-6)
# agent.n_envs = 1
agent.target_update_interval = 4
agent.learn(total_timesteps=1e7, log_interval=10, callback=callback_list)
agent.save(f"./tensorboard/{NAME}/model")
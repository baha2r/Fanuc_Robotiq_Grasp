import gym
import numpy as np
from robotiqGymEnv import robotiqGymEnv


from stable_baselines3 import PPO, SAC
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.algorithms import bc

env = robotiqGymEnv()
venv = DummyVecEnv([lambda: RolloutInfoWrapper(env)])
modeldir = "./models/20230207-12:41PM_SAC_rtop1/best_model.zip"
expert = SAC.load(modeldir)

reward, _ = evaluate_policy(expert, env, 10)
print(f"eval policy rew: {reward}")

rng = np.random.default_rng()
rollouts = rollout.rollout(
    policy = expert,
    venv = venv,
    sample_until = rollout.make_sample_until(min_episodes=10),
    rng=rng,
)
transitions = rollout.flatten_trajectories(rollouts)

bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=transitions,
    rng=rng,
)

reward_before_training, _ = evaluate_policy(bc_trainer.policy, env, 10)
print(f"Reward before training: {reward_before_training}")

bc_trainer.train(n_epochs=2000)
reward_after_training, _ = evaluate_policy(bc_trainer.policy, env, 10)
print(f"Reward after training: {reward_after_training}")


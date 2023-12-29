import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
import gymnasium
sys.modules["gym"] = gymnasium
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from robotiqGymEnv import robotiqGymEnv
from stable_baselines3.common.monitor import Monitor


def main():

  env = robotiqGymEnv(records=False, renders=False)
  # env = make_vec_env(lambda: env, n_envs=4)

  # rewa = evaluate_policy(model, env, deterministic=True, return_episode_rewards = True)
  dir = "models/20230316-03:42PM_SAC/best_model.zip"
  # dir = "tensorboard/20230127-03:21PM_SAC/model.zip"
  model = SAC.load(dir)
  evalenv = Monitor(env)

  mean_reward, std_reward = evaluate_policy(model, evalenv, n_eval_episodes=10, deterministic=False, render=False)
  print(f"mean_reward: {mean_reward:.2f} +/- {std_reward}")

  # dones = False
  # obs = env.reset()
 
  # while not dones:
  #     action, _states = model.predict(obs)
  #     obs, rewards, dones, info = env.step(action)
  #     # env._r_topology()

  #     # print(f"block position: {obs[19:22]}")
  #     # print(f"gripper psoition action: {action[0:3]}")
  #     # print(f"gripper orientation norm action: {np.linalg.norm(action[3:6])/ np.sqrt(3)}")
  #     # print(f"total normal force: {env._contactinfo()[4]}")
  #     # print(f"reward: {rewards}")
  #     # print(env._contactinfo()[4])
  #     # print(np.linalg.norm(env._p.getBaseVelocity(env._robotiq.robotiqUid)[1]))
  #     if dones:
  #         env.reset()
      # env.render()

if __name__ == "__main__":
  main()
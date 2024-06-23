from robotiqGymEnv import robotiqGymEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import SAC

# Create the environment

# It will check your custom environment and output additional warnings if needed
# check_env(env)
model = SAC.load("models/20230316-03:42PM_SAC/best_model.zip")

counter = 0
for i in range(300,500):
    print(f"Episode {i}")
    env = robotiqGymEnv(store_data=True, data_path = f"SAC_trained_eps/test{i}.pkl")
    state = env.reset()
    done = 0
    while not done:
        action, _ = model.predict(state, deterministic=True)
        state, reward, done, info = env.step(action)
    # print(f"action: {action}, reward: {reward}, done: {done}, info: {info}")
    if info["is_success"] == True:
        counter += 1
# print(f"Episode {episode} finished")
print(f"Success rate: {counter/10}")
 
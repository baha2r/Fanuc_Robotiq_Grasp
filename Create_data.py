from robotiqGymEnv import robotiqGymEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import SAC

# Create the environment

# It will check your custom environment and output additional warnings if needed
# check_env(env)
model = SAC.load("models/20230316-03:42PM_SAC/best_model.zip")

counter = 0
for i in range(100):
    env = robotiqGymEnv(store_data=False)
    state = env.reset()
    done = 0
    while not done:
        action, _ = model.predict(state, deterministic=True)
        state, reward, done, info = env.step(action)
    print(f"Episode {i} with success {info} finished")
    if info["is_success"] == True:
        counter += 1
# print(f"Episode {episode} finished")
print(f"Success rate: {counter/100}")
 
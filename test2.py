from numpy import loadtxt
from matplotlib import pyplot as plt
import numpy as np


data = loadtxt('closestpsuccess.csv', delimiter=',')
data2 = loadtxt('closestcontact3.csv', delimiter=',')

reward = loadtxt('Rewardsuccess.csv', delimiter=',')
contact = loadtxt('Rewardcontact3.csv', delimiter=',')

plt.figure()
plt.plot(data, label="with tactile feedback")
plt.plot(data2, label="without tactile feedback")
plt.plot(np.zeros(len(data)), linestyle='--')
plt.title("Closest Point")
plt.xlabel("Timestep")
plt.ylabel("Closest Point")
plt.legend()
plt.show()
# savetxt('Reward.csv', rewards_, delimiter=' ')

plt.figure()
plt.plot(reward, label="with tactile feedback")
plt.plot(contact, label="without tactile feedback")
plt.plot(np.ones(len(reward)) *3, linestyle='--')
plt.title("Reward")
plt.xlabel("Timestep")
plt.ylabel("Reward")
plt.legend()
plt.show()

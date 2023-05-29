from numpy import loadtxt
from matplotlib import pyplot as plt
import numpy as np


data = loadtxt('csvfiles/closestpsuccess.csv', delimiter=',')
data2 = loadtxt('csvfiles/closestcontact3.csv', delimiter=',')
data3 = loadtxt('closestpoint.csv', delimiter=',')
data4 = loadtxt('closestpointbadrotate.csv', delimiter=',')

reward = loadtxt('csvfiles/Rewardsuccess.csv', delimiter=',')
contact = loadtxt('csvfiles/Rewardcontact3.csv', delimiter=',')
contact2 = loadtxt('Reward.csv', delimiter=',')
contact3 = loadtxt('Rewardbadrotate.csv', delimiter=',')

plt.figure()
plt.plot(data, color='orange'  ,label="with tactile feedback")
plt.plot(data2, color='grey', label="without tactile feedback")
plt.plot(data3,color='orange', label="_nolegend_")
plt.plot(data4, color='grey',label="_nolegend_")

plt.plot(np.zeros(len(data)), linestyle='--')
plt.title("Closest Point")
plt.xlabel("Timestep")
plt.ylabel("Closest Point")
plt.legend()
plt.show()
# savetxt('Reward.csv', rewards_, delimiter=' ')

plt.figure()
plt.plot(reward, color='orange'  ,label="with tactile feedback")
plt.plot(contact, color='grey',label="without tactile feedback")
plt.plot(contact2,color='orange'  , label="_nolegend_")
plt.plot(contact3,color='grey', label="_nolegend_")
plt.plot(np.ones(len(reward)) *3, linestyle='--')
plt.title("Reward")
plt.xlabel("Timestep")
plt.ylabel("Reward")
plt.legend()
plt.show()

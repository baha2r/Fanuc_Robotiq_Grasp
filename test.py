import numpy as np
import matplotlib.pyplot as plt

# x has 1000 data points all zero
x = np.zeros(10000)
noise = np.random.normal(0, 0.02, x.shape)
# plot noise distribution
plt.hist(noise, bins=50)
plt.show()
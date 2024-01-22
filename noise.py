import numpy as np
import matplotlib.pyplot as plt

# Generate 1000 samples of normal (Gaussian) noise around zero
mean = 0  # mean value of zero
std_deviation = 0.01  # standard deviation
samples = np.random.normal(mean, std_deviation, 10000)

# Plotting the distribution
plt.figure()
plt.hist(samples, bins=50)

# Adding titles and labels
plt.title('Distribution of 1000 Gaussian Noise Samples')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Show the plot
plt.show()

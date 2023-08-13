import numpy as np
gripper = [7.8762039e-11, -1.3363775e-10,  1.0000000e+00]
target  = [2.2337140e-01,  7.0342982e-01,  8.2052636e-01]
rel     = np.array([2.2337140e-01,  7.0342982e-01, -1.7947362e-01])
min_dist= np.array([1.8782738e-06, -5.9628837e-05,  7.8282371e-02])

# subtract rel from min_dist and print
print(min_dist - rel)

import debug

success = []
for i in range(100):
    suc = debug.main()
    success.append(suc)

# calculate the success percentage
success_perc = sum(success)/len(success)
print(success_perc)

# stationary target
# 0.86 wo noise
# 0.9 wo noise
# 0.89 wo noise
# 0.74 w noise normal 0.01
# 0.71 w noise uniform 0.01

# moving target
# 0.8 wo noise
# 0.86 wo noise
# 0.86 w noise normal 0.01
# 0.75 w noise normal 0.01

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# file imports
import pendulum_td3
import pendulum_ppo
import time

#  data = []  # for each episode's data
#  data2 = [1, 200, 82]
#  data3 = [2, 220, 83]
#
#  data.append(data2)
#  data.append(data3)
#  print(data)
repeat = 5
tde_times_pendulum = []
ppo_times_pendulum = []
sac_times_pendulum = []

for i in range(repeat):
    start = time.time()
    td3_df = pendulum_td3.td3_sim(0, 500, 200)
    td3_boxplot = td3_df.boxplot(column='Reward')
    end = time.time()
    tde_times_pendulum.append(end - start)
    # start = time.time()
    # ppo_df = pendulum_ppo.ppo_sim(0, 100, 200)
    # end = time.time()
    # tde_times_pendulum.append(end-start)

plt.show()
print(td3_df)

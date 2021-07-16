import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# file imports
import iiwa_ppo
import iiwa_sac
import iiwa_td3
import time

max_episodes = 500
max_steps = 200
repeat = 5

td3_times_iiwa = []
ppo_times_iiwa = []
sac_times_iiwa = []

for i in range(repeat):
    # Pendulum TD3
    start = time.time()  # start timer
    td3_df = iiwa_td3.td3_sim(0, max_episodes, max_steps)  # run algorithm
    end = time.time()  # end timer
    td3_times_iiwa.append(end - start)  # append timer in list
    # td3_boxplot = td3_df.boxplot(column='Reward')  # get reward boxplot

    # Pendulum PPO
    start = time.time()
    ppo_df = iiwa_ppo.ppo_sim(0, max_episodes, max_steps)
    end = time.time()
    ppo_times_iiwa.append(end - start)
    # ppo_boxplot = ppo_df.boxplot(column='Reward')

    # Pendulum SAC
    start = time.time()
    sac_df = iiwa_sac.sac_sim(0, max_episodes, max_steps)
    end = time.time()
    sac_times_iiwa.append(end - start)
    # sac_boxplot = sac_df.boxplot(column='Reward')

    # iiwa TD3

pendulum_times = [td3_times_iiwa, ppo_times_iiwa, sac_times_iiwa]
fig = plt.figure()

plt.boxplot(pendulum_times)

plt.show()
plt.savefig('boxplots.png')
# print(td3_df)

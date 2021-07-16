import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# file imports
import pendulum_td3
import pendulum_ppo
import pendulum_sac
import iiwa_ppo
import iiwa_sac
import iiwa_td3
import time

max_episodes = 100
max_steps = 100
repeat = 1
tde_times_pendulum = []
ppo_times_pendulum = []
sac_times_pendulum = []
tde_times_iiwa = []
ppo_times_iiwa = []
sac_times_iiwa = []

for i in range(repeat):
    # Pendulum TD3
    start = time.time()  # start timer
    td3_df = pendulum_td3.td3_sim(0, max_episodes, max_steps)  # run algorithm
    end = time.time()  # end timer
    tde_times_pendulum.append(end - start)  # append timer in list
    # td3_boxplot = td3_df.boxplot(column='Reward')  # get reward boxplot

    # Pendulum PPO
    start = time.time()
    ppo_df = pendulum_ppo.ppo_sim(0, max_episodes, max_steps)
    end = time.time()
    ppo_times_pendulum.append(end - start)
    # ppo_boxplot = ppo_df.boxplot(column='Reward')

    # Pendulum SAC
    start = time.time()
    sac_df = pendulum_sac.sac_sim(0, max_episodes, max_steps)
    end = time.time()
    sac_times_pendulum.append(end - start)
    # sac_boxplot = sac_df.boxplot(column='Reward')

    # iiwa TD3

pendulum_times = [tde_times_pendulum, ppo_times_pendulum, sac_times_pendulum]
fig = plt.figure()

plt.boxplot(pendulum_times)

plt.show()
plt.savefig('boxplots.png')
# print(td3_df)

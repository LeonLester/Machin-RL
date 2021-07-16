import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# file imports
import iiwa_ppo
import iiwa_sac
import iiwa_td3
import pendulum_ppo
import pendulum_sac
import pendulum_td3
import time

max_episodes = 500
max_steps = 200
repeat = 5


def pendulum_stuff():
    td3_times_pendulum = []
    ppo_times_pendulum = []
    sac_times_pendulum = []
    for i in range(repeat):
        print(f"Run #:[{i}/{repeat}]", end="\n")

        # Pendulum TD3
        start = time.time()  # start timer
        td3_df_p = pendulum_td3.td3_sim(0, max_episodes, max_steps)  # run algorithm
        end = time.time()  # end timer
        td3_times_pendulum.append(end - start)  # append timer in list
        # td3_boxplot = td3_df.boxplot(column='Reward')  # get reward boxplot
        print("TD3 done")

        # Pendulum PPO
        start = time.time()
        ppo_df_p = pendulum_ppo.ppo_sim(0, max_episodes, max_steps)
        end = time.time()
        ppo_times_pendulum.append(end - start)
        # ppo_boxplot = ppo_df.boxplot(column='Reward')
        print("PPO done")

        # Pendulum SAC
        start = time.time()
        sac_df_p = pendulum_sac.sac_sim(0, max_episodes, max_steps)
        end = time.time()
        sac_times_pendulum.append(end - start)
        # sac_boxplot = sac_df.boxplot(column='Reward')
        print("SAC done")

    pendulum_times = [td3_times_pendulum, ppo_times_pendulum, sac_times_pendulum]
    fig = plt.figure()

    plt.boxplot(pendulum_times)

    plt.show()
    plt.savefig('pendulum_boxplots.png')


def iiwa_stuff():
    td3_times_iiwa = []
    ppo_times_iiwa = []
    sac_times_iiwa = []
    for i in range(repeat):
        print(f"Run #:[{i}/{repeat}]", end="\n")

        # iiwa TD3
        start = time.time()  # start timer
        td3_df_i = iiwa_td3.td3_sim(0, max_episodes, max_steps)  # run algorithm
        end = time.time()  # end timer
        td3_times_iiwa.append(end - start)  # append timer in list
        print("TD3 done")

        # iiwa PPO
        start = time.time()
        ppo_df_i = iiwa_ppo.ppo_sim(0, max_episodes, max_steps)
        end = time.time()
        ppo_times_iiwa.append(end - start)
        print("PPO done")

        # iiwa SAC
        start = time.time()
        sac_df_i = iiwa_sac.sac_sim(0, max_episodes, max_steps)
        end = time.time()
        sac_times_iiwa.append(end - start)
        print("SAC done")

    iiwa_times = [td3_times_iiwa, ppo_times_iiwa, sac_times_iiwa]
    fig = plt.figure()

    plt.boxplot(iiwa_times)

    plt.show()
    plt.savefig('iiwa_boxplots.png')


pendulum_stuff()
iiwa_stuff()

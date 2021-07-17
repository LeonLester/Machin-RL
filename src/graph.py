import csv
import matplotlib.pyplot as plt

with open('csv/pendulum_data5.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)

data = [list(map(float, sublist)) for sublist in data]

fig = plt.figure()
fig.suptitle('Pendulum', fontsize=14, fontweight='bold')

ax = fig.add_subplot(111)
ax.boxplot(data)

ax.set_xlabel('Algorithms')
ax.set_ylabel('Seconds')

plt.xticks([1, 2, 3], ['TD3', 'PPO', 'SAC'])
plt.gcf()
plt.savefig('pendulum_boxplots.png')
plt.show()

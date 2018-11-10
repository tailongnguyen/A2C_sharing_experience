import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

folder = 'Map1_Ep12'
types = ['rewards', 'redundant_steps']
NUM = 0

logs_None = []
logs_Oracle = []
logs_Share = []
for i in range(3):
	logs_Oracle.append(pd.read_csv(folder + "/run_num_task_4-share_exp-num_episode_12-num_iters_50-lr_0.005-use_gae-oracle-tag-map_1_test_{}_{}.csv".format(i, types[NUM])))
	logs_None.append(pd.read_csv(folder + "/run_num_task_4-num_episode_12-num_iters_50-lr_0.005-use_gae-tag-map_1_test_{}_{}.csv".format(i, types[NUM])))
	logs_Share.append(pd.read_csv(folder + "/run_num_task_4-share_exp-num_episode_12-num_iters_50-lr_0.005-use_gae-tag-map_1_test_{}_{}.csv".format(i, types[NUM])))

labels = ["None", "Oracle", "Share-Z"]
for i, logs in enumerate([logs_None, logs_Oracle, logs_Share]):
	color = 'C' + str(i)
	mean_steps = np.mean(np.vstack([log['Step'].tolist() for log in logs]), 0)
	mean_value = np.mean(np.vstack([log['Value'].tolist() for log in logs]), 0)
	max_value  = np.max(np.vstack([log['Value'].tolist() for log in logs]), 0)
	min_value  = np.min(np.vstack([log['Value'].tolist() for log in logs]), 0)

	smooth = 10

	plt.plot(mean_steps[::smooth], mean_value[::smooth], color = color, label=labels[i])
	plt.fill_between(mean_steps[::smooth], min_value[::smooth], max_value[::smooth], alpha = 0.2, facecolor=color,linewidth=0, linestyle='dashdot', antialiased=True)

plt.title("Map 1 | 12 episodes | 50 steps")
plt.xlabel('Samples')
plt.ylabel(types[NUM])
plt.legend()
plt.grid(True)
plt.show()
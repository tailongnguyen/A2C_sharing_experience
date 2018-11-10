import matplotlib.pyplot as plt 
import matplotlib as mpl
# mpl.rcParams['figure.dpi'] = 160

import numpy as np 
import os 
import pandas as pd 

types = ['Non', 'oracle', 'Share_samples']
colors = ['C0', 'C1', 'C3']
root = "/home/yoshi/HMI/current-project/logs/log_pretrained_24_transfer_to13/"
eps = ['8',  '16', '24']

smooth = 5
fig = plt.figure()
index = 0
lines = []
labels = []
for ep in eps:
	logs = {}
	for t in types:
		logs[t] = [pd.read_csv(os.path.join(root, '{}_{}'.format(t, ep), \
					"run_.-tag-map_2_test_{}_rewards_task_{}.csv".format(ep, i))) for i in range(4)]
		logs[t] = [pd.read_csv(os.path.join(root, '{}_{}'.format(t, ep), \
					"run_.-tag-map_2_test_{}_rewards.csv".format(ep)))] + logs[t]

		assert len(logs[t]) == 5

	for i in range(5):
		index += 1
		ax = plt.subplot(len(eps), 5, index)
		plt.xlabel('{} episodes_task {}'.format(ep, i) if i > 0 else '{} episodes_average'.format(ep))
		for j, t in enumerate(types):
			y = logs[t][i]['Value']
			ax.xaxis.set_major_locator(plt.MaxNLocator(4))
			smoothed_y = [np.mean(y[max(0, i - smooth):min(i + smooth, len(y)-1)]) for i in range(len(y))]
			
			line = ax.plot(logs[t][i]['Step'], smoothed_y, color = colors[j])
			lines.append(line[0])
			labels.append(t)
			plt.plot(logs[t][i]['Step'], y, color = colors[j], alpha = 0.3)
			plt.grid(True)

# print(*zip(lines, labels))
fig.legend(lines[:3], types, loc = 8, ncol=3)
fig = mpl.pyplot.gcf()
fig.set_size_inches(30, 15)
fig.savefig('compare_transfer.png', bbox_inches='tight', dpi = 250)
# plt.show() 

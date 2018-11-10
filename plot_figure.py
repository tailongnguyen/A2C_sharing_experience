import matplotlib
matplotlib.use('Agg')
import numpy as np           		# Handle matrices
import random                		# Handling random number generation
import time                  		# Handling time calculation
import math
import matplotlib.pyplot as plt 	# Display graphs
from matplotlib.pyplot import figure
import os

class PlotFigure(object):
	
	def __init__(self, save_name, env, num_task, save_folder):
		self.env = env
		self.state_space = env.state_space
		self.num_task = num_task
		self.save_name = save_name
		self.save_folder = save_folder
		self.z_map = {}
		self.visit = np.zeros(self.env.map_array.shape, dtype = int)
		for i in range(num_task - 1):
			for j in range(i+1, num_task):
				self.z_map[i, j] = np.zeros(self.env.map_array.shape, dtype = int)

		if not os.path.isdir(save_folder):
			os.mkdir(save_folder)

	def _plot_point(self, ax, point, angle, length):
		x, y = point

		endy = length * math.sin(math.radians(angle)) + y
		endx = length * math.cos(math.radians(angle)) + x

		ax.plot([x, endx], [y, endy], color = 'blue')

	def _plot_star(self, ax, orig, lengths, max_length=0.5, angles=[270, 225, 180, 135, 90, 45, 0, 315]):
		max_len = max(lengths)
		for i, angle in enumerate(angles):
			self._plot_point(ax, orig, angle, lengths[i]*1.0 / max_len * max_length)

	def plot(self, policy, epoch):
		plt.clf()

		plt.figure()
		num_rows = self.num_task // 2 + self.num_task % 2 
		num_cols = 2 if self.num_task > 1 else 1
		for index in range(self.num_task):
			ax = plt.subplot(num_rows, num_cols, index + 1)
			for y in range(self.env.map_array.shape[0]):
				for x in range(self.env.map_array.shape[1]):
					if self.env.MAP[y][x] != 'x':
						self._plot_star(ax, (x, y), policy[x,y,index, 1])
						plt.plot([x,], [y,], marker='o', markersize=1, color="green")
					else:
						plt.scatter(x, y, marker='x', color="black", s = 10)

		if not os.path.isdir(os.path.join(self.save_folder, self.save_name)):
			os.mkdir(os.path.join(self.save_folder, self.save_name))

		plt.savefig(os.path.join(self.save_folder, self.save_name, str(epoch) + '.png'), bbox_inches='tight', dpi=250)

	def save_z(self):
		np.save(os.path.join(self.save_folder, self.save_name, 'visit'), self.visit)
		for i in range(self.num_task - 1):
			for j in range(i+1, self.num_task):
				np.save(os.path.join(self.save_folder, self.save_name, 'z_map_{}_{}'.format(i, j)), self.z_map[i, j])
					
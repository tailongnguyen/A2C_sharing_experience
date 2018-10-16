import numpy as np 
import matplotlib.pyplot as plt 

def laser(x, y, m):
	assert m[y][x] != 0
	move = [[0, 1], [0, -1], [1, 0], [-1, 0], [1, 1], [-1, -1], [1, -1], [-1, 1]]
	res = []
	for mv in move:
		nx, ny = x + mv[0], y + mv[1]
		temp = 0
		while m[ny][nx] > 0:
			plt.scatter(nx, ny, marker='o', color="green", s = 5)
			nx, ny = nx + mv[0], ny + mv[1]
			temp += 1

		res.append(temp)

	plt.annotate(str(res), (x, y))

	return res

map_array = np.array([
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0],
[0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0],
[0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0],
[0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0],
[0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0],
[0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0],
[0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0],
[0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0],
[0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
[0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 0],
[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
, dtype = int)

move = [[0, 1], [0, -1], [1, 0], [-1, 0], [1, 1], [-1, -1], [1, -1], [-1, 1]]

# plt.xlim([-1, map_array.shape[1]])
# plt.ylim([-1, map_array.shape[0]])

# state_space = [list(z) for z in  zip(np.where(map_array != 0)[1].tolist(), np.where(map_array != 0)[0].tolist())]
# rands = state_space[np.random.choice(range(len(state_space)))]
# print(rands)
# for y in range(map_array.shape[0]):
#     for x in range(map_array.shape[1]):
#         if map_array[y][x] == 0:
#             plt.scatter(x, y, marker='x', color="red")
#         elif x == rands[0] and y == rands[1]:
#         	laser(x, y, map_array)

# plt.show()

def min_dist(map_array):

	distance = np.zeros_like(map_array) - 1
	target = [list(z) for z in  zip(np.where(map_array == 3)[0].tolist(), np.where(map_array == 3)[1].tolist())][1]
	distance[target[0], target[1]] = 0
	visisted = {}
	for i in range(distance.shape[0]):
		for j in range(distance.shape[1]):
			visisted[i, j] = False

	queue = []
	queue.append(target)

	while len(queue) > 0:
		pos = queue[0]
		queue = queue[1:]

		visisted[pos[0], pos[1]] = True

		for m in move:

			neighbor = [pos[0] + m[0], pos[1] + m[1]]

			if map_array[neighbor[0], neighbor[1]] == 0:
				continue 

			if not visisted[neighbor[0], neighbor[1]]:
				distance[neighbor[0], neighbor[1]] = distance[pos[0], pos[1]] + 1			
				queue.append(neighbor)
				visisted[neighbor[0], neighbor[1]] = True


	print(distance)
	return distance

# def adv_map():
distance = min_dist(map_array)
adv = {}

for i in range(distance.shape[1]):
	for j in range(distance.shape[0]):
		if distance[j][i] == -1:
			continue

		for m_i, m in enumerate(move):
			x, y = m
			if distance[j + y][i + x] == -1:
				adv[i, j, m_i] = -2
				continue

			if distance[j + y][i + x] < distance[j][i]:
				adv[i, j, m_i] = 1
			elif distance[j + y][i + x] > distance[j][i]:
				adv[i, j, m_i] = -1
			else:
				adv[i, j, m_i] = 0

# def plot_point(ax, point, angle, length):
# 	x, y = point

# 	endy = length * math.sin(math.radians(angle)) + y
# 	endx = length * math.cos(math.radians(angle)) + x

# 	ax.plot([x, endx], [y, endy], color = 'blue')

# def plot_star(ax, orig, lengths, max_length=0.5, angles=[270, 225, 180, 135, 90, 45, 0, 315]):
# 		max_len = max(lengths)
# 		for i, angle in enumerate(angles):
# 			plot_point(ax, orig, angle, lengths[i]*1.0 / max_len * max_length)

# ax = plt.subplot(111)

# for i in range(distance.shape[1]):
# 	for j in range(distance.shape[0]):
# 		plot_star(ax, (x, y), policy[x,y,index, 1])
# 		plt.plot([x,], [y,], marker='o', markersize=1, color="green")
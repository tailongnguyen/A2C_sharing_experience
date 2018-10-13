from sxsy import SXSY 
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

distance = np.zeros_like(map_array)
target = [list(z) for z in  zip(np.where(map_array == 3)[0].tolist(), np.where(map_array == 3)[1].tolist())][1]
move = [[0, 1], [0, -1], [1, 0], [-1, 0], [1, 1], [-1, -1], [1, -1], [-1, 1]]

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
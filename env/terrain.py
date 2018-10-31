import matplotlib.pyplot as plt
import random
import numpy as np

from collections import Counter
from random import randint
try:
    from .sxsy import SXSY
    from .map import ENV_MAP
    from .controller import Player
except ModuleNotFoundError:
    from sxsy import SXSY
    from map import ENV_MAP
    from controller import Player

class Terrain:
    def __init__(self, map_index, use_laser = False):
        self.MAP = ENV_MAP[map_index]['map']
        self.ORACLE = ENV_MAP[map_index]['oracle']
        self.num_task = 0

        for row in self.MAP:
            characters = Counter(row)
            if len(characters) > 2:
                self.num_task += len(characters) - 2
        self.map_array = np.array([[m_ for m_ in mi] for mi in self.MAP])
        self.reward_locs = []
        for i in range(self.num_task):
            self.reward_locs.append([np.where(self.map_array == str(i+1))[1][0], np.where(self.map_array == str(i+1))[0][0]])

        self.state_space = [list(z) for z in  zip(np.where(self.map_array != 'x')[1].tolist(), np.where(self.map_array != 'x')[0].tolist())]

        self.state_to_index = np.zeros(self.map_array.shape).astype(int) - 1
        
        for i, s in enumerate(self.state_space):
            self.state_to_index[s[1]][s[0]] = i
        
        assert np.sum(self.state_to_index != -1) == len(self.state_space)

        self.action_size = 8
        self.reward_range = 1.0
        self.reward_goal = 1.0

        if not use_laser:
            self.cv_state_onehot = np.identity(len(self.state_space), dtype=int)
        else:
            self.cv_state_onehot = [0] * len(self.state_space)

            for s in self.state_space:
                state_index = self.state_to_index[s[1]][s[0]]
                self.cv_state_onehot[state_index] = np.array(self.laser(s[0], s[1]))

            self.cv_state_onehot = np.asarray(self.cv_state_onehot)
            # self.cv_state_onehot = self.cv_state_onehot

        self.cv_action_onehot = np.identity(self.action_size, dtype=int)
        
        self.min_dist = []
        self.advs = []
        for i in range(self.num_task):
            self.min_dist.append(self.cal_min_dist(i))
            self.advs.append(self.adv_map(self.min_dist[-1]))

    def adv_map(self, distance):
        move = [[0, 1], [0, -1], [1, 0], [-1, 0], [1, 1], [-1, -1], [1, -1], [-1, 1]]
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

        return adv

    def cal_min_dist(self, task_idx):
        distance = np.zeros(self.map_array.shape) - 1
        target = self.reward_locs[task_idx][::-1]
        distance[target[0], target[1]] = 0
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
                if self.map_array[neighbor[0], neighbor[1]] == 'x':
                    continue 

                if not visisted[neighbor[0], neighbor[1]]:
                    distance[neighbor[0], neighbor[1]] = distance[pos[0], pos[1]] + 1           
                    queue.append(neighbor)
                    visisted[neighbor[0], neighbor[1]] = True

        return distance

    def laser(self, x, y):
        assert self.map_array[y][x] != 0
        move = [[0, 1], [0, -1], [1, 0], [-1, 0], [1, 1], [-1, -1], [1, -1], [-1, 1]]
        res = []
        for mv in move:
            nx, ny = x + mv[0], y + mv[1]
            temp = 0
            while self.map_array[ny][nx] > 0:
                nx, ny = nx + mv[0], ny + mv[1]
                temp += 1

            res.append(temp)

        return res

    def getreward(self):
        done = False
        reward = -0.01

        x_pos, y_pos = self.reward_locs[self.task]

        if abs(self.player.x - x_pos) < self.reward_range and abs(self.player.y - y_pos) < self.reward_range:
            reward = self.reward_goal
            done = True

        return reward, done

    def checkepisodeend(self):
        for x_pos, y_pos in self.reward_locs:
            if abs(self.player.x - x_pos) < self.reward_range and abs(self.player.y - y_pos) < self.reward_range:
                return 1
        return 0

    def plotgame(self):

        plt.clf()
        plt.xlim([-1, self.map_array.shape[1]])
        plt.ylim([-1, self.map_array.shape[0]])

        # for y in range(self.map_array.shape[0]):
        #     for x in range(self.map_array.shape[1]):
        #         if self.MAP[y][x] == 'x':
        #             plt.scatter(x, y, marker='x', color="red")

        for (x, y) in self.state_space:
            plt.scatter(x, y, marker='o', color="green", s = 5)


        for x_pos, y_pos in self.reward_locs:
            plt.scatter(x_pos, y_pos, marker='o', color="blue")

        plt.show()

    def resetgame(self, task, sx, sy):
        #self.player = Player(7, 1, self)
       
        self.player = Player(sx, sy, self)

        self.task = task
            
if __name__ == '__main__':
    ter = Terrain(2)
    # print(ter.state_space)
    ter.plotgame()
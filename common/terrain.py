import matplotlib.pyplot as plt
import random
import numpy as np
import time
import multiprocessing
from random import randint

try:
    from .map import ENV_MAP
except ModuleNotFoundError:
    from map import SXSY
    
def make_env(**kwargs):
    def _thunk():
        return Terrain(**kwargs)

    return _thunk

class Terrain:
    def __init__(self, map_index, use_laser = False, immortal = False, task = 0):
        self.MAP = ENV_MAP[map_index]['map']
        self.map_array = np.array(self.MAP, dtype = int)
        self.cv_action=[[0,-1],[-1,-1],[-1,0],[-1,1],[0,1],[1,1],[1,0],[1,-1]]

        self.trajectory = []

        # plt.ion()
        # self.fig = plt.figure()

        self.position = None
        self.task = task

        self.reward_locs = [list(z) for z in  zip(np.where(self.map_array == 3)[1].tolist(), np.where(self.map_array == 3)[0].tolist())]
        self.state_space = [list(z) for z in  zip(np.where(self.map_array != 0)[1].tolist(), np.where(self.map_array != 0)[0].tolist())]

        self.state_to_index = np.zeros_like(self.MAP) - 1
        
        for i, s in enumerate(self.state_space):
            self.state_to_index[s[1]][s[0]] = i
        
        assert np.sum(self.state_to_index != -1) == len(self.state_space)

        self.action_size = 8
        self.reward_range = 1.0
        self.reward_goal = 1.0
        
        self.num_task = len(self.reward_locs)
        self.immortal = immortal

        if not use_laser:
            self.cv_state_onehot = np.identity(len(self.state_space), dtype=int)
        else:
            self.cv_state_onehot = [0] * len(self.state_space)

            for s in self.state_space:
                state_index = self.state_to_index[s[1]][s[0]]
                self.cv_state_onehot[state_index] = np.array(self.laser(s[0], s[1]))

            self.cv_state_onehot = np.asarray(self.cv_state_onehot)

        self.cv_action_onehot = np.identity(self.action_size, dtype=int)
        self.cv_task_onehot = np.identity(len(self.reward_locs), dtype=int)
        
        self.observation_space = self.cv_state_onehot.shape[1]
        self.action_space = self.cv_action_onehot.shape[1]

        self.min_dist = []
        for i in range(self.num_task):
            self.min_dist.append(self.cal_min_dist(i))

    def cal_min_dist(self, task_idx):
        distance = np.zeros_like(self.map_array)
        target = [list(z) for z in  zip(np.where(self.map_array == 3)[0].tolist(), np.where(self.map_array == 3)[1].tolist())][task_idx]
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

                if self.map_array[neighbor[0], neighbor[1]] == 0:
                    continue 

                if not visisted[neighbor[0], neighbor[1]]:
                    distance[neighbor[0], neighbor[1]] = distance[pos[0], pos[1]] + 1           
                    queue.append(neighbor)
                    visisted[neighbor[0], neighbor[1]] = True

        return distance

    def reset(self):
        reached = True
        np.random.seed(int(multiprocessing.current_process().name.split('-')[1]) * int(time.time() * 1000 % 1000))

        while reached:
            self.position = self.state_space[np.random.choice(range(len(self.state_space)))]
            reached = self.reach_target()

        cur_x, cur_y = self.position
        self.trajectory = [self.position]
        return (cur_x, cur_y)

    def reset_task(self):
        pass
    
    def step(self, action):
        cur_x, cur_y = self.position
        x, y = self.cv_action[action]
        reward = -0.01

        if self.MAP[cur_y + y][cur_x + x] == 0:
            if self.immortal:
                ob = (cur_x, cur_y)
                done = 0
            else:
                reward = -1.0
                ob = (cur_x + x, cur_y + y)
                done = 1
        else:
            self.position = (cur_x + x, cur_y + y)
            ob = (cur_x + x, cur_y + y)
            done = 0

            if self.reach_target():
                reward = self.reward_goal
                done = 1

        self.trajectory.append(self.position)

        return ob, reward, done

    def reach_target(self):
        cur_x, cur_y = self.position
        x_pos, y_pos = self.reward_locs[self.task]

        if abs(cur_x - x_pos) < self.reward_range and abs(cur_y - y_pos) < self.reward_range:
            return True
        return False

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

    def clear_plot(self):
        plt.clf()
        
    def plotgame(self, action):

        plt.xlim([-1, self.map_array.shape[1]])
        plt.ylim([-1, self.map_array.shape[0]])

        for y in range(self.map_array.shape[0]):
            for x in range(self.map_array.shape[1]):
                if self.MAP[y][x] == 0:
                    plt.scatter(x, y, marker='x', color="red")

        for (x, y) in self.state_space:
            plt.scatter(x, y, marker='o', color="green", s = 5)

        for x_pos, y_pos in self.reward_locs:
            plt.scatter(x_pos, y_pos, marker='*', color="yellow")

        for idx, (x, y) in enumerate(self.trajectory):
            plt.scatter(x, y, marker='^', color="blue", s = 30)

        plt.plot([pos[0] for pos in self.trajectory], [pos[1] for pos in self.trajectory])
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
            
if __name__ == '__main__':
    ter = Terrain(4)
    ter.resetgame(1, 1, 1)
    ter.plotgame()
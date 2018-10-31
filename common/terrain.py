import matplotlib.pyplot as plt
import random
import numpy as np
import time
import os
import pickle
import multiprocessing
from random import randint

try:
    from .map import ENV_MAP
except ModuleNotFoundError:
    from map import ENV_MAP
    
def make_env(**kwargs):
    def _thunk():
        return Terrain(**kwargs)

    return _thunk

class Terrain:
    def __init__(self, map_index, use_laser = False, immortal = False, task = 0, save_folder = None, plotgame = False):
        self.MAP = ENV_MAP[map_index]['map']
        self.map_array = np.array(self.MAP, dtype = int)
        self.cv_action=[[0,-1],[-1,-1],[-1,0],[-1,1],[0,1],[1,1],[1,0],[1,-1]]

        self.trajectory = {}
        self.save_folder = save_folder

        if plotgame:
            plt.ion()
            self.fig = plt.figure()

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

        self.episode = 0
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
        distance = np.zeros_like(self.map_array) - 1
        target = [list(z) for z in  zip(np.where(self.map_array == 3)[0].tolist(), np.where(self.map_array == 3)[1].tolist())][task_idx]
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

        if self.episode > 0 and self.episode == 2000:
            if not os.path.isdir(self.save_folder):
                os.makedirs(self.save_folder)

            with open(os.path.join(self.save_folder, '{}_traj_{}.pkl'.format(multiprocessing.current_process().name, self.episode)), 'wb') as f:
                pickle.dump(self.trajectory, f, pickle.HIGHEST_PROTOCOL)
    
        self.episode += 1
        self.trajectory[self.episode] = [self.position]
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

        self.trajectory[self.episode].append(self.position)

        start_state = self.trajectory[self.episode][0] 
        last_state = self.position
        redundant = len(self.trajectory[self.episode]) + self.min_dist[self.task][last_state[1], last_state[0]] - self.min_dist[self.task][start_state[1], start_state[0]]

        return ob, reward, done, redundant

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
        
    def plotgame(self, trajectory):


        for ep_idx, episode in trajectory.items():
            if ep_idx < 1800:
                continue
            plt.title("Epoch {}".format(ep_idx))
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

            for idx, (x, y) in enumerate(trajectory[ep_idx]):
                plt.scatter(x, y, marker='^', color="blue", s = 30)
                plt.plot([pos[0] for pos in trajectory[ep_idx][:idx + 1]], [pos[1] for pos in trajectory[ep_idx][:idx + 1]])
            
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()

            self.clear_plot()
        
            
if __name__ == '__main__':
    ter = Terrain(map_index = 4, plotgame = True)
    trajectory = pickle.load(open("../plot/2018-10-19_00-01-47_test_ppo/Process-2_traj_2000.pkl", "rb"))
    ter.plotgame(trajectory)
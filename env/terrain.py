import matplotlib.pyplot as plt
import matplotlib as mpl
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
        self.map_index = map_index

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
        num_plots = self.num_task * (self.num_task - 1) / 2
        num_rows = num_plots // 2 + num_plots % 2 
        num_cols = 2 if num_plots > 1 else 1
        index = 0
        for i in range(self.num_task - 1):
            for j in range(i+1, self.num_task):
                index += 1
                plt.subplot(num_rows, num_cols, index)
                plt.title('{}_{}'.format(i+1, j+1))
                zmap = np.load('../visualize/ZMap/Map_{}/z_map_{}_{}.npy'.format(self.map_index, i, j))
                plt.xlim([-1, self.map_array.shape[1]])
                plt.ylim([-1, self.map_array.shape[0]])
                plt.tick_params(
                        axis='x',          # changes apply to the x-axis
                        which='both',      # both major and minor ticks are affected
                        bottom=False,      # ticks along the bottom edge are off
                        top=False,         # ticks along the top edge are off
                        labelbottom=False)

                cm = plt.cm.get_cmap('Reds')
                x_s = []
                y_s = []
                c_s = []
                s_s = []
                for y in range(self.map_array.shape[0]):
                    for x in range(self.map_array.shape[1]):
                        if self.MAP[y][x] == 'x':
                            # plt.scatter(x, y, marker='x', color="black", s = 10)
                            if self.MAP[y][x-1] == 'x' and x > 0:
                                plt.plot([x-1, x], [y, y], color='black')
                            if self.MAP[y-1][x] == 'x' and y > 0:
                                plt.plot([x, x], [y-1, y], color='black')
                        elif self.MAP[y][x] == '.':
                            x_s.append(x)
                            y_s.append(y)
                            c_s.append(zmap[y][x])
                            s_s.append(int(zmap[y][x] / np.max(zmap) * 80))
                            # plt.text(x, y, s = zmap[y][x])
                        else:
                            plt.scatter(x, y, marker='*', color="green", s = 50)
                            plt.text(x + 0.3, y - 0.5, s = self.MAP[y][x])

                sc = plt.scatter(x_s, y_s, c = c_s, vmin=np.unique(c_s)[1], vmax=np.max(c_s) + 1000, s= s_s, cmap=cm)
                plt.colorbar(sc)

        plt.show()

    def plot_z(self, weights_path):
        import sys
        import tensorflow as tf
        sys.path.insert(0, '/home/yoshi/HMI/current-project/')
        from network import ZNetwork

        plt.clf()
        num_plots = self.num_task * (self.num_task - 1) / 2
        num_rows = num_plots // 2 + num_plots % 2 
        num_cols = 2 if num_plots > 1 else 1
        index = 0
        scatter_size = 500
        for i in range(self.num_task - 1, 0, -1):
            for j in range(i-1, -1, -1):
                index += 1
                plt.subplot(num_rows, num_cols, index)
                plt.title('{}_{}'.format(j+1, i+1))
                
                plt.xlim([-1, self.map_array.shape[1]])
                plt.ylim([-1, self.map_array.shape[0]])
                plt.tick_params(
                        axis='x',          # changes apply to the x-axis
                        which='both',      # both major and minor ticks are affected
                        bottom=False,      # ticks along the bottom edge are off
                        top=False,         # ticks along the top edge are off
                        labelbottom=False)
                plt.tick_params(
                        axis='y',          # changes apply to the x-axis
                        which='both',      # both major and minor ticks are affected
                        bottom=False,      # ticks along the bottom edge are off
                        top=False,         # ticks along the top edge are off
                        labelbottom=False)

                tf.reset_default_graph()
                sess = tf.Session()
                oracle = ZNetwork(
                            state_size = self.cv_state_onehot.shape[1],
                            action_size = 2,
                            learning_rate = 0.005,
                            name = 'oracle_{}_{}'.format(j, i)
                            )

                oracle.restore_model(sess, weights_path)

                zmap = np.zeros(self.map_array.shape)
                cm = plt.cm.get_cmap('jet')

                x_s = []
                y_s = []
                c_s = []
                s_s = []
                wx_s = []
                wy_s = []
                for y in range(self.map_array.shape[0]):
                    for x in range(self.map_array.shape[1]):
                        if self.MAP[y][x] == 'x':
                            wx_s.append(x)
                            wy_s.append(y)
                        elif self.MAP[y][x] == '.':
                            state_index = self.state_to_index[y][x]
                            o = sess.run(
                                        oracle.oracle,
                                        feed_dict={
                                            oracle.inputs: [self.cv_state_onehot[state_index].tolist()]
                                        })
                            zmap[y][x] = o[0][1]
                            x_s.append(x)
                            y_s.append(y)
                            c_s.append(zmap[y][x])
                            # s_s.append(int(zmap[y][x] * 200))
                        else:
                            # plt.scatter(x, y, marker='*', color="green", s = 50)
                            plt.text(x - 0.3, y - 0.3 , s = self.MAP[y][x], fontsize = 20)            

                sc = plt.scatter(x_s, y_s, marker = 's', c = c_s, vmin=np.min(c_s), vmax=np.max(c_s), s= scatter_size, cmap=cm)
                plt.scatter(wx_s, wy_s, marker = 's', c = 'black', s = scatter_size)
                plt.colorbar(sc)
                np.save("zmap_{}_{}".format(j, i), zmap)
        fig = mpl.pyplot.gcf()
        fig.set_size_inches(24, 29)
        fig.savefig('Zmap.png', bbox_inches='tight', dpi = 250)

    def resetgame(self, task, sx, sy):
        #self.player = Player(7, 1, self)
       
        self.player = Player(sx, sy, self)

        self.task = task
            
if __name__ == '__main__':
    ter = Terrain(1)
    # print(ter.state_space)
    ter.plot_z("/home/yoshi/HMI/current-project/logs/2018-11-08_17-37-19_new/num_task_4-share_exp-num_episode_12-num_iters_50-lr_0.005-use_gae/checkpoints")

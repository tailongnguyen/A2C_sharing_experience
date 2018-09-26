import matplotlib.pyplot as plt
import random
import numpy as np
import json
from env.map import ENV_MAP
from env.sxsy import SXSY
from random import randint

map_array = np.array(ENV_MAP[5]['map']).astype(int)

state_space = [list(z) for z in  zip(np.where(map_array != 0)[1].tolist(), np.where(map_array != 0)[0].tolist())]

SMAP = []
for i in range(5000):
	ep_inits = []
	for e in range(20):
		rands = state_space[np.random.choice(range(len(state_space)))]
		ep_inits.append((rands[0], rands[1]))
	SMAP.append(ep_inits)

file = open('temp.txt','w')	
file.write(str(SMAP))


import matplotlib.pyplot as plt
import random
import numpy as np
import json
from env.terrain import Terrain
from env.sxsy import SXSY
from random import randint

ter = Terrain(1)

state_space = ter.state_space
print(len(state_space))
SMAP = []
for i in range(5000):
	ep_inits = []
	for e in range(20):
		rands = state_space[np.random.choice(range(len(state_space)))]
		ep_inits.append((rands[0], rands[1]))
	SMAP.append(ep_inits)

file = open('temp.txt','w')	
file.write(str(SMAP))


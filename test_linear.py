import matplotlib.pyplot as plt
import numpy as np
import random

from easy21 import *
from montecarlo import MonteCarloSim
from mpl_toolkits.mplot3d import Axes3D
from rl import *
from sarsa import SarsaSim, SarsaLinearSim
from tqdm import tqdm #for progress bars

l = SarsaLinearSim(epsilon=0.05,
                   alpha=0.01,
                   gamma=1,
                   lambdah=0.3,
                   debug=False)
for _ in tqdm(range(50000)):
    l.sim_step()

Z = np.zeros((21, 10))
for x in range(1, 11):
    for y in range(1, 22):
        state = State(player_score=y, dealer_score=x)
        action = l.value_func.get_action(state)
        Z[y-1, x-1] = l.value_func.get_state_action(state, action)

X, Y = np.meshgrid(range(1, 11), range(1, 22))
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('Dealer score')
ax.set_ylabel('Player score')
ax.set_zlabel('Value')
ax.set_title('Sarsa with Approximation value function')
surf = ax.plot_surface(X, Y, Z)
plt.show()
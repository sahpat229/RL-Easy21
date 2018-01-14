from easy21 import *
import matplotlib.pyplot as plt
from montecarlo import MonteCarloSim
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random
from rl import *
from sarsa import SarsaSim

mc = MonteCarloSim(100, 10, 21, False)
for _ in range(100000):
    mc.sim_step()

Z = np.zeros((21, 10))
for x in range(1, 11):
    for y in range(1, 22):
        state = State(player_score=y, dealer_score=x)
        action = mc.value_func.get_action(state)
        Z[y-1, x-1] = mc.value_func.get_state_action(state, action)

X, Y = np.meshgrid(range(1, 11), range(1, 22))
fig = plt.figure()
plt.title('Monte Carlo value function')
ax = fig.gca(projection='3d')
ax.set_xlabel('Dealer score')
ax.set_ylabel('Player score')
ax.set_zlabel('Value')
surf = ax.plot_surface(X, Y, Z)
plt.show()

ss = SarsaSim(100, 10, 21, 1, 0.9, False)
for _ in range(100000):
    ss.sim_step()

Z = np.zeros((21, 10))
for x in range(1, 11):
    for y in range(1, 22):
        state = State(player_score=y, dealer_score=x)
        action = ss.value_func.get_action(state)
        Z[y-1, x-1] = ss.value_func.get_state_action(state, action)

X, Y = np.meshgrid(range(1, 11), range(1, 22))
fig = plt.figure()
plt.title('Sarsa value function')
ax = fig.gca(projection='3d')
ax.set_xlabel('Dealer score')
ax.set_ylabel('Player score')
ax.set_zlabel('Value')
surf = ax.plot_surface(X, Y, Z)
plt.show()
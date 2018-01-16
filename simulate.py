import matplotlib.pyplot as plt
import numpy as np
import random

from easy21 import *
from montecarlo import MonteCarloSim
from mpl_toolkits.mplot3d import Axes3D
from rl import *
from sarsa import SarsaSim, SarsaLinearSim
from tqdm import tqdm #for progress bars

mc = MonteCarloSim(100, 10, 21, False)
for _ in tqdm(range(100000)):
    mc.sim_step()

Z = np.zeros((21, 10))
for x in range(1, 11):
    for y in range(1, 22):
        state = State(player_score=y, dealer_score=x)
        action = mc.value_func.get_action(state)
        Z[y-1, x-1] = mc.value_func.get_state_action(state, action)

X, Y = np.meshgrid(range(1, 11), range(1, 22))
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('Dealer score')
ax.set_ylabel('Player score')
ax.set_zlabel('Value')
ax.set_title('Monte Carlo value function')
surf = ax.plot_surface(X, Y, Z)
plt.show()

deviations = [0 for _ in range(11)]
lambdahs = np.linspace(0, 1, 11)
for iteration, lambdah in enumerate(lambdahs):
    ss = SarsaSim(100, 10, 21, 1, lambdah, False)
    for _ in tqdm(range(1000)):
        ss.sim_step()
    deviations[iteration] = ss.get_deviation(mc)
plt.scatter(x=lambdahs,
            y=deviations)
plt.xlabel('Lambda')
plt.ylabel('Mean squared deviation')
plt.title('Mean squared deviation vs Lambda')
plt.show()

deviations = [[], []]
ss = SarsaSim(100, 10, 21, 1, 0, False)
for _ in tqdm(range(1000)):
    ss.sim_step()
    deviations[0].append(ss.get_deviation(mc))

ss = SarsaSim(100, 10, 21, 1, 1, False)
for _ in tqdm(range(1000)):
    ss.sim_step()
    deviations[1].append(ss.get_deviation(mc))
lambda_0, = plt.plot(range(1000), deviations[0])
lambda_1, = plt.plot(range(1000), deviations[1])
plt.xlabel('Episode No.')
plt.ylabel('Mean squared deviation')
plt.title('Mean squared deviation vs Lambda')
plt.legend([lambda_0, lambda_1], 
           ['Lambda=0', 'Lambda=1'])
plt.show()

deviations = [0 for _ in range(11)]
lambdahs = np.linspace(0, 1, 11)
for iteration, lambdah in enumerate(lambdahs):
    sl = SarsaLinearSim(epsilon=0.05,
                        alpha=0.01,
                        gamma=1,
                        lambdah=lambdah,
                        debug=False)
    for _ in tqdm(range(1000)):
        sl.sim_step()
    deviations[iteration] = sl.get_deviation(mc)
plt.scatter(x=lambdahs,
            y=deviations)
plt.xlabel('Lambda')
plt.ylabel('Mean squared deviation')
plt.title('Mean squared deviation vs Lambda')
plt.show()

deviations = [[], []]
sl = SarsaLinearSim(epsilon=0.05,
                    alpha=0.01,
                    gamma=1,
                    lambdah=0,
                    debug=False)
for _ in tqdm(range(1000)):
    sl.sim_step()
    deviations[0].append(sl.get_deviation(mc))

sl = SarsaLinearSim(epsilon=0.05,
                    alpha=0.01,
                    gamma=1,
                    lambdah=1,
                    debug=False)
for _ in tqdm(range(1000)):
    sl.sim_step()
    deviations[1].append(sl.get_deviation(mc))
lambda_0, = plt.plot(range(1000), deviations[0])
lambda_1, = plt.plot(range(1000), deviations[1])
plt.xlabel('Episode No.')
plt.ylabel('Mean squared deviation')
plt.legend([lambda_0, lambda_1], 
           ['Lambda=0', 'Lambda=1'])
plt.show()
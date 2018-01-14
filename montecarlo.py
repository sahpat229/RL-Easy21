from easy21 import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random
from rl import *

class MonteCarloSim():
    def __init__(self, n0, dealer_max, player_max, debug=False):
        self.value_func = ValueFunction(dealer_max, player_max)
        self.n0 = n0
        self.state_action_counter = np.zeros((dealer_max+1, player_max+1, 2))
        self.debug = debug

    def sim_step(self):
        game = GameInstance()
        state, reward = game.initialState()
        state_list = []
        action_list = []
        n_list = []

        while not state.terminated:
            state_list.append(state)
            epsilon = float(self.n0) / (self.n0 + np.sum(self.state_action_counter[state.dealer_score, 
                                                                                   state.player_score, 
                                                                                   :]))
            explore = np.random.choice([True, False], p=[epsilon, 1-epsilon])
            if explore:
                action = [True, False][random.randint(0, 1)]
            else:
                action = self.value_func.get_action(state)
            action_list.append(action)

            self.state_action_counter[state.dealer_score,
                                      state.player_score,
                                      int(action)] += 1
            n_list.append(self.state_action_counter[state.dealer_score,
                                                    state.player_score,
                                                    int(action)])
            state, reward = game.step(action)
        final_state = state

        state_tracker = set()
        for state, n, action in zip(state_list, n_list, action_list):
            if state not in state_tracker:
                orig_q = self.value_func.get_state_action(state, action)
                new_q = orig_q + (1 / float(n))*(reward - orig_q)
                self.value_func.insert_state_action(state, action, new_q)
                state_tracker.add(state)

        if self.debug:
            str_state_list = State.str_state_list(state_list)
            str_action_list = [str(action) for action in action_list]
            for str_state, str_action in zip(str_state_list, str_action_list):
                print(str_state+"->"+str_action+"->", end="")
            print(State.str_state_list([final_state])[0], "Reward:", reward)
            print(n_list)

        return reward

# mc = MonteCarloSim(100, 10, 21, False)
# for _ in range(100000):
#     mc.sim_step()

# Z = np.zeros((21, 10))
# for x in range(1, 11):
#     for y in range(1, 22):
#         state = State(player_score=y, dealer_score=x)
#         action = mc.value_func.get_action(state)
#         Z[y-1, x-1] = mc.value_func.get_state_action(state, action)

# X, Y = np.meshgrid(range(1, 11), range(1, 22))
# print(X.shape, Y.shape, Z.shape)
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# surf = ax.plot_surface(X, Y, Z)
# plt.show()

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
print(X.shape, Y.shape, Z.shape)
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z)
plt.show()
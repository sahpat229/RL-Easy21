from easy21 import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random

class ValueFunction():
    def __init__(self, dealer_max, player_max):
        self.q = np.zeros((dealer_max+1, player_max+1, 2))

    def get_action(self, state):
        if not (1 <= state.dealer_score <= 10 and 1 <= state.player_score <= 21):
            return [False, True][random.randint(0, 1)]

        values = self.q[state.dealer_score, state.player_score, :]
        if values.max() == 0:
            if np.sum(values == 0) == 2:
                return [False, True][random.randint(0, 1)]
            else:
                return [False, True][np.argmax(values)]
        else:
            return [False, True][np.argmax(values)]

    def insert_state_action(self, state, action, g_return):
        self.q[state.dealer_score, state.player_score, int(action)] = g_return

    def get_state_action(self, state, action):
        if not (1 <= state.dealer_score <= 10 and 1 <= state.player_score <= 21):
            return 0
        return self.q[state.dealer_score, state.player_score, int(action)]

    def visualize(self):
        Z = np.zeros((21, 10))
        for x in range(1, 11):
            for y in range(1, 22):
                state = State(player_score=y, dealer_score=x)
                action = self.get_action(state)
                Z[y-1, x-1] = int(action)
        X, Y = np.meshgrid(range(1, 11), range(1, 22))
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X, Y, Z)
        plt.show()
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

class LinearValueFunc():
    def __init__(self):
        # 36 features
        self.weights = np.zeros(36)

    def bin_score(self, score, bins):
        one_hots = []
        for _bin in bins:
            one_hots.append(int(_bin[0] <= score <= _bin[1]))
        return one_hots

    def combine_encoded(self, dealer_encoded, player_encoded, action_encoded):
        combined_encoding = [0 for _ in range(36)]
        dealer_contrib = 0
        for index, dealer_hot in enumerate(dealer_encoded):
            if dealer_hot == 1:
                dealer_contrib = index * len(player_encoded) * len(action_encoded)
        action_contrib = 0
        for index, action_hot in enumerate(action_encoded):
            if action_hot == 1:
                action_contrib = index * 1
        for index, player_hot in enumerate(player_encoded):
            if player_hot == 1:
                player_contrib = index * len(action_encoded)
                combined_encoding[dealer_contrib + action_contrib + player_contrib] = 1
        return combined_encoding

    def featurize(self, state, action):
        dealer_encoded = self.bin_score(state.dealer_score,
                                        [[1, 4], [4, 7], [7, 10]])
        player_encoded = self.bin_score(state.player_score,
                                        [[1, 6], [4, 9], [7, 12], [10, 15], [13, 18], [16, 21]])
        action_encoded = self.bin_score(int(action),
                                        [[0, 0], [1, 1]])
        combined_encoding = self.combine_encoded(dealer_encoded=dealer_encoded,
                                                 player_encoded=player_encoded,
                                                 action_encoded=action_encoded)
        return np.array(combined_encoding)

    def get_state_action(self, state, action):
        if not (1 <= state.dealer_score <= 10 and 1 <= state.player_score <= 21):
            return 0
        else:
            encoding = self.featurize(state, action)
            return np.dot(encoding, self.weights)

    def get_action(self, state):
        if not (1 <= state.dealer_score <= 10 and 1 <= state.player_score <= 21):
            return [False, True][random.randint(0, 1)]
        value_true = self.get_state_action(state, True)
        value_false = self.get_state_action(state, False)
        if value_true == value_false:
            return [False, True][random.randint(0, 1)]
        else:
            return [False, True][value_false < value_true]

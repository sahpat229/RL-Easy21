from easy21 import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random
from rl import *

class SarsaSim():
    def __init__(self, n0, dealer_max, player_max, gamma, lambdah, debug=False):
        self.value_func = ValueFunction(dealer_max, player_max)
        self.n0 = n0
        self.state_action_counter = np.zeros((dealer_max+1, player_max+1, 2))
        self.debug = debug
        self.dealer_max = dealer_max
        self.player_max = player_max
        self.gamma = gamma
        self.lambdah = lambdah

    def find_action(self, state, epsilon):
        explore = np.random.choice([True, False], p=[epsilon, 1-epsilon])
        if explore:
            action = [True, False][random.randint(0, 1)]
        else:
            action = self.value_func.get_action(state)
        return action    

    def sim_step(self):
        game = GameInstance()
        eligibilities = np.zeros((self.dealer_max+1, self.player_max+1, 2))
        state, reward = game.initialState()

        while not state.terminated:
            epsilon = float(self.n0) / (self.n0 + np.sum(self.state_action_counter[state.dealer_score, 
                                                                                   state.player_score, 
                                                                                   :]))
            action = self.find_action(state, epsilon)
            self.state_action_counter[state.dealer_score,
                                      state.player_score,
                                      int(action)] += 1
            new_state, reward = game.step(action)
            new_action = self.find_action(new_state, epsilon)
            delta = reward + self.gamma*self.value_func.get_state_action(new_state, new_action) - \
                self.value_func.get_state_action(state, action)
            eligibilities[state.dealer_score, state.player_score, int(action)] += 1
            n = self.state_action_counter[state.dealer_score,
                                          state.player_score,
                                          int(action)]
            self.value_func.q = self.value_func.q + (1/n)*delta*eligibilities
            eligibilities = self.gamma*self.lambdah*eligibilities
            state = new_state
            action = new_action

        return reward

    def get_deviation(self, other):
        total_dev = 0
        for dealer_score in range(1, 11):
            for player_score in range(1, 22):
                state = State(player_score=player_score,
                              dealer_score=dealer_score)
                dev = (self.value_func.get_state_action(state, True) - \
                       other.value_func.get_state_action(state, True))**2
                dev += (self.value_func.get_state_action(state, False) - \
                       other.value_func.get_state_action(state, False))**2
                total_dev += dev
        mean_dev = total_dev / (10*21*2)
        return mean_dev

class SarsaLinearSim():
    def __init__(self, epsilon, alpha, gamma, lambdah, debug=False):
        self.value_func = LinearValueFunc()
        self.debug = debug
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.lambdah = lambdah

    def find_action(self, state, epsilon):
        explore = np.random.choice([True, False], p=[epsilon, 1-epsilon])
        if explore:
            action = [True, False][random.randint(0, 1)]
        else:
            action = self.value_func.get_action(state)
        return action    

    def sim_step(self):
        game = GameInstance()
        eligibility = 0
        state, reward = game.initialState()

        while not state.terminated:
            action = self.find_action(state, self.epsilon)
            new_state, reward = game.step(action)
            new_action = self.find_action(new_state, self.epsilon)
            delta = reward + self.gamma*self.value_func.get_state_action(new_state, new_action) - \
                self.value_func.get_state_action(state, action)
            eligibility = self.gamma*self.lambdah*eligibility + self.value_func.featurize(state, action)
            self.value_func.weights += self.alpha*delta*eligibility
            state = new_state
            action = new_action

        return reward

    def get_deviation(self, other):
        total_dev = 0
        for dealer_score in range(1, 11):
            for player_score in range(1, 22):
                state = State(player_score=player_score,
                              dealer_score=dealer_score)
                dev = (self.value_func.get_state_action(state, True) - \
                       other.value_func.get_state_action(state, True))**2
                dev += (self.value_func.get_state_action(state, False) - \
                       other.value_func.get_state_action(state, False))**2
                total_dev += dev
        mean_dev = total_dev / (10*21*2)
        return mean_dev
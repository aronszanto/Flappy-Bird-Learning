# Get relative distance to center of pipe, abs. dist to left of pipe. Round both to nearest 10 or 5 or whatever.
# Also get some information about velocity (i.e. is bird falling or rising)
# Conctenate into something hashable: rel_x...rel_y...vel...
# Return the hashable representation of the states.

# def discretize(self. x, self.y):
# nextPipe = flappy.getRandomPipe()
# upperPipe = nextPipe[0]
# lowerPipe = nextPipe[1]
# y_coord = lowerPipe['y'] - PIPEGAPSIZE + 37
# playerMidPos = state.x + IMAGES['player'][0].get_width() / 2
# xPreRound = state.y - y_coord
# yPreRound = state.x - pipeMidPos
# rel_x = round(xPreRound, 2)
# rel_y = round(yPreRound, 2)
from collections import defaultdict
import random
FALL, FLAP = 0, 1


class QLearningAgeny:

    def __init__(self):
        self.q_values = defaultdict(float)
        self.epsilon = 0.1
        self.alpha = 0.8
        self.gamma = 0.8
        self.actions = list([FALL, FLAP])

    def get_q_value(self, state, action):
        return self.q_values[state, action]

    def get_value(self, state):
        return max(self.get_q_value(state, FALL), self.get_q_value(state, FLAP))

    def get_greedy_action(self, state):
        return FALL if self.get_q_value(state, FALL) >= self.get_q_value(state, FLAP) else FLAP

    def get_action(self, state):
        return random.choice(self.actions) if random.random() < self.epsilon else self.get_greedy_action(state)

    def update(self, state, action, next_state, reward):
        q = self.get_q_value(state, action)
        q_ = q + self.alpha * (reward + self.gamma * self.get_value(next_state) - q)
        self.q_values[(state, action)] = q_



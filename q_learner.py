from collections import defaultdict
import random
import os
import pickle
import math

FALL, FLAP = 0, 1


class QLearner:

    def __init__(self, path=None, ld=0):
        self.q_values = self.import_q_values(path) if path else defaultdict(float)
        self.epsilon = 0.5
        self.alpha = 0.5
        self.gamma = 0.8
        self.actions = list([FALL, FLAP])
        self.episodes = 0
        self.history = list()
        self.ld = ld
        self.grid_unit = 20
        self.death_value = -20
        self.dump_interval = 20

    def get_current_epsilon(self):
        return self.epsilon * math.exp(-self.episodes / 0.01)

    def import_q_values(self, path):
        if os.path.isfile(path):
            with open(path) as infile:
                self.q_values = pickle.load(infile)

    def dump_q_values(self, path):
        with open(path, 'w') as outfile:
            pickle.dump(self.q_values, outfile)

    def get_q_value(self, state, action):
        return self.q_values[state, action]

    def get_value(self, state):
        return max([self.get_q_value(state, action) for action in self.actions]) if state else -1

    def get_greedy_action(self, state):
        return FALL if self.get_q_value(state, FALL) >= self.get_q_value(state, FLAP) else FLAP

    def get_action(self, state):
        action = random.choice(self.actions) if random.random() < self.get_current_epsilon() else self.get_greedy_action(state)
        self.history.append((state, action))
        return action

    def update(self, state, action, next_state, reward):
        q = self.get_q_value(state, action)
        q_ = q + self.alpha * (reward + self.gamma * self.get_value(next_state) - q)
        self.q_values[state, action] = q_

    def assign_credit(self, t, n, r=1.0):
        for t_ in range(t, t - n, -1):
            s, a = self.history[t_]
            s_ = self.history[t_ + 1][0] if t_ + 1 < len(self.history) else None
            if not s_:
                r = self.death_value
            self.update(s, a, s_, r / n)

    def learn_from_episode(self):
        num_actions = len(self.history)
        for t in range(num_actions):
            n = min(self.ld, t) + 1
            self.assign_credit(t, n)
        self.history = list()
        self.episodes += 1

        if self.episodes % self.dump_interval == 0:
            self.dump_q_values("training.pkl")

    def extract_state(self, x_offset, y_offset, y_vel):
        return (
                x_offset - x_offset % self.grid_unit,
                y_offset - y_offset % self.grid_unit,
                y_vel,
               )

    def take_action(self, game_state):
        state = self.extract_state(*game_state)
        return self.get_action(state)

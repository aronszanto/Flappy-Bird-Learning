from collections import defaultdict
import random
import os
import pickle
import sys

FALL, FLAP = 0, 1


class QLearner:

    def __init__(self, path=None, ld=0):
        self.q_values = self.import_q_values(path) if path else defaultdict(float)
        self.epsilon = 0.01
        self.alpha = 1
        self.gamma = 1
        self.actions = list([FALL, FLAP])
        self.episodes = 0
        self.history = list()
        self.ld = ld
        self.y_unit = 30  # [-125, 160] potential values
        self.x_unit = 40  # [30, 430] potential values
        self.v_unit = 1   # [-10. 10] potential values
        self.death_value = -1000
        self.dump_interval = 50
        self.max_episodes = 3000

    def off_policy(self):
        # TODO exponential cooling
        return random.random() < self.epsilon

    def import_q_values(self, path):
        if os.path.isfile(path):
            with open(path) as infile:
                self.q_values = pickle.load(infile)

    def dump_q_values(self, path):
        with open(path, 'w') as outfile:
            pickle.dump(self.q_values, outfile)

    def get_q_value(self, state, action):
        return self.q_values[state, action]

    def set_q_value(self, state, action, q_):
        self.q_values[state, action] = q_

    def get_value(self, state):
        # TODO What should value of a terminal state be?
        return max([self.get_q_value(state, action) for action in self.actions]) if state else 0

    def get_greedy_action(self, state):
        return FALL if self.get_q_value(state, FALL) >= self.get_q_value(state, FLAP) else FLAP

    def get_action(self, state):
        action = random.choice(self.actions) if self.off_policy() else self.get_greedy_action(state)
        self.history.append((state, action))
        return action

    def calculate_reward(self, state, n=1.0):

        reward = 1.0

        if not state:
            reward = self.death_value

        rel_y = state[1]
        if -20 <= rel_y <= 20:
            reward = 25.0
        elif -40 <= rel_y <= 40:
            reward = 15.0

        return reward * n

    def update(self, state, action, next_state, n=1.0):
        reward = self.calculate_reward(next_state, n=n)
        q = self.get_q_value(state, action)
        q_ = q + self.alpha * (reward + self.gamma * self.get_value(next_state) - q)
        self.set_q_value(state, action, q_)

    def assign_credit(self, t, n):
        for t_ in range(t, t - n, -1):
            s, a = self.history[t_]
            s_ = self.history[t_ + 1][0] if t_ + 1 < len(self.history) else None
            self.update(s, a, s_, n=n)

    def learn_from_episode(self):
        num_actions = len(self.history)
        for t in range(num_actions):
            n = min(self.ld, t) + 1
            self.assign_credit(t, n)

        # Clear episode's history
        self.history = list()
        self.episodes += 1

        if self.episodes % self.dump_interval == 0:
            self.dump_q_values("training.pkl")

        if self.episodes == self.max_episodes + 1:
            sys.exit()

    def extract_state(self, x_offset, y_offset, y_vel):
        return (
                x_offset - x_offset % self.x_unit,
                y_offset - y_offset % self.y_unit,
                y_vel - y_vel % self.v_unit,
               )

    def take_action(self, game_state):
        state = self.extract_state(*game_state)
        return self.get_action(state)

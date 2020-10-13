import random


class RandomAgent(object):
    def __init__(self):
        self.action_space = ['left', 'right', 'forward',
                             'backward', 'up', 'down']  # in TreasureCube
        self.Q = []

    def take_action(self, state):
        action = random.choice(self.action_space)
        return action

    # implement your train/update function to update self.V or self.Q
    # you should pass arguments to the train function
    def train(self, state, action, next_state, reward):
        pass

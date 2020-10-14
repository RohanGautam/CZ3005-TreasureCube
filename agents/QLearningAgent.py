from itertools import product as cartesianProduct
import random


class QLearningAgent(object):
    def __init__(self):
        self.action_space = ['left', 'right', 'forward',
                             'backward', 'up', 'down']
        self.states = [''.join(map(str, x))
                       for x in cartesianProduct([0, 1, 2, 3], repeat=3)]

    def take_action(self, state) -> str:
        # pass
        return random.choice(self.action_space)
        # action000

    # implement your train/update function to update self.V or self.Q
    # you should pass arguments to the train function
    def train(self, state: str, action: str, next_state: str, reward: int) -> None:
        # `next_state` might not always be a result of `action` on `state`
        #  `next_state` is tied to `reward`
        # calculate Q:
        pass

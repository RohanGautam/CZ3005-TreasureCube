from itertools import product as cartesianProduct
import random


class QLearningAgent(object):
    def __init__(self):
        self.action_space = ['right', 'left', 'forward',
                             'backward', 'up', 'down']
        self.opposites = {
            'left': 'right',
            'right': 'left',
            'forward': 'backward',
            'backward': 'forward',
            'up': 'down',
            'down': 'up',
        }
        self.states = [''.join(map(str, x))
                       for x in cartesianProduct([0, 1, 2, 3], repeat=3)]
        self.action_index = {a: i for i, a in enumerate(self.action_space)}
        self.index_action = {i: a for i, a in enumerate(self.action_space)}
        self.Q = {state: [0] * len(self.action_space) for state in self.states}

        # how important we should consider future rewards(very)
        self.gamma = 0.9
        # learning rate
        self.alpha = 0.5

        # controls random exploration. not used here, as it is modelled in the
        # fact that state transition doesnt always go in the intended direction
        # self.epsilon = 0.2

    def take_action(self, state) -> str:
        return self.index_action[self.Q[state].index(max(self.Q[state]))]
        # randomly explore `self.epsilon` * 100 of the time
        # if random.uniform(0, 1) < self.epsilon:
        #     return random.choice(self.action_space)
        # else:
        #     return self.index_action[self.Q[state].index(max(self.Q[state]))]

    def train(self, state: str, action: str, next_state: str, reward: int) -> None:
        '''
        `next_state` might not always be a result of `action` on `state`.\n
        `reward` is tied to `next_state`       
        '''
        # Q(S,A) = self.Q[state][self.action_index[action]]
        Qold = self.Q[state][self.action_index[action]]
        self.Q[state][self.action_index[action]] = \
            Qold + \
            self.alpha * \
            (reward + self.gamma*(max(self.Q[next_state])) - Qold)

        print('', end='')

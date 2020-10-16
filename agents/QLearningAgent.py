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
        self.Q = {state: [0]*len(self.action_space) for state in self.states}
        self.gamma = 0.9
        self.alpha = 0.01
        self.epsilon = 0.2

    def getPerpendicularActions(self, action: str) -> list:
        return [a for a in self.action_space if a != self.opposites[action] and a != action]

    def getSurroundingStates(self, state: str, directions: list) -> dict:
        state = list(map(int, state))
        totalStates = {}
        for d in directions:
            s = state[:]
            assert d in self.action_space
            # y axis modified
            if d == 'left':
                s[1] -= 1
            elif d == 'right':
                s[1] += 1
            # z axis modified
            elif d == 'up':
                s[2] += 1
            elif d == 'down':
                s[2] -= 1
            # x axis modified
            elif d == 'forward':
                s[0] += 1
            elif d == 'backward':
                s[0] -= 1
            # append if it's a valid coordinate
            if all([0 <= i <= 3 for i in s]):
                totalStates[''.join(map(str, s))] = d
        return totalStates

    def maxRewardOnAction(self, state: str):
        states_can_travel = self.getSurroundingStates(
            state, self.action_space)
        # check if the states can go to terminal
        if '333' in states_can_travel:
            return 1
        else:
            return -0.1

    def actionOutcome(self, state: str, action: str) -> str:
        org = state
        state = list(map(int, state))
        # y axis modified
        if action == 'left':
            state[1] -= 1
        elif action == 'right':
            state[1] += 1
        # z axis modified
        elif action == 'up':
            state[2] += 1
        elif action == 'down':
            state[2] -= 1
        # x axis modified
        elif action == 'forward':
            state[0] += 1
        elif action == 'backward':
            state[0] -= 1

        if all([0 <= i <= 3 for i in state]):
            return ''.join(map(str, state))
        else:
            return org

    def take_action(self, state) -> str:
        # randomly explore 20% of the time
        # if random.uniform(0, 1) < self.epsilon:
        #     return random.choice(self.action_space)
        # else:
        return self.index_action[self.Q[state].index(max(self.Q[state]))]
        # return self.index_action[self.Q[state].index(max(self.Q[state]))]

    # implement your train/update function to update self.V or self.Q
    # you should pass arguments to the train function
    def train(self, state: str, action: str, next_state: str, reward: int) -> None:
        '''
        `next_state` might not always be a result of `action` on `state`.\n
        `reward` is tied to `next_state`       
        '''
        # calculate Q:
        # Q(S,A) = self.Q[state][self.action_index[action]]
        # if reward == 1:
        #     print('reward!')

        # only if it moves as intended
        if self.actionOutcome(state, action) == next_state:
            Qold = self.Q[state][self.action_index[action]]
            self.Q[state][self.action_index[action]] = \
                Qold + \
                self.alpha * \
                (reward + self.gamma*(max(self.Q[next_state])) - Qold)

        print('', end='')

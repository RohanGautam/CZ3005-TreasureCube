from itertools import product as cartesianProduct


class ValueIterationAgent(object):
    def __init__(self):
        self.action_space = ['left', 'right', 'forward',
                             'backward', 'up', 'down']  # in TreasureCube
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
        # self.state_num = len(self.states)  # in TreasureCube, 64 coordinates
        # initialise each state as a key with an empty list as the value
        self.Q = dict(zip(self.states, [[]]*len(self.states)))
        # initialise with zeroes
        self.V = [0] * len(self.states)
        self.discountFactor = 0.5

    def getPerpendiculars(self, action: str) -> list:
        return [a for a in self.action_space if a != self.opposites[action] and a != action]

    def getPerpendicularCoords(self, state: str, action: str) -> list:
        state = list(map(int, state))
        perpStates = []
        directions = self.getPerpendiculars(action)
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
                perpStates.append(s)
        return [''.join(map(str, c)) for c in perpStates]

    def take_action(self, state) -> str:
        '''this should be based on the Value table, and should model the probabilities of taking others'''
        return 'pass'

    # implement your train/update function to update self.V or self.Q
    # you should pass arguments to the train function
    # stronger typing for clarity
    def train(self, state: str, action: str, next_state: str, reward: int) -> None:
        '''
        Train the agent.\n
        @param `state` : A string of length 3 containing the coordinates, eg: "000", "121"\n
        @param `action` : A string from the action space, eg: "left", "right"\n
        @param `next_state` : A string of length 3 containing the coordinates for the next state, eg: "000", "121"\n
        @param `reward` : An integer representing the reward during transition from `state` to `next_state`, eg: -0.1
        '''
        # find the tranisition fn(s) for the current state move.
        # remember, it's nondeterministic so it might not always happen

        # We have transition probabilities (0.1 for perpendicular states)
        # We have discount factor
        # We know reward (-0.1), passed to us
        # we can get expected rewards from V0, my max(Q) at that coord
        # We need to know the coords of the perpendicular directions

        # Q[state] =


def test(test, expected, errMsg, show=False):
    print("passed" if test == expected else errMsg)
    if (show):
        print(test)


# testing
if __name__ == '__main__':
    agent = ValueIterationAgent()
    # treasure cube
    test(len(agent.states), 64, "[test] invalid number of states")
    # getPerpendiculars
    test(agent.getPerpendiculars('left'), [
         'forward', 'backward', 'up', 'down'], "[test] wrong perpendiculars", show=False)
    # getPerpendicularCoords
    test(sorted(agent.getPerpendicularCoords('000', 'left')), sorted(
        ['001', '100']), "[test] wrong perpendiculars", show=True)
    test(sorted(agent.getPerpendicularCoords('001', 'right')), sorted(
        ['000', '002', '101']), "[test] wrong perpendiculars", show=True)

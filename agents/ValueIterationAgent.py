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
        # self.Q = dict(zip(self.states, [[0]]*len(self.states)))
        # self.Q = {state: [-1000] for state in self.states}
        self.V = {state: 0 for state in self.states}
        # initialise with zeroes
        self.discountFactor = 0.5
        self.pi = {}

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

    def getPerpendicularStates(self, state: str, action: str) -> dict:
        return self.getSurroundingStates(state, self.getPerpendicularActions(action))

    def getAllStatesSurrounding(self, state: str) -> dict:
        return self.getSurroundingStates(state, self.action_space)

    def take_action(self, state) -> str:
        '''
            return the best action you can take at that state determined by the policy
        '''
        return self.pi.get(state, 'right')

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
        def succProbReward(s, a):
            # returns (newState, prob, reward)[]
            # for other actions it can take
            other_states_can_reach = self.getPerpendicularStates(s, a)
            l = [(s, 0.1, reward) for s in other_states_can_reach]
            # for intended actions
            l.append((next_state, 0.6, reward))
            return l

        def Q(s, a):
            return sum(prob*(r + self.discountFactor*self.V[newState])
                       for newState, prob, r in succProbReward(s, a))

        isEnd = True if reward == 1 else False
        newV = {}
        actions_at_s = [a for a in self.getPerpendicularActions(action)]
        actions_at_s.append(action)
        for s in self.states:
            if isEnd and s == state:
                newV[s] = 0
                # return
            else:
                newV[s] = max(Q(s, a) for a in actions_at_s)
        # update values
        self.V = newV

        # choose action that maximises Q at that state
        for s in self.states:
            self.pi[s] = max((Q(s, a), a) for a in actions_at_s)[1]


def test(test, expected, errMsg, show=False):
    print("passed" if test == expected else errMsg)
    if (show):
        print(test)


# testing
if __name__ == '__main__':
    # showLogs = True
    showLogs = False

    agent = ValueIterationAgent()
    # treasure cube
    test(len(agent.states), 64,
         "[test] invalid number of states", show=showLogs)
    # getPerpendiculars
    test(agent.getPerpendicularActions('left'), [
         'forward', 'backward', 'up', 'down'], "[test] wrong perpendiculars", show=showLogs)
    # getPerpendicularStates
    test(sorted(agent.getPerpendicularStates('000', 'left').keys()), sorted(
        ['001', '100']), "[test] wrong perpendiculars", show=showLogs)
    test(sorted(agent.getPerpendicularStates('001', 'right').keys()), sorted(
        ['000', '002', '101']), "[test] wrong perpendiculars", show=showLogs)
    # getAllStatesSurrounding
    test(sorted(agent.getAllStatesSurrounding('000').keys()), sorted(
        ['001', '100', '010']), "[test] wrong perpendiculars", show=showLogs)
    test(sorted(agent.getAllStatesSurrounding('111').keys()), sorted(
        ['110', '101', '011', '112', '121', '211']), "[test] wrong perpendiculars", show=showLogs)

import argparse
import matplotlib.pyplot as plt

from environment import TreasureCube
from agents.RandomAgent import RandomAgent
from agents.ValueIterationAgent import ValueIterationAgent


def showPlot(X, Y, xlabel, ylabel):
    plt.plot(X, Y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.show()
    return plt


def test_cube(max_episode, max_step):
    env = TreasureCube(max_step=max_step)
    agent = ValueIterationAgent()
    episode_rewards = []
    for epsisode_num in range(0, max_episode):
        state = env.reset()
        terminate = False
        t = 0
        episode_reward = 0
        while not terminate:
            action = agent.take_action(state)
            reward, terminate, next_state = env.step(action)
            episode_reward += reward
            # you can comment the following two lines, if the output is too much
            # env.render()  # comment
            # print(f'step: {t}, action: {action}, reward: {reward}')  # comment
            t += 1
            agent.train(state, action, next_state, reward)
            state = next_state
        # print(
        #     f'episode: {epsisode_num}, total_steps: {t} episode reward: {episode_reward}')
        episode_rewards.append(episode_reward)

    return showPlot(list(range(max_episode)), episode_rewards,
                    'episode', 'episode rewards')


test_cube(500, 500)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Test')
#     parser.add_argument('--max_episode', type=int, default=500)
#     parser.add_argument('--max_step', type=int, default=500)
#     args = parser.parse_args()

#     test_cube(args.max_episode, args.max_step)

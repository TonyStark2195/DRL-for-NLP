import random
import time

import numpy as np
import pandas as pd


class RandomSearchAgent(object):
    """
    This agent randomly searches the action space and remembers:
     - the best trace and the best final cumulative reward for each state
         - best trace is a list of all game states and actions resulting in the largest cumulative reward
     - list of all endings
     - average reward
     - other useful statistic...

    This agent is very useful for determining various parameters of unknown games.
    """

    def __init__(self):
        random.seed(0)

        self.bestReward = -np.math.inf
        self.bestTrace = []

        # reward and trace for one episode
        self.totalReward = 0
        self.trace = []

        self.endings = []
        # experience is a set of all traces, where a trace is a set of tuples: [state, action, reward]
        self.experience = []

        self.states = []
        self.rewards = []

        self.reset()

    def act(self, state, actions, reward):

        self.totalReward += reward

        action = None
        index = None

        if actions:
            index = random.randint(0, len(actions) - 1)
            action = actions[index]
            # print('choosing action:', action)

        self.trace.append([state, actions, reward, action])
        return index

    def reset(self):

        if not self.trace:
            return

        if self.totalReward > self.bestReward:
            self.bestReward = self.totalReward
            self.bestTrace = self.trace
            print('new best reward : {0:10.3f}'.format(self.bestReward))
            print('new best actions: ', [x[3] for x in self.bestTrace])
            print('last state: ', self.trace[-1][0])
            print('----')
            print(self.trace[-10:])
            print('----')

        for state, _, _, _ in self.trace:
            if state in self.states:
                index = self.states.index(state)
                self.rewards[index] = max(self.totalReward, self.rewards[index])
            else:
                self.states.append(state)
                self.rewards.append(self.totalReward)

                # self.experience.append(self.trace)

        ending = self.trace[-1][0]

        if ending not in self.endings:
            self.endings.append(ending)

        self.totalReward = 0
        self.trace = []


def run_experiment(n_trials, log_folder, simulator, file_name_prefix, episodes, runs_per_episode, max_steps):
    rewards = [
        run(simulator=simulator,
            log_folder=log_folder,
            file_name=f"{i}-{file_name_prefix}",
            episodes=episodes,
            runs_per_episode=runs_per_episode,
            max_steps=max_steps)
        for i in range(n_trials)
    ]

    average_rewards = rewards[0]
    for trial_rewards in rewards[1:]:
        average_rewards = average_rewards + trial_rewards

    print("average-rewards...")
    average_rewards = average_rewards / n_trials
    average_rewards_df = pd.DataFrame(data={"rewards": average_rewards})
    average_rewards_df.to_csv(f"artifacts/{file_name_prefix}-rewards.csv", index=False, header=False)


def run(simulator, log_folder, file_name, episodes, runs_per_episode, max_steps):
    agent = RandomSearchAgent()
    start_time = time.time()

    rewards = []
    words = 0
    descriptions = 0
    positives = 0

    for i in range(episodes):

        episode_rewards = []
        print('episode', i)

        for _ in range(runs_per_episode):

            steps = 0

            while steps < max_steps:

                (text, actions, reward) = simulator.read()
                words += len(text.split())
                descriptions += 1

                player_input = agent.act(text, actions, reward)

                if player_input is None:
                    if agent.totalReward == 0:
                        agent.reset()
                        simulator.restart()
                        steps = 0
                        continue
                    else:
                        if agent.totalReward > 0:
                            positives += 1
                        episode_rewards.append(agent.totalReward)
                        agent.reset()
                        simulator.restart()
                        break

                else:
                    simulator.write(player_input)

                steps += 1

        rewards.append(episode_rewards)
    print('positive rewards', positives)

    flatten_rewards = [reward for episode_rewards in rewards for reward in episode_rewards]
    with open(f"{log_folder}/{file_name}.txt", 'w') as f:
        f.write('{:.2f}'.format(words / descriptions))

    end_time = time.time()
    print("Duration: " + str(end_time - start_time))

    print()
    print('Best rewards for all states:')
    print(len(agent.states), ' states, ', len(agent.rewards), ' rewards')
    print(agent.rewards)
    print(list(zip(agent.rewards, agent.states)))
    print()

    endings = agent.endings

    print('ENDINGS:', len(endings), ' ----------------------------- ')
    for ending in endings:
        print(ending)
        print('*********************')

    return np.array(flatten_rewards)
import gym

import numpy as np
import math

LEARNING_RATE = 0.1
DISCOUNT_RATE = 0.9

class Agent(object):
    def __init__(self, state_size=None, number_of_actions=1):
        self.state_size = state_size
        self.number_of_actions = number_of_actions

        # self.number_of_policy = self.state_size * self.number_of_actions
        self.actions = np.arange(0, 1, 1/self.number_of_actions) + 1/self.number_of_actions
        self.theta_mu = np.zeros(state_size)
        self.theta_sigma = np.zeros(1)

    def get_action(self, state):
        mu = np.sum(state * self.theta_mu) / self.state_size
        sigma = 1/(1+np.exp(self.theta_sigma))
        action = np.random.normal(mu, sigma)
        return 0 if action < 0 else 1
        # for i,v in np.ndenumerate(self.actions):
        #     if action < v:
        #         return i[0]
    
    def update_model(self, state, action, reward, next_state):
        TDError = reward + DISCOUNT_RATE * self.get_value(next_state) - self.get_value(state)

        mu = np.sum(state * self.theta_mu) / self.state_size
        self.theta_mu += LEARNING_RATE * TDError * (self.actions[action] - mu) * state

        sigma = 1/(1+np.exp(self.theta_sigma))
        self.theta_sigma += LEARNING_RATE * TDError * ((self.actions[action] - mu)**2 - sigma**2)

    def get_value(self, state):
        return np.sum(self.theta_mu * state) / self.state_size

env = gym.make('CartPole-v0')
env.reset()

total_reward = 0
steps = []
agent = Agent(state_size=env.observation_space.shape[0], number_of_actions=env.action_space.n)

for i_episode in range(200):
    observation = env.reset()
    for t in range(200):
        env.render()
        # print(observation)
        action = agent.get_action(observation)
        # action = env.action_space.sample()
        prev_obseration = observation
        observation, reward, done, info = env.step(action)
        agent.update_model(prev_obseration, action, reward, observation)
        total_reward += reward
        if done:
            break
    print("Episode {} finished after {} timesteps".format(i_episode+1, t+1))
    steps.append(t+1)



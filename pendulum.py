# coding:utf-8

import gym

import numpy as np
import tensorflow as tf

import dqn

LEARNING_RATE = 0.01
DISCOUNT_RATE = 0.99
BATCH_SIZE = 100
EPSILON = 1.0
EPSILON_MIN = 0.0
EPSILON_DECAY = 0.0005
HIDDEN_LAYERS = [100,100,100]
EXP_SIZE = 1000
EXP_SIZE = 1000

EPISODE_NUM = 150
SKIP_TRAIN_COUNT = 5
COPY_WEIGHT_COUNT = 20
TRANING_COUNT = SKIP_TRAIN_COUNT

RESULT_PATH = "result/pendulum"
IS_SAVE = False

env = gym.make('Pendulum-v0')

np.random.seed(100)

# actions = np.asarray([env.action_space.low, env.action_space.high])
distance = 1.0
actions = np.arange(env.action_space.low[0], env.action_space.high[0]+distance, distance)
actions = np.reshape(actions, [len(actions),1])

STATE_NUM = env.observation_space.shape[0]
ACTION_NUM = len(actions)

tf.reset_default_graph()

with tf.Session() as sess:
    agent = dqn.agent(sess, state_size=STATE_NUM, action_num=ACTION_NUM, batch_size=BATCH_SIZE, hidden_layer_size=HIDDEN_LAYERS)

    init = tf.initialize_all_variables()
    sess.run(init)
    agent.copy_network()

    if IS_SAVE:
        env.monitor.start(RESULT_PATH, force=True)
    
    step = 1
    for i_episode in range(EPISODE_NUM):
        observation = env.reset()
        total_reward = 0.0
        for t in range(200):
            # env.render()
            action_index = agent.get_action(observation)
            agent.decrease_epsilon()
            prev_observation = observation
            observation, reward, done, info = env.step(actions[action_index])
            agent.add_experience(prev_observation, action_index, reward, observation, done)
            total_reward += reward
            if EXP_SIZE <= step:
                if step % SKIP_TRAIN_COUNT == 0 or done:
                    for k in range(TRANING_COUNT):
                      agent.train()

                if step % COPY_WEIGHT_COUNT == 0:
                    agent.copy_network()

            step += 1

            if done:
                break
        print("{},{},{}".format(i_episode+1, total_reward, agent.epsilon))
    if IS_SAVE:
        env.monitor.close()


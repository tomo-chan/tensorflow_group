# coding:utf-8

import gym

import numpy as np
import tensorflow as tf

import dqn_image as dqn

LEARNING_RATE = 0.001
DISCOUNT_RATE = 0.99
BATCH_SIZE = 32
EPSILON = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY = 1/100000
HIDDEN_LAYERS = [256]
EXP_SIZE = 100

EPISODE_NUM = 1000
SKIP_TRAIN_COUNT = 4
COPY_WEIGHT_COUNT = 10000
TRANING_COUNT = SKIP_TRAIN_COUNT

RESULT_PATH = "result/breakout"
IS_SAVE = False

env = gym.make('Breakout-v0')

np.random.seed(100)

ACTION_NUM = env.action_space.n

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))

tf.reset_default_graph()

target_input = tf.placeholder(tf.float32, [None, 210, 160, 3])
processed_input = tf.image.resize_images(tf.image.rgb_to_grayscale(target_input), (84, 84))

with tf.Session(config=config) as sess:
    agent = dqn.agent(sess, image_height=84, image_width=84, image_channels=1, action_num=ACTION_NUM, batch_size=BATCH_SIZE, 
    # agent = dqn.agent(sess, image_height=210, image_width=160, image_channels=3, action_num=ACTION_NUM, batch_size=BATCH_SIZE, 
    hidden_layer_size=HIDDEN_LAYERS, cnn_layer_size=[], epsilon_decay=EPSILON_DECAY, epsilon_end=EPSILON_MIN, experience_size=EXP_SIZE)

    init = tf.global_variables_initializer()
    sess.run(init)
    agent.copy_network()

    if IS_SAVE:
        env.monitor.start(RESULT_PATH, force=True)
    
    step = 1
    for i_episode in range(EPISODE_NUM):
        observation = env.reset()
        observation = sess.run(processed_input, feed_dict={target_input: [observation]})
        total_reward = 0.0
        done = False
        while done == False:
#            env.render()
            action = agent.get_action(observation)
            agent.decrease_epsilon()
            prev_observation = observation
            observation, reward, done, info = env.step(action)
            observation = sess.run(processed_input, feed_dict={target_input: [observation]})
            agent.add_experience(prev_observation, action, reward, observation, done)
            total_reward += reward

            # 経験が貯まるまで学習をスキップ
            if EXP_SIZE <= step:
                # 一定間隔で学習する
                if step % SKIP_TRAIN_COUNT == 0 or done:
                    # 連続学習
                    for k in range(TRANING_COUNT):
                      agent.train()

                # 一定間隔で重みをTarget networkにコピー
                if step % COPY_WEIGHT_COUNT == 0:
                    agent.copy_network()

            step += 1

            if done:
                break
        print("{},{},{},{}".format(i_episode+1, step-1, total_reward, agent.epsilon))
    if IS_SAVE:
        env.monitor.close()


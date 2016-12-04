# coding:utf-8

import gym

from collections import deque
import numpy as np
import tensorflow as tf

LEARNING_RATE = 0.01
DISCOUNT_RATE = 0.99
EPISODE_NUM = 150
BATCH_SIZE = 20
EPSILON = 1
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.0001
EXPLORATION = 1000
HIDDEN_LAYERS = [100,100,100]
EXP_SIZE = 1000
SKIP_TRAIN_COUNT = 1
COPY_WEIGHT_COUNT = 20
RESULT_PATH = "result/pendulum"
IS_SAVE = False

env = gym.make('Pendulum-v0')

np.random.seed(100)

replay_memory = deque(maxlen=EXP_SIZE)
actions = np.asarray([env.action_space.low, env.action_space.high])

STATE_NUM = env.observation_space.shape[0]
ACTION_NUM = len(actions)

tf.reset_default_graph()

def leaky_relu(x,alpha=0.2):
    return tf.maximum(alpha*x,x)

def create_q_net(name, input, node_num):
    in_layer = input
    node_num = STATE_NUM
    variables = []
    biases = []
    for i,n in enumerate(HIDDEN_LAYERS):
        with tf.variable_scope(str(name)+'hidden'+str(i)) as scope:
            hidden_w = tf.Variable(tf.truncated_normal([node_num, n], stddev=0.01))
            hidden_b = tf.Variable(tf.zeros([n]))
            out_layer = leaky_relu(tf.matmul(in_layer, hidden_w) + hidden_b)
            in_layer = out_layer
            node_num = n
            variables.append(hidden_w)
            biases.append(hidden_b)

    with tf.variable_scope(str(name)+'q_net') as scope:
        q_net_w = tf.Variable(tf.truncated_normal([node_num, ACTION_NUM], stddev=0.01))
        q_net_b = tf.Variable(tf.zeros([ACTION_NUM]))
        q_net = tf.matmul(in_layer, q_net_w) + q_net_b
        variables.append(q_net_w)
        biases.append(q_net_b)
    
    return q_net, variables, biases

input = tf.placeholder(tf.float32, [None, STATE_NUM])
q_net, var_q, bias_q = create_q_net('q_net', input, STATE_NUM)
target = tf.placeholder(tf.float32, [None, STATE_NUM])
tar_net, var_tar, bias_tar = create_q_net('tar_net', target, STATE_NUM)

q_val = tf.placeholder(tf.float32, [None, ACTION_NUM])
loss = tf.reduce_mean(tf.square(q_val - q_net))
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

tar_ops = []
for v_q,v_tar,b_q,b_tar in zip(var_q, var_tar, bias_q, bias_tar):
    tar_ops.append(v_tar.assign(v_q))
    tar_ops.append(b_tar.assign(b_q))

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    for op in tar_ops:
        sess.run(op)
    epsilon = EPSILON
    if IS_SAVE:
        env.monitor.start(RESULT_PATH, force=True)
    for i_episode in range(EPISODE_NUM):
        observation = env.reset()
        total_reward = 0.0
        for t in range(200):
            env.render()
            if np.random.random() < epsilon:
                action_index = np.random.choice(ACTION_NUM, 1)[0]
            else:
                action_index = np.argmax(sess.run(q_net, feed_dict={input: [observation]}))
            # action = [2.0] if action[0] > 0 else [-2.0]
            if epsilon > EPSILON_MIN:
                epsilon -= EPSILON_DECAY
            prev_observation = observation
            observation, reward, done, info = env.step(actions[action_index])
            replay_memory.append((prev_observation, action_index, reward, observation, done))
            total_reward += reward
            # for k in range(1):
            if EXP_SIZE <= len(replay_memory):
                if (t+1) % SKIP_TRAIN_COUNT == 0 or done:
                    input_batch = []
                    action_batch = []
                    reward_batch = []
                    target_batch = []
                    done_batch = []
                    minibatch_size = min(len(replay_memory), BATCH_SIZE)
                    minibatch_indexes = np.random.choice(len(replay_memory), minibatch_size)
                    for i in minibatch_indexes:
                        s_batch = replay_memory[i][0]
                        a_batch = replay_memory[i][1]
                        r_batch = replay_memory[i][2]
                        ns_batch = replay_memory[i][3]
                        d_batch = replay_memory[i][4]

                        input_batch.append(s_batch)
                        action_batch.append(a_batch)
                        reward_batch.append(r_batch)
                        target_batch.append(ns_batch)
                        done_batch.append(d_batch)

                    input_batch = np.reshape(input_batch, [minibatch_size, STATE_NUM])
                    ns_batch = np.reshape(target_batch, [minibatch_size, STATE_NUM])
                    value_batch = sess.run(q_net, feed_dict={input: input_batch})

                    target_value = sess.run(tar_net, feed_dict={target: ns_batch})
                    target_value_index = np.argmax(target_value, axis=1)
                    for i in range(minibatch_size):
                        value_batch[i][action_batch[i]] = reward_batch[i] if done_batch[i] else reward_batch[i] + DISCOUNT_RATE * target_value[i][target_value_index[i]]

                    sess.run(optimizer, feed_dict={input: input_batch, q_val: value_batch})
                    # print("loss: {}".format(sess.run(loss, feed_dict={input: input_batch, q_val: value_batch})))

                    if (t+1) % COPY_WEIGHT_COUNT == 0:
                        # print('before:{},{}'.format(sess.run(var_q[0])[0][0], sess.run(var_tar[0])[0][0]))
                        for op in tar_ops:
                            sess.run(op)
                        # print('after:{},{}'.format(sess.run(var_q[0])[0][0], sess.run(var_tar[0])[0][0]))

                    if done:
                        break
        print("{},{},{}".format(i_episode+1, total_reward, epsilon))
    if IS_SAVE:
        env.monitor.close()


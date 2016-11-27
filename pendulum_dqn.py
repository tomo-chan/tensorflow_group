import gym

import numpy as np
import math
import tensorflow as tf

replay_memory = []
LEARNING_RATIO = 0.1
DISCOUNT_RATIO = 0.99
EPISODE_NUM = 1000
BATCH_SIZE = 100
EPSILON = 0.1
HIDDEN_LAYERS = [200,200]

env = gym.make('Pendulum-v0')

STATE_NUM = env.observation_space.shape[0]
ACTION_NUM = env.action_space.shape[0]
ACTION_BOUND = env.action_space.high[0]

tf.reset_default_graph()

input = tf.placeholder(tf.float32, [None, STATE_NUM])

in_layer = input
node_num = STATE_NUM
for i,n in enumerate(HIDDEN_LAYERS):
    with tf.variable_scope('hidden'+str(i)) as scope:
        hidden_w = tf.Variable(tf.truncated_normal([node_num, n], stddev=0.01))
        hidden_b = tf.Variable(tf.zeros([n]))
        out_layer = tf.nn.relu(tf.matmul(in_layer, hidden_w) + hidden_b)
        in_layer = out_layer
        node_num = n

with tf.variable_scope('q_net') as scope:
    q_net_w = tf.Variable(tf.truncated_normal([node_num, ACTION_NUM], stddev=0.01))
    q_net_b = tf.Variable(tf.zeros([ACTION_NUM]))
    q_net = tf.matmul(in_layer, q_net_w) + q_net_b

target = tf.placeholder(tf.float32, [None, ACTION_NUM])

loss = tf.reduce_mean(tf.square(target - q_net))
optimizer = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATIO).minimize(loss)

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)

    for i_episode in range(EPISODE_NUM):
        observation = env.reset()
        total_reward = 0.0
        epsilon = EPSILON
        for t in range(200):
            env.render()
            # 初回と2回目以降で配列の形が異なるのでここで整える
            observation = np.reshape(observation,[STATE_NUM])
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = sess.run(q_net, feed_dict={input: [observation]})[0]
            prev_observation = observation
            # epsilon -= 1.0/100000
            observation, reward, done, info = env.step(action)
            replay_memory.append([prev_observation, action, reward, observation, done])
            total_reward += reward
            if (not t == 0 and t % (BATCH_SIZE-1) == 0) or done:
                target_batch = []
                input_batch = []
                minibatch_size = min(len(replay_memory), BATCH_SIZE)
                minibatch_indexes = np.random.randint(0, len(replay_memory), minibatch_size)
                for i in minibatch_indexes:
                    s_batch = replay_memory[i][0]
                    a_batch = replay_memory[i][1]
                    r_batch = replay_memory[i][2]
                    ns_batch = replay_memory[i][3]
                    done_batch = replay_memory[i][4]

                    input_batch.append(s_batch)
                    action_t = sess.run(q_net, feed_dict={input: [ns_batch]})[0]
                    target_batch.append(r_batch if done_batch else r_batch + DISCOUNT_RATIO * action_t)

                sess.run(optimizer, feed_dict={input: input_batch, target: target_batch})
            if done:
                break
        print("{},{}".format(i_episode+1, total_reward))


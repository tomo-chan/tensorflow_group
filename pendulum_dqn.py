import gym

from collections import deque
import numpy as np
import tensorflow as tf

LEARNING_RATE = 0.1
DISCOUNT_RATE = 0.99
EPISODE_NUM = 5000
BATCH_SIZE = 100
EPSILON = 1
EPSILON_DECAY = 0.005
HIDDEN_LAYERS = [100,100,100]
EXP_SIZE = 1000

replay_memory = deque(maxlen=EXP_SIZE)

env = gym.make('Pendulum-v0')

STATE_NUM = env.observation_space.shape[0]
ACTION_NUM = env.action_space.shape[0]
ACTION_BOUND = env.action_space.high[0]

tf.reset_default_graph()

def create_q_net(name, input, node_num):
    in_layer = input
    node_num = STATE_NUM
    variables = []
    biases = []
    for i,n in enumerate(HIDDEN_LAYERS):
        with tf.variable_scope(str(name)+'hidden'+str(i)) as scope:
            hidden_w = tf.Variable(tf.truncated_normal([node_num, n], stddev=0.01))
            hidden_b = tf.Variable(tf.zeros([n]))
            out_layer = tf.nn.relu(tf.matmul(in_layer, hidden_w) + hidden_b)
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
loss = tf.reduce_mean(tf.square(q_net - q_val))
optimizer = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

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
    for i_episode in range(EPISODE_NUM):
        observation = env.reset()
        total_reward = 0.0
        for t in range(200):
            env.render()
            # 初回と2回目以降で配列の形が異なるのでここで整える
            observation = np.reshape(observation,[STATE_NUM])
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = sess.run(q_net, feed_dict={input: [observation]})[0]
            action = [2.0] if action[0] > 0 else [-2.0]
            prev_observation = observation
            epsilon = epsilon - EPSILON_DECAY if epsilon > 0 else epsilon
            observation, reward, done, info = env.step(action)
            replay_memory.append((prev_observation, action, reward, observation, done))
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
                    action_t = sess.run(tar_net, feed_dict={target: [ns_batch]})[0]
                    target_batch.append(r_batch if done_batch else r_batch + DISCOUNT_RATE * action_t)

                sess.run(optimizer, feed_dict={input: input_batch, q_val: target_batch})
                for op in tar_ops:
                    sess.run(op)

            if done:
                break
        print("{},{}".format(i_episode+1, total_reward))


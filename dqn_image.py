# coding:utf-8

from collections import deque
import numpy as np
import tensorflow as tf

class qnetwork(object):

    def __init__(self, image_channels, output_size=1, hidden_layer_size=[100, 100], cnn_layer_size=None):
        self.image_channels    = image_channels
        self.output_size       = output_size
        self.hidden_layer_size = hidden_layer_size
        self.cnn_layer_size    = cnn_layer_size

    def create_network(self, name, input):
        '''
        指定されたパラメータからネットワークを作成します。
        '''
        in_layer = input
        node_num = 0
        variables = []
        biases = []

        if not self.cnn_layer_size == None:
            with tf.variable_scope(str(name)+'_cnn'+str(1)) as scope:
                cnn_w = tf.Variable(tf.truncated_normal([8, 8, self.image_channels, 32], stddev=0.1))
                cnn_b = tf.Variable(tf.zeros([32]))
                conv2d = tf.nn.relu(tf.nn.conv2d(in_layer, cnn_w, strides=[1, 4, 4, 1], padding='SAME') + cnn_b)
                #out_layer = tf.nn.max_pool(conv2d, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                #in_layer = out_layer
                in_layer = conv2d
                node_num = 32
                variables.append(cnn_w)
                biases.append(cnn_b)
            
            with tf.variable_scope(str(name)+'_cnn'+str(2)) as scope:
                cnn_w = tf.Variable(tf.truncated_normal([4, 4, node_num, 64], stddev=0.1))
                cnn_b = tf.Variable(tf.zeros([64]))
                conv2d = tf.nn.relu(tf.nn.conv2d(in_layer, cnn_w, strides=[1, 2, 2, 1], padding='SAME') + cnn_b)
                #out_layer = tf.nn.max_pool(conv2d, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                #in_layer = out_layer
                in_layer = conv2d
                node_num = 64
                variables.append(cnn_w)
                biases.append(cnn_b)

            with tf.variable_scope(str(name)+'_cnn'+str(3)) as scope:
                cnn_w = tf.Variable(tf.truncated_normal([3, 3, node_num, 64], stddev=0.1))
                cnn_b = tf.Variable(tf.zeros([64]))
                conv2d = tf.nn.relu(tf.nn.conv2d(in_layer, cnn_w, strides=[1, 1, 1, 1], padding='SAME') + cnn_b)
                #out_layer = tf.nn.max_pool(conv2d, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                #in_layer = out_layer
                in_layer = conv2d
                node_num = 64
                variables.append(cnn_w)
                biases.append(cnn_b)

            # node_num = 2*2*node_num
            in_layer = tf.reshape(in_layer, [-1, node_num])

        for i, n in enumerate(self.hidden_layer_size):
            with tf.variable_scope(str(name)+'_hidden'+str(i)) as scope:
                hidden_w = tf.Variable(tf.truncated_normal([node_num, n], stddev=0.01))
                hidden_b = tf.Variable(tf.zeros([n]))
                out_layer = self.leaky_relu(tf.matmul(in_layer, hidden_w) + hidden_b)
                in_layer = out_layer
                node_num = n
                variables.append(hidden_w)
                biases.append(hidden_b)

        with tf.variable_scope(str(name)+'_q_net') as scope:
            q_net_w = tf.Variable(tf.truncated_normal([node_num, self.output_size], stddev=0.01))
            q_net_b = tf.Variable(tf.zeros([self.output_size]))
            q_net = tf.matmul(in_layer, q_net_w) + q_net_b
            variables.append(q_net_w)
            biases.append(q_net_b)
        
        return q_net, variables, biases

    def leaky_relu(self, x, alpha=0.2):
        '''
        Leaky Leru関数を実行します。
        '''
        return tf.maximum(alpha*x,x)

class agent(object):
  
    def __init__(self, sess, image_height, image_width, image_channels, action_num, learning_rate=0.01, discount_rate=0.99, batch_size=100,
            epsilon=1, epsilon_end=0, epsilon_decay=0.0005, experience_size=1000, hidden_layer_size=[100,100],
            cnn_layer_size=None):
        self.sess              = sess               # TensorflowのSession
        self.image_height      = image_height       # 入力状態数
        self.image_width       = image_width        # 入力状態数
        self.image_channels    = image_channels     # 入力状態数
        self.action_num        = action_num         # 行動数
        self.learning_rate     = learning_rate      # 学習率
        self.discount_rate     = discount_rate      # 割引率
        self.batch_size        = batch_size         # 学習のバッチサイズ
        self.epsilon           = epsilon            # ε-greedyのε
        self.epsilon_end       = epsilon_end        # εの最小値
        self.epsilon_decay     = epsilon_decay      # εの減衰率
        self.experience_size   = experience_size    # Replay experienceに蓄積するメモリ数
        self.hidden_layer_size = hidden_layer_size  # Q networkの各隠れ層のノード数
        self.cnn_layer_size    = cnn_layer_size     # Q networkの各CNN層の設定

        self.D = deque(maxlen=self.experience_size) # Replay experience
        
        # Tensorflowでは最初にネットワークを定義する必要がある。

        model = qnetwork(self.image_channels, self.action_num, self.hidden_layer_size, self.cnn_layer_size)

        # Q networkとTarget networkを作成する
        self.input_state = tf.placeholder(tf.float32, [None, self.image_height, self.image_width, self.image_channels])
        self.q_network, var_q, bias_q = model.create_network("q_net", self.input_state)
        self.target_state = tf.placeholder(tf.float32, [None, self.image_height, self.image_width, self.image_channels])
        self.target_network, var_tar, bias_tar = model.create_network("target_net", self.target_state)

        # Q networkからTarget networkに重みとバイアスをコピーする
        tar_ops = []
        for v_q,v_tar,b_q,b_tar in zip(var_q, var_tar, bias_q, bias_tar):
            tar_ops.append(v_tar.assign(v_q))
            tar_ops.append(b_tar.assign(b_q))
        self.target_ops = tar_ops

        # 誤差計算
        self.y_value = tf.placeholder(tf.float32, [None, self.action_num])
        loss = tf.reduce_mean(tf.square(self.y_value - self.q_network))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)
    
    def train(self):
        '''
        ネットワークをトレーニングする
        '''
        s, a, r, ns, done = self.get_experiences() 
        q_value = self.sess.run(self.q_network, feed_dict={self.input_state: s}) # 経験値のQ値を取得する

        target_value = self.sess.run(self.target_network, feed_dict={self.target_state: ns}) # Target networkからQ値を取得する
        target_value_index = np.argmax(target_value, axis=1)
        for i in range(self.batch_size):
            # 教師信号を計算する
            q_value[i][a[i]] = r[i] if done[i] else r[i] + self.discount_rate * target_value[i][target_value_index[i]]

        # 誤差計算
        self.sess.run(self.optimizer, feed_dict={self.input_state: s, self.y_value: q_value})

    def copy_network(self):
        for op in self.target_ops:
            self.sess.run(op)
    
    def get_action(self, state):
        '''
        ε-greedy法で行動選択する
        '''
        if np.random.random() < self.epsilon:
            action_index = np.random.choice(self.action_num, 1)[0]
        else:
            action_index = np.argmax(self.sess.run(self.q_network, feed_dict={self.input_state: [state]}))
        return action_index
    
    def decrease_epsilon(self):
        '''
        εを減衰させます
        '''
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay
    
    def add_experience(self, state, action, reward, next_state, done):
        self.D.append((state, action, reward, next_state, done))

    def get_experiences(self):
        '''
        蓄積された経験からランダムにバッチ数分取得します
        '''
        input_batch = []
        action_batch = []
        reward_batch = []
        target_batch = []
        done_batch = []
        batch_indexes = np.random.choice(len(self.D), self.batch_size)
        for i in batch_indexes:
            s_batch = self.D[i][0]
            a_batch = self.D[i][1]
            r_batch = self.D[i][2]
            ns_batch = self.D[i][3]
            d_batch = self.D[i][4]

            input_batch.append(s_batch)
            action_batch.append(a_batch)
            reward_batch.append(r_batch)
            target_batch.append(ns_batch)
            done_batch.append(d_batch)

        input_batch = np.reshape(input_batch, [self.batch_size, self.image_height, self.image_width, self.image_channels])
        target_batch = np.reshape(target_batch, [self.batch_size, self.image_height, self.image_width, self.image_channels])

        return input_batch, action_batch, reward_batch, target_batch, done_batch

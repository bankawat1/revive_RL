import tensorflow as tf

class QNetwork:
    def __init__(self,action_repeat, learning_rate=0.01, state_size=10, # 9 states
                 action_size=4, action_split = 3, hidden_size=10, 
                 name='QNetwork'):
        # state inputs to the Q-network
        with tf.variable_scope(name):
            self.inputs_ = tf.placeholder(tf.float32, [None, state_size*action_repeat], name='inputs')
            
            # One hot encode the actions to later choose the Q-value for the action
            self.actions_ = tf.placeholder(tf.int32, [None], name='actions')
            
            #convert actions to their corresponding index...
            one_hot_actions = tf.one_hot(self.actions_, action_split)
            
            # Target Q values for training
            self.targetQs_ = tf.placeholder(tf.float32, [None], name='target')
            
            self.is_train = tf.placeholder(tf.bool, name ='is_Train')
            
            # ReLU hidden layers
            #norm_input = tf.layers.batch_normalization(self.inputs_,training=self.is_train)
            self.fc1 = tf.contrib.layers.fully_connected(self.inputs_, hidden_size,activation_fn=tf.nn.elu)
            
            #norm_fc1 = tf.layers.batch_normalization(self.fc1,training=self.is_train)                          
            self.fc2 = tf.contrib.layers.fully_connected(self.fc1, hidden_size,activation_fn=tf.nn.elu)
                                      
            #norm_fc2 = tf.layers.batch_normalization(self.fc2,training=self.is_train)
            self.fc3 = tf.contrib.layers.fully_connected(self.fc2, hidden_size,activation_fn=tf.nn.elu)

            # Linear output layer
            self.output = tf.contrib.layers.fully_connected(self.fc3, action_split, 
                                                            activation_fn=None)
            
            ### Train with loss (targetQ - Q)^2
            # output has length 2, for two actions. This next line chooses
            # one value from output (per row) according to the one-hot encoded actions.
            self.Q = tf.reduce_sum(tf.multiply(self.output, one_hot_actions), axis=1)
            
            self.loss = tf.reduce_mean(tf.square(self.targetQs_ - self.Q))
            
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            gvs = optimizer.compute_gradients(self.loss)
            capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
            self.opt = optimizer.apply_gradients(capped_gvs)

            #self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
import numpy as np
from task import Task
from q_network import QNetwork
from memory import Memory

class My_Agent():
    def __init__(self, task):
        # Task (environment) information
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_split = task.action_split
        self.action_repeat = task.action_repeat
        
        self.action_low = task.action_low
        self.action_high = task.action_high
        self.action_range = self.action_high - self.action_low

        # Episode variables
        self.reset_episode()
        self.explore_stop = 0.01
        self.explore_start = 1.0
        self.decay_rate = 0.00001  
        self.gamma = 0.99
        
        self.total_steps = 0
        
        # Network parameters
        hidden_size = 64               # number of units in each Q-network hidden layer
        learning_rate = 0.00001         # Q-network learning rate
        
        #initialize QNetwork class here...
        self.q_net = QNetwork(hidden_size=hidden_size, learning_rate=learning_rate, action_split = self.action_split,action_repeat = self.action_repeat)
        

        memory_size = 10000            # memory capacity
        self.batch_size = 60                # experience mini-batch size
        self.pretrain_length = self.batch_size   # number experiences to pretrain the memory
        
        self.memory = Memory(max_size=memory_size)
        
        self.loss_list = []

    def reset_episode(self):
        self.episode_reward = 0.0
        self.count = 0
        state = self.task.reset()
        self.initial_state = state
        return state

    def step(self,state, action, reward, next_state, done,sess):
        # Save experience / reward
        self.episode_reward += reward
        self.count += 1        
        
        # Add experience to memory...
        self.memory.add((state, action, reward, next_state))
        
        # Learn and Update Network parameters...
        self.learn(sess)
            

    def act(self, state,sess,is_training=True):
        # Choose action based on given state and policy
        self.total_steps += 1
        
        if is_training:
            # Explore or Exploit
            explore_p = self.explore_stop + (self.explore_start - self.explore_stop)*np.exp(-self.decay_rate*self.total_steps) 
            if explore_p > np.random.rand():
                # Make a random action
                #action = np.random.choice(np.arange(self.action_split),size=4)
                action = [np.random.choice(self.action_split)]*4
            else:
                # Get action from Q-network
                feed = {self.q_net.inputs_: state.reshape((1, *state.shape)),self.q_net.is_train : False}
                Qs = sess.run(self.q_net.output, feed_dict=feed)
                max_Qval_index = np.argmax(Qs)
                #action = list(self.task.dict_index_to_levels[max_Qval_index])
                action = np.array([max_Qval_index]*4)
        else:
            # Get action from Q-network
            feed = {self.q_net.inputs_: state.reshape((1, *state.shape)),self.q_net.is_train : False}
            Qs = sess.run(self.q_net.output, feed_dict=feed)
            max_Qval_index = np.argmax(Qs)
            #action = list(self.task.dict_index_to_levels[max_Qval_index])
            action = np.array([max_Qval_index]*4)
        
        return action

    def learn(self,sess):
        #use q_net class to update weights after a step by agent.
        
        # Sample mini-batch from memory
        batch = self.memory.sample(self.batch_size)
        states = np.array([each[0] for each in batch])
        actions = np.array([each[1] for each in batch])
        rewards = np.array([each[2] for each in batch])
        next_states = np.array([each[3] for each in batch])
        
        #idx_list = [self.task.dict_levels_to_index[tuple(val)] for val in actions]
        idx_list = [val[0] for val in actions]
        
        actions = np.array(idx_list)
        # Train network
        target_Qs = sess.run(self.q_net.output, feed_dict={self.q_net.inputs_: next_states,self.q_net.is_train : False})
        
        # Set target_Qs to 0 for states where episode ends
        episode_ends = (next_states == self.initial_state).all(axis=1)
        target_Qs[episode_ends] = np.zeros(self.action_split)

        targets = rewards + self.gamma * np.max(target_Qs, axis=1)

        loss, _ = sess.run([self.q_net.loss, self.q_net.opt],
                            feed_dict={self.q_net.inputs_: states,
                                       self.q_net.targetQs_: targets,
                                       self.q_net.actions_: actions,self.q_net.is_train : True})

        self.loss_list.append(loss)

        
    def pretrain_memory(self):
        # Make a bunch of random actions and store the experiences
        #action = np.random.choice(np.arange(self.action_split),size=4)
        action = [np.random.choice(self.action_split)]*4
        state, r, d = self.task.step(action)
        
        for ii in range(self.pretrain_length):

            # Make a random action
            #action = np.random.choice(np.arange(self.action_split),size=4)
            action = [np.random.choice(self.action_split)]*4
            next_state, reward, done = self.task.step(action)
            self.memory.add((state, action, reward, next_state))
            state = next_state
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
           # Learn by random policy search, using a reward-based score
#         self.score = self.total_reward / float(self.count) if self.count else 0.0
#         if self.score > self.best_score:
#             self.best_score = self.score
#             self.best_w = self.w
#             self.noise_scale = max(0.5 * self.noise_scale, 0.01)
#         else:
#             self.w = self.best_w
#             self.noise_scale = min(2.0 * self.noise_scale, 3.2)
#         self.w = self.w + self.noise_scale * np.random.normal(size=self.w.shape)  # equal noise in all directions
                 
            
            
            
            
            
#             if done:
#                 # The simulation fails so no next state
# #                 next_state = np.zeros(state.shape)
#                 # Add experience to memory
#                 memory.add((state, action, reward, next_state))

#                 # Start new episode
# #                 env.reset()
#                 # Take one random step to get the pole and cart moving
# #                 state, reward, done, _ = env.step(env.action_space.sample())
#             else:
#                 # Add experience to memory
#                 memory.add((state, action, reward, next_state))
#                 state = next_state


#      state = np.append(state,-1) #this initializes rotor number at the end...
#             for i in range(self.action_size):
#                 state[-1] = i #this updates rotor number at the end...
#                 feed = {self.q_net.inputs_: state.reshape((1, *state.shape))}
#                 Qs = sess.run(self.q_net.output, feed_dict=feed)
#                 action[i] = np.argmax(Qs)

#         self.w = np.random.normal(
#             size=(self.state_size, self.action_size),  # weights for simple linear policy: state_space x action_space
#             scale=(self.action_range / (2 * self.state_size))) # start producing actions in a decent range

        # Score tracker and learning parameters
#         self.best_w = None
#         self.best_score = -np.inf
#         self.noise_scale = 0.1
        
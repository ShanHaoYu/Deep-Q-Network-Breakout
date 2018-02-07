from agent_dir.agent import Agent

import os
import random
import numpy as np
import h5py
import tensorflow as tf
from collections import deque
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Conv2D, Flatten, Dense, LeakyReLU, Multiply, Maximum, Add, merge
from keras.optimizers import RMSprop,Adam
import keras.backend as K
from keras.layers import Lambda

from keras.backend.tensorflow_backend import set_session

random.seed(1)
np.random.seed(1)
tf.reset_default_graph()
tf.set_random_seed(1)

# reference : https://github.com/tokb23/dqn/blob/master/dqn.py

class Agent_DQN(Agent):
    def __init__(self, env, args):
        super(Agent_DQN,self).__init__(env)

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = args.gpu_frac
        set_session(tf.Session(config=config))
        
        # parameters
        self.frame_width = args.frame_width
        self.frame_height = args.frame_height
        self.num_steps = args.num_steps
        self.state_length = args.state_length
        self.gamma = args.gamma
        self.exploration_steps = args.exploration_steps
        self.initial_epsilon = args.initial_epsilon
        self.final_epsilon = args.final_epsilon
        self.initial_replay_size = args.initial_replay_size
        self.num_replay_memory = args.num_replay_memory
        self.batch_size = args.batch_size
        self.target_update_interval = args.target_update_interval
        self.train_interval = args.train_interval
        self.learning_rate = args.learning_rate
        self.min_grad = args.min_grad
        self.save_interval = args.save_interval
        self.no_op_steps = args.no_op_steps
        self.save_network_path = args.save_network_path
        self.save_summary_path = args.save_summary_path
        self.test_dqn_model_path = args.test_dqn_model_path
        self.exp_name = args.exp_name
        self.ddqn = args.ddqn
        self.dueling = args.dueling
        self.test_path = args.test_path

        if args.optimizer.lower() == 'adam':
            self.opt = Adam(lr=self.learning_rate)
        else:
            self.opt = RMSprop(lr=self.learning_rate, decay=0, rho=0.99, epsilon=self.min_grad)
     
        # environment setting
        self.env = env
        self.num_actions = env.action_space.n
        
        self.epsilon = self.initial_epsilon
        self.epsilon_step = (self.initial_epsilon - self.final_epsilon) / self.exploration_steps
        self.t = 0

        # Input that is not used when fowarding for Q-value 
        # or loss calculation on first output of model 
        self.dummy_input = np.zeros((1,self.num_actions))
        self.dummy_batch = np.zeros((self.batch_size,self.num_actions))

        # for summary & checkpoint
        self.total_reward = 0.0
        self.total_q_max = 0.0
        self.total_loss = 0.0
        self.duration = 0
        self.episode = 0
        self.last_30_reward = deque()
        if not os.path.exists(self.save_network_path):
            os.makedirs(self.save_network_path)
        if not os.path.exists(self.save_summary_path):
            os.makedirs(self.save_summary_path)

        # Create replay memory
        self.replay_memory = deque()

        # Create q network
        self.q_network = self.build_network()

        # Create target network
        self.target_network = self.build_network()

        # load model for testing, train a new one otherwise
        if args.test_dqn:
            self.q_network.load_weights(self.test_dqn_model_path)
        else:
            self.log = open(self.save_summary_path+self.exp_name+'.log','w')

        # Set target_network weight
        self.target_network.set_weights(self.q_network.get_weights())
            

    def init_game_setting(self):
        pass


    def train(self):
        while self.t <= self.num_steps:
            terminal = False
            observation = self.env.reset()
            for _ in range(random.randint(1, self.no_op_steps)):
                last_observation = observation
                observation, _, _, _ = self.env.step(0)  # Do nothing
            while not terminal:
                last_observation = observation
                action = self.make_action(last_observation,test=False)
                observation, reward, terminal, _ = self.env.step(action)
                self.run(last_observation, action, reward, terminal, observation)
        


    def make_action(self, observation, test=True):
        """
        ***Add random action to avoid the testing model stucks under certain situation***
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)

        Return:
            action: int
                the predicted action from trained model
        """
        if not test:
            if self.epsilon >= random.random() or self.t < self.initial_replay_size:
               action = random.randrange(self.num_actions)
            else:
                action = np.argmax(self.q_network.predict([np.expand_dims(observation,axis=0),self.dummy_input])[0])
            # Anneal epsilon linearly over time
            if self.epsilon > self.final_epsilon and self.t >= self.initial_replay_size:
                self.epsilon -= self.epsilon_step
        else:
            if 0.005 >= random.random():
                action = random.randrange(self.num_actions)
            else:
                action = np.argmax(self.q_network.predict([np.expand_dims(observation,axis=0),self.dummy_input])[0])

        return action

    def build_network(self):
        # Consturct model
        input_frame = Input(shape=(self.frame_width, self.frame_height, self.state_length))
        action_one_hot = Input(shape=(self.num_actions,))
        conv1 = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(input_frame)
        conv2 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(conv1)
        conv3 = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(conv2)
        flat_feature = Flatten()(conv3)
        hidden_feature = Dense(512)(flat_feature)
        lrelu_feature = LeakyReLU()(hidden_feature)
        q_value_prediction = Dense(self.num_actions)(lrelu_feature)

        if self.dueling:
            # Dueling Network
            # Q = Value of state + (Value of Action - Mean of all action value)
            hidden_feature_2 = Dense(512,activation='relu')(flat_feature)
            state_value_prediction = Dense(1)(hidden_feature_2)
            q_value_prediction = merge([q_value_prediction, state_value_prediction], mode = lambda x: x[0]-K.mean(x[0])+x[1], 
                                        output_shape = (self.num_actions,))
        

        select_q_value_of_action = Multiply()([q_value_prediction,action_one_hot])
        target_q_value = Lambda(lambda x:K.sum(x, axis=-1, keepdims=True),output_shape=lambda_out_shape)(select_q_value_of_action)
        
        model = Model(inputs=[input_frame,action_one_hot], outputs=[q_value_prediction, target_q_value])
        
        # MSE loss on target_q_value only
        model.compile(loss=['mse','mse'], loss_weights=[0.0,1.0],optimizer=self.opt)

        return model        

    def run(self, state, action, reward, terminal, observation):
        next_state = observation

        # Store transition in replay memory
        self.replay_memory.append((state, action, reward, next_state, terminal))
        if len(self.replay_memory) > self.num_replay_memory:
            self.replay_memory.popleft()

        if self.t >= self.initial_replay_size:
            # Train network
            if self.t % self.train_interval == 0:
                self.train_network()

            # Update target network
            if self.t % self.target_update_interval == 0:
                self.target_network.set_weights(self.q_network.get_weights())

            # Save network
            if self.t % self.save_interval == 0:
                save_path = self.save_network_path + '/' + self.exp_name+'_'+str(self.t)+'.h5'
                self.q_network.save(save_path)
                print('Successfully saved: ' + save_path)

        self.total_reward += reward
        self.total_q_max += np.max(self.q_network.predict([np.expand_dims(state,axis=0),self.dummy_input])[0])
        self.duration += 1

        if terminal:
            # Observe the mean of rewards on last 30 episode
            self.last_30_reward.append(self.total_reward)
            if len(self.last_30_reward)>30:
                self.last_30_reward.popleft()

            # Log message
            if self.t < self.initial_replay_size:
                mode = 'random'
            elif self.initial_replay_size <= self.t < self.initial_replay_size + self.exploration_steps:
                mode = 'explore'
            else:
                mode = 'exploit'
            print('EPISODE: {0:6d} / TIMESTEP: {1:8d} / DURATION: {2:5d} / EPSILON: {3:.5f} / AVG_REWARD: {4:2.3f} / AVG_MAX_Q: {5:2.4f} / AVG_LOSS: {6:.5f} / MODE: {7}'.format(
                self.episode + 1, self.t, self.duration, self.epsilon,
                np.mean(self.last_30_reward), self.total_q_max / float(self.duration),
                self.total_loss / (float(self.duration) / float(self.train_interval)), mode))
            print('EPISODE: {0:6d} / TIMESTEP: {1:8d} / DURATION: {2:5d} / EPSILON: {3:.5f} / AVG_REWARD: {4:2.3f} / AVG_MAX_Q: {5:2.4f} / AVG_LOSS: {6:.5f} / MODE: {7}'.format(
                self.episode + 1, self.t, self.duration, self.epsilon,
                np.mean(self.last_30_reward), self.total_q_max / float(self.duration),
                self.total_loss / (float(self.duration) / float(self.train_interval)), mode),file=self.log)

            # Init for new game
            self.total_reward = 0
            self.total_q_max = 0
            self.total_loss = 0
            self.duration = 0
            self.episode += 1

        self.t += 1

    def train_network(self):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        terminal_batch = []
        y_batch = []


        # Sample random minibatch of transition from replay memory
        minibatch = random.sample(self.replay_memory, self.batch_size)
        for data in minibatch:
            state_batch.append(data[0])
            action_batch.append(data[1])
            reward_batch.append(data[2])
            next_state_batch.append(data[3])
            terminal_batch.append(data[4])

        # Convert True to 1, False to 0
        terminal_batch = np.array(terminal_batch) + 0
        # Q value from target network
        target_q_values_batch = self.target_network.predict([list2np(next_state_batch),self.dummy_batch])[0]

        # create Y batch depends on dqn or ddqn
        if self.ddqn:
            next_action_batch = np.argmax(self.q_network.predict([list2np(next_state_batch),self.dummy_batch])[0], axis=-1)
            for i in range(self.batch_size):
                y_batch.append(reward_batch[i] + (1-terminal_batch[i])*self.gamma*target_q_values_batch[i][next_action_batch[i]] )
            y_batch = list2np(y_batch)
        else:
            y_batch = reward_batch + (1 - terminal_batch) * self.gamma * np.max(target_q_values_batch, axis=-1)
        
        a_one_hot = np.zeros((self.batch_size,self.num_actions))
        for idx,ac in enumerate(action_batch):
            a_one_hot[idx,ac] = 1.0

        loss = self.q_network.train_on_batch([list2np(state_batch),a_one_hot],[self.dummy_batch,y_batch])

        self.total_loss += loss[1]

def list2np(in_list):
    return np.float32(np.array(in_list))

def lambda_out_shape(input_shape):
    shape = list(input_shape)
    shape[-1] = 1
    return tuple(shape)

        


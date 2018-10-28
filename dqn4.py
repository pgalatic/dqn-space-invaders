# author: Paul Galatic
# inspired by https://github.com/keon/deep-q-learning/blob/master/dqn.py
# inspired by https://yanpanlau.github.io/2016/07/10/FlappyBird-Keras.html

import gym
import os
import keras
import argparse
from replay_buffer import ReplayBuffer
import numpy as np

GAMMA = 0.95
EPSILON_MAX = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.9995
LEARNING_RATE = 0.0001
ACTION_SIZE = 6
MAX_SIZE = 2000
BATCH_SIZE = 32
MODEL_NAME = 'dqn4.model'
MODEL_NAME_NO_MEMORY = 'dqn4_NO_MEMORY.model'

class DQN_Agent_4():
    def __init__(self, has_memory=False):
        self.memory = ReplayBuffer(MAX_SIZE if has_memory else 2)
        self.model_name = MODEL_NAME if has_memory else MODEL_NAME_NO_MEMORY
        self.model = self._build_model()
        self.last_observation = np.zeros(shape=(1, 105, 80, 1))
        self.last_action = np.squeeze(np.eye(ACTION_SIZE)[0])
        
        print(self.model_name)
        
        self.epsilon = EPSILON_MAX
        
        if os.path.exists(MODEL_NAME):
            self.load()
    
    def _build_model(self):
        """
        Takes frame by frame input and outputs a one-hot vector that describes
        the action to take.
        """
        frames_in = keras.layers.Input(shape=(105, 80, 1), name='frame_input')
        
        print('INPUT\t\t', frames_in.shape)
        
        # pdb.set_trace()
        
        # convolutions
        # (105, 80, 1) -> (101, 76, 32)
        conv_1 = keras.layers.convolutional.Conv2D(
            filters=32, kernel_size=(5, 5), strides=(1, 1),
            activation='relu', input_shape=(105, 80, 1), batch_size=1
        )(frames_in)
        print('CONV_1\t\t', conv_1.shape)
        # (101, 76, 32) -> (50, 38, 32)
        max_pool_1 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv_1)
        print('MAX_POOL_1\t', max_pool_1.shape)
        
        # (50, 38, 32) -> (49, 37, 64)
        conv_2 = keras.layers.convolutional.Conv2D(
            filters=64, kernel_size=(2, 2), activation='relu'
        )(max_pool_1)
        print('CONV_2\t\t', conv_2.shape)
        # (49, 37, 64) -> (24, 18, 64)
        max_pool_2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv_2)
        print('MAX_POOL_2\t', max_pool_2.shape)
        
        # flat layer
        flat = keras.layers.Flatten()(max_pool_2)
        print('FLAT\t\t', flat.shape)
        
        # hidden layer
        hidden = keras.layers.Dense(128, activation='relu')(flat)
        print('HIDDEN\t\t', hidden.shape)
        
        # output layer
        output = keras.layers.Dense(ACTION_SIZE, activation='softmax')(hidden)
        print('OUTPUT\t\t', output.shape)
        
        model = keras.models.Model(input=frames_in, output=output)
        optim = keras.optimizers.RMSprop(
            lr=LEARNING_RATE, rho=GAMMA, epsilon=EPSILON_MIN)
        model.compile(optim, loss='mse')
        
        return model
    
    def save(self):
        self.model.save_weights(self.model_name)
        print('SAVED MODEL')
    
    def load(self):
        self.model.load_weights(self.model_name)
        print('LOADED MODEL')
    
    def normalize(self, img):
        return img * (255/img.max())
    
    def shape(self, img):
        return np.reshape(img, (1, 105, 80, 1))
    
    def grayscale(self, img):
        return np.mean(img, axis=2).astype(np.uint8)
    
    def downsample(self, img):
        return img[::2, ::2]
    
    def preprocess(self, img):
        return self.shape(self.normalize(self.grayscale(self.downsample(img))))
    
    def rewind(self):
        """
        Applies Q-learning step based on randomly sampled memory episodes.
        Q learning, as it is implemented here, can be described as follows:
        
        1) Sample a batch of episodes (memories).
        2) For each episode:
            a) let the input of the episode be the previous state
            b) predict the best action to take, given that previous state
            c) predict the best action to take, given the state we ended up in
            d) predict the maximum future reward we could get to based on the 
                state we ended up in
            e) fit the model so that it always chooses the action with the 
                greatest possible future reward, based on its memory
        """
        minibatch = self.memory.sample(BATCH_SIZE)
        
        inputs  = np.zeros((BATCH_SIZE, 105, 80, 1))
        targets = np.zeros((BATCH_SIZE, ACTION_SIZE))
        
        for i in range(len(minibatch)):
            last_obs    = minibatch[i][0]
            last_act    = minibatch[i][1]
            reward      = minibatch[i][2]
            obs         = minibatch[i][3]
            done        = minibatch[i][4]
            
            print(last_act)
            
            inputs[i:i+1] = last_obs
            targets[i] = self.model.predict(last_obs)
            Q_prediction = self.model.predict(obs)
            if done:
                targets[i, int(np.argmax(last_act[0]))] = reward
            else:
                targets[i, int(np.argmax(last_act[0]))] = reward + GAMMA * np.max(Q_prediction)

        self.model.fit(inputs, targets, epochs=1, verbose=0)
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY
    
    # You should modify this function
    def act(self, observation, reward, done, training=False):
        """
        Queries the agent for an action. The action can be random, based on an 
        Epsilon factor that decreases over the course of a training session. 
        For testing, only the model itself is queried.
        """
        observation = self.preprocess(observation)
        if done:
            reward = -1
    
        # don't choose random actions during test
        if (np.random.rand() <= self.epsilon) and training:
            # choose random row from identity matrix (random one-hot vector)
            action = np.eye(ACTION_SIZE)[np.random.choice(ACTION_SIZE, 1)]
        else:
            action = self.model.predict(observation)
        
        self.last_action = action
        
        # sample memory and train
        if training:
            self.memory.add(self.last_observation, self.last_action, reward, observation, done)
            self.rewind()
        
        action = np.argmax(action[0])
        return action
    
def loop(env, agent, training):
    """
    Basic training loop; given an environment, calls Agent.act() until the 
    game is finished.
    """
    reward = 0
    done = False
    score = 0
    special_data = {}
    special_data['ale.lives'] = 3
    ob = env.reset()
    while not done:
        
        action = agent.act(ob, reward, done, training=training)
        ob, reward, done, _ = env.step(action)
        score += reward
        # env.render()
     
    # Close the env and write monitor result info to disk
    # print ("Your score: %d" % score)
    return score
    
def train():
    """
    Sets up environment and writes scores to file.
    """
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('training', nargs='?', default=False, type=bool, help='Set to \'True\' to train')
    args = parser.parse_args()

    env = gym.make('SpaceInvaders-v0')
    env.seed()
    agent = DQN_Agent_4()
    total_score = 0
    
    for i in range(10):
        for i in range(10):
            score = loop(env, agent, args.training)
            total_score += score
            with open('dqn4_results.csv', 'a') as f:
                f.write(str(int(score)))
                f.write(', ')
        print('Average score: %d' % (total_score / 10))
        total_score = 0
        if args.training:
            agent.save()

    
    env.close()
    
if __name__ == '__main__':
    train()
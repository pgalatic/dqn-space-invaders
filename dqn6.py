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
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.9995
LEARNING_RATE = 0.0001
ACTION_SIZE = 6
MAX_SIZE = 2000
BATCH_SIZE = 32
MODEL_NAME = 'dqn6.model'

class DQN_Agent_6():
    def __init__(self):
        self.memory = ReplayBuffer(MAX_SIZE)
        self.model = self._build_model()
        self.last_observation = np.zeros(shape=(1, 105, 80, 1))
        self.last_action = np.eye(ACTION_SIZE)[0]
        
        self.epsilon = EPSILON_MAX
        
        if os.path.exists(MODEL_NAME):
            self.load()
        else:
            print('NEW MODEL')
    
    def _build_model(self):
        """
        Takes frame by frame input and outputs a one-hot vector that describes
        the action to take.
        """
        frames_in = keras.layers.Input(shape=(105, 80, 1), name='frame_input')
        # actions_in = keras.layers.Input(shape=(1, 6, 1), name='action_input')
        
        print('INPUT\t\t', frames_in.shape)
        
        # pdb.set_trace()
        
        # convolutions
        # (105, 80, 1) -> (52, 40, 32)
        conv_1 = keras.layers.convolutional.Conv2D(
            filters=32, kernel_size=(4, 4), strides=(2, 2),
            activation='relu', input_shape=(105, 80, 1), batch_size=1
        )(frames_in)
        print('CONV_1\t\t', conv_1.shape)
        
        # (52, 40, 32) -> (26, 20, 32)
        max_pool_1 = keras.layers.MaxPooling2D(
                       pool_size=(2, 2), strides=(2, 2))(conv_1)
        print('MAX_POOL_1\t', max_pool_1.shape)
        
        # (26, 20, 32) -> (13, 10, 64)
        conv_2 = keras.layers.convolutional.Conv2D(
            filters=64, kernel_size=(4, 4), strides=(2, 2), 
            activation='relu'
        )(max_pool_1)
        print('CONV_2\t\t', conv_2.shape)
        
        # flat layer
        flat = keras.layers.Flatten()(conv_2)
        print('FLAT\t\t', flat.shape)
        
        # hidden layer
        hidden_1 = keras.layers.Dense(128, activation='relu')(flat)
        hidden_2 = keras.layers.Dense(ACTION_SIZE, activation='softmax')(hidden_1)
        print('HIDDEN\t\t', hidden_2.shape)
        
        # output layer
        # output = keras.layers.multiply([hidden_2, actions_in])
        # print('OUTPUT\t\t', output.shape)
        
        model = keras.models.Model(input=frames_in, output=hidden_2)
        optim = keras.optimizers.RMSprop(
            lr=LEARNING_RATE, rho=GAMMA, epsilon=EPSILON_MIN)
        model.compile(optim, loss='mse')
        
        return model
    
    def save(self, name=MODEL_NAME):
        self.model.save_weights(name)
        print('SAVED MODEL')
    
    def load(self, name=MODEL_NAME):
        self.model.load_weights(name)
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
        
        for i in range(len(minibatch)):
            last_obs    = minibatch[i][0]
            last_act    = minibatch[i][1]
            reward      = minibatch[i][2]
            obs         = minibatch[i][3]
            done        = minibatch[i][4]
            
            Q_prediction = self.model.predict(obs)
            if done:
                target = reward
            else:
                target = reward + GAMMA * np.amax(Q_prediction)
            Q_prediction[0, np.argmax(last_act)] = target
            self.model.fit(last_obs, Q_prediction, epochs=1, verbose=0)
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
            action = self.model.predict(observation)[0]
        
        self.last_action = action
        
        # sample memory and train
        if training:
            self.memory.add(self.last_observation, self.last_action, reward, observation, done)
            self.rewind()
        
        return np.argmax(action)

def loop(env, agent, training=False):
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
    print ("Your score: %d" % score)
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
    agent = DQN_Agent_6()
    total_score = 0
    
    if args.training:
        print('TRAINING')
    else:
        print('TESTING')
    
    for i in range(10):
        for i in range(10):
            score = loop(env, agent, args.training)
            total_score += score
            with open('dqn6_results.csv', 'a') as f:
                f.write(str(int(score)))
                f.write(', ')
        print('Average score: %d' % (total_score / 10))
        total_score = 0
        if args.training:
            agent.save()

    
    env.close()

if __name__ == '__main__':
    train()
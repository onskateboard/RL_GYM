# -*- coding: utf-8 -*-
### https://keon.github.io/deep-q-learning/
import tensorflow as tf
import random
import gym
import gym_maze
from copy import deepcopy
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import math
EPISODES = 2000

##more and more and even more

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        '''' build neural network with two layers for choosing the action'''
        model = Sequential()
        model.add(Dense(25, input_dim=self.state_size, activation='relu'))
        model.add(Dense(25, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def memorize(self, state, action, reward, next_state, done):
        '''the replay buffer to use random batches for training'''
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        '''choose eighter random action based on the explore rate or use the model to predict'''
        if np.random.rand() <= explore_rate:
            return env.action_space.sample()
        act_values = self.model.predict(state)
        return int(np.argmax(act_values[0]))  # returns action

    def train(self):
        ''' train the model
         use the model_freeze to make the target values and then overwrite the weights from the model
         the replay buffer is 256 entries, we choose a sample with 32 entries for 10 to train the network
         '''

        for i in range(10):

            minibatch = random.sample(self.memory, 32)
            model_freeze = deepcopy(self.model)
            state_list = []
            target_list = []

            for state, action, reward, next_state, done in minibatch:
                #only one network
                target_y = reward
                if not done:
                    # berechne den Zielwert aus dem reward
                    target_y = (reward + self.gamma *
                              np.amax(model_freeze.predict(next_state)[0]))
                target = self.model.predict(state) #predict all the q values for the actions
                target[0][action] = target_y # fill in the target_y value on the right position (right action)
                # based on the model I calculte the distance to the target_y from the freeze network
                target_list.append(list(target))
                state_list.append(list(state))

            self.model.fit(state_list, target_list, epochs=1, verbose=0)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

def state_to_bucket(state):
    bucket_indice = []
    for i in range(len(state)):
        if state[i] <= STATE_BOUNDS[i][0]:
            bucket_index = 0
        elif state[i] >= STATE_BOUNDS[i][1]:
            bucket_index = NUM_BUCKETS[i] - 1
        else:
            # Mapping the state bounds to the bucket array
            bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
            offset = (NUM_BUCKETS[i]-1)*STATE_BOUNDS[i][0]/bound_width
            scaling = (NUM_BUCKETS[i]-1)/bound_width
            bucket_index = int(round(scaling*state[i] - offset))
        bucket_indice.append(bucket_index)
    return tuple(bucket_indice)

def get_explore_rate(t):
    return max(MIN_EXPLORE_RATE, min(0.8, 1.0 - math.log10((t+1)/(10*DECAY_FACTOR))))

if __name__ == "__main__":
    env = gym.make('maze-sample-5x5-v0')
    NUM_BUCKETS = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
    MAZE_SIZE = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
    STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))
    MIN_EXPLORE_RATE = 0.001
    MIN_LEARNING_RATE = 0.2
    DECAY_FACTOR = np.prod(MAZE_SIZE, dtype=float) / 10.0
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    #agent.load("./cartpole-dqn.h5")
    done = False
    batch_size = 32
    explore_rate = get_explore_rate(0)
    #      PLAY IT
    #while True:
    #     state = env.reset()
    #     env.render()
    #     state = np.reshape(state, [1, state_size])
    #     action = agent.act(state)
    #     env.step(action)

    for e in range(EPISODES):
        rewardskum = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(250):
            env.render()
            action = agent.act(state)
            try:
                next_state, reward, done, _ = env.step(action)
                rewardskum +=reward
            except(ValueError):
                pass
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.memorize(state, action, reward, next_state, done)
            state = next_state
            if done:
                # print("episode: {}/{}, score: {}, e: {:.2}"
                #       .format(e, EPISODES, time, agent.epsilon))
                explore_rate = get_explore_rate(e)
                break
            if len(agent.memory) > 512:
                agent.train()
        explore_rate = get_explore_rate(e)
        print(f"Explore rate: {explore_rate}")
        print(f"Epsiode: {e}")
        print(f"Rewards: {rewardskum}")
        if e % 10 == 0:

            agent.save("./cartpole-dqn1.h5")
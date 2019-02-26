from keras.layers.core import Flatten
from keras.layers import Dense, Conv2D, MaxPooling2D
from keras.models import Sequential, load_model
from keras.optimizers import Adam
import numpy as np
import random
from collections import deque

import ai_util as util

class Agent:
    
    def __init__(self, state_shape, n_actions, exploit=False):
        self.memory = deque(maxlen=1000)
        self.n_actions = n_actions
        self.state_shape = state_shape
        
        self.gamma = 0.99
        
        if exploit:
            self.epsilon = 0.1
        else:
            self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.0001

    def new_model(self):
        model = Sequential()

        model.add(Conv2D(32,
            kernel_size=(8, 4),
            strides=(2, 2),
            activation="relu",
            input_shape=self.state_shape))

        model.add(Conv2D(64,
            kernel_size=(4, 4),
            strides=(2, 2),
            activation="relu"))
        
        model.add(Conv2D(64,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation="relu"))

        model.add(Flatten())
        model.add(Dense(512, activation="relu"))
        model.add(Dense(self.n_actions, activation="linear"))

        model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer=Adam(lr=self.learning_rate))

        self.model = model
        
    def save_model(self, name):
        try:
            self.model.save(name)
        except KeyboardInterrupt:
            self.model.save(name) 
            raise
        
    def load_model(self, name):
        self.model = load_model(name) 
        
    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.n_actions)
        return self._predict(state)
    
    def remember(self, state, action, reward, next_state, done):
        new_row = np.array([state, action, reward, next_state, done])
        self.memory.append(new_row)
        
    def replay(self, batch_size):
        if len(self.memory) > batch_size:
            batch = random.sample(self.memory, batch_size)
            batch = np.array(batch)
            self._fit(batch, self.gamma, self.n_actions)
    
    def _fit(self, batch, gamma, n_outputs):
        # Data "wrangling"
        states, actions, rewards, next_states, done = np.hsplit(batch, batch.shape[1])
        actions = util.one_hot_encode(n_outputs, actions[:, 0]) > 0 
        rewards = rewards[:, 0]
        done = done[:, 0]
        states = util.stack(states)
        next_states = util.stack(next_states)
        
        # Predict future
        predicted_future_Q_values = self.model.predict(next_states)
        predicted_future_rewards = np.amax(predicted_future_Q_values, axis=1)
        
        # Calculate expected q values
        not_done_target = np.logical_not(done) * (rewards + self.gamma * predicted_future_rewards)
        done_targets = done * rewards
        targets = not_done_target + done_targets
        
        # Set expected q values for the actions in question
        target_Q_values = self.model.predict(states)
        target_Q_values[actions] = targets
        
        self.model.fit(states, target_Q_values, epochs=1, verbose=0)
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def _predict(self, state):
        input_state = np.stack([state])
        Q_values = self.model.predict(input_state)[0]
        return Q_values.argmax(axis=0)
    
     
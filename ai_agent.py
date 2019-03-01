from keras.layers.core import Flatten
from keras.layers import Dense, Conv2D, MaxPooling2D
from keras.models import Sequential, load_model
from keras.optimizers import Adam
import numpy as np
import random
from collections import deque

import ai_util as util

class Agent:
    
    def __init__(self, state_shape, n_actions, epsilon=0.1):
        self.memory = deque(maxlen=10000)
        self.n_actions = n_actions
        self.state_shape = state_shape
        self.gamma = 0.99
        self.epsilon = epsilon
        self.learning_rate = 0.0001

    def new_model(self):
        model = Sequential()

        model.add(Conv2D(16,
            kernel_size=(4, 4),
            strides=(2, 2),
            activation="relu",
            input_shape=self.state_shape))
        
        model.add(Conv2D(32,
            kernel_size=(4, 4),
            strides=(2, 2),
            activation="relu",
            input_shape=self.state_shape))

        model.add(Flatten())
        model.add(Dense(256, activation="relu"))
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
        state = np.moveaxis(state, 0, -1)
        next_state = np.moveaxis(next_state, 0, -1)
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
            
    def _predict(self, state):
        state = np.moveaxis(state, 0, -1)
        input_state = np.stack([state])
        Q_values = self.model.predict(input_state)[0]
        return Q_values.argmax(axis=0)
    
     
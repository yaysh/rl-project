from keras.layers.core import Flatten
from keras.layers import Dense, Conv2D, MaxPooling2D
from keras.models import Sequential, load_model
import numpy as np
import random
from collections import deque

import ai_util as util

class Agent:
    
    def __init__(self, state_shape, n_actions):
        self.memory = deque(maxlen=1000)
        self.n_actions = n_actions
        self.state_shape = state_shape
        
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

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
            activation="relu"))

        model.add(Flatten())
        model.add(Dense(512, activation="relu"))
        model.add(Dense(self.n_actions, activation="linear"))

        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        self.model = model
        
    def save_model(self, name):
        self.model.save(name) 
        
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
        """
        states, actions, rewards, next_states, done = np.hsplit(batch, batch.shape[1])

        states = util.stack(states)
        next_states = util.stack(next_states)
        actions = util.one_hot_encode(n_outputs, actions) 

        next_Q_values = self.model.predict(next_states)
        Q_values = rewards + gamma * np.max(next_Q_values, axis=0)

        self.model.fit(
            states, 
            actions * Q_values,
            epochs=1, 
            batch_size=len(states), 
            verbose=0
        )
        """
        for state, action, reward, next_state, done in batch:
            state = np.stack([state])
            next_state = np.stack([next_state])
            
            target = reward
            
            if not done:
                predicted_future_Q_values = self.model.predict(state)[0]
                predicted_future_reward = np.amax(predicted_future_Q_values)
                target = reward + self.gamma * predicted_future_reward
            
            target_Q_values = self.model.predict(state)
            target_Q_values[0][action] = target
            
            self.model.fit(state, target_Q_values, epochs=1, verbose=0)
            
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
    
    def _predict(self, state):
        input_state = np.stack([state])
        Q_values = self.model.predict(input_state)[0]
        return Q_values.argmax(axis=0)
    
     
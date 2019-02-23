from keras.layers.core import Flatten
from keras.layers import Dense, Conv2D, MaxPooling2D
from keras.models import Sequential, load_model
import numpy as np

import ai_util as util

class Agent:

    def build_model(self, state_shape, n_actions):
        model = Sequential()

        model.add(Conv2D(16,
            kernel_size=(4, 4),
            strides=(2, 2),
            activation="relu",
            input_shape=state_shape))

        model.add(Conv2D(32,
            kernel_size=(4, 4),
            strides=(2, 2),
            activation="relu"))

        model.add(Flatten())
        model.add(Dense(512, activation="relu"))
        model.add(Dense(n_actions))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        self.model = model
        
    def save_model(self, name):
        self.model.save(name) 
        
    def load_model(self, name):
        self.model = load_model(name) 
    
    def fit(self, batch, gamma, n_outputs):
        states, actions, rewards, next_states, done = np.hsplit(batch, batch.shape[1])

        states = util.stack(states)
        next_states = util.stack(next_states)
        actions = util.one_hot_encode(n_outputs, actions) # TODO: Needs one hot encoding

        next_Q_values = self.model.predict(next_states)
        Q_values = rewards + gamma * np.max(next_Q_values, axis=0)

        self.model.fit(
            states, 
            actions * Q_values,
            epochs=1, 
            batch_size=len(states), 
            verbose=0
        )
    
    def predict(self, state):
        input_state = np.stack([state])
        Q_values = self.model.predict(input_state)[0]
        return Q_values.argmax(axis=0)
    
     
import random
import numpy as np

class Memory:

    def __init__(self, size):
        self.size = size
        self.memory = np.zeros(shape=(0, 5))

    def add(self, state, action, reward, next_state, done):
        new_row = np.array([[state, action, reward, next_state, done]])
        self.memory = np.append(self.memory, new_row, axis=0)
        if self.size < self.memory.shape[0]:
            self.memory = np.delete(self.memory, 0, axis=0)

    def sample(self, batch_size):
        return self.memory[np.random.choice(self.memory.shape[0], batch_size, replace=False)]
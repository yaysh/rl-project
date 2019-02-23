import numpy as np

def stack(array):
    stacked_array = np.stack(array[0])
    for i in range(1, array.shape[0]):
        stacked_array = np.append(stacked_array, np.stack(array[i]), axis=0)
    return stacked_array

def one_hot_encode(size, values):
    vector = np.array(values, dtype=int).reshape(-1)
    return np.eye(size, dtype=int)[vector]

def update_states(states, state):
    states = np.delete(states, 0, axis=2)
    states = np.concatenate((states, state[..., np.newaxis]), axis=2)
    return states

def create_states_arr(state):
    states = state[..., np.newaxis]
    for i in range(2):
        states = np.append(states, states, axis=2)
    return np.stack(states)
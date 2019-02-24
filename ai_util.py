import numpy as np

def stack(array):
    stacked_array = np.stack(array[0])
    for i in range(1, array.shape[0]):
        stacked_array = np.append(stacked_array, np.stack(array[i]), axis=0)
    return stacked_array

def one_hot_encode(size, values):
    vector = np.array(values, dtype=int).reshape(-1)
    return np.eye(size, dtype=int)[vector]

def update_state_arr(state, frame):
    state = np.delete(state, 0, axis=2)
    state = np.concatenate((state, frame[..., np.newaxis]), axis=2)
    return state

def create_state_arr(frame):
    state = frame[..., np.newaxis]
    for i in range(2):
        state = np.append(state, state, axis=2)
    return np.stack(state)
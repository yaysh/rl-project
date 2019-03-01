import numpy as np
from collections import deque

def stack(array):
    stacked_array = np.stack(array[0])
    for i in range(1, array.shape[0]):
        stacked_array = np.append(stacked_array, np.stack(array[i]), axis=0)
    return stacked_array

def one_hot_encode(size, values):
    vector = np.array(values, dtype=int).reshape(-1)
    return np.eye(size, dtype=int)[vector]

def update_state_arr(state, frame_1, frame_2):
    next_state = state.copy()
    next_state.append(frame_1)
    next_state.append(frame_2)
    return next_state

def create_state_arr(frame):
    state = deque(maxlen=4)
    for _ in range(4):
        state.append(frame)
    return state
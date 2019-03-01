import numpy as np
from collections import deque

def one_hot_encode(size, values):
    vector = np.array(values, dtype=np.uint8).reshape(-1)
    return np.eye(size, dtype=np.bool)[vector]

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
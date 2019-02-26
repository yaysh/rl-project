from IPython import display 
from collections import deque

log_queue = deque(maxlen=10)
def log(text):
    log_queue.append(text)
    display.clear_output(wait=True)
    print("\n".join(log_queue), flush=True)

from IPython import display 
from collections import deque
#import sys

log_queue = deque(maxlen=10)
def log(text):
    log_queue.append(text)
    display.clear_output(wait=True)
    for row in log_queue:
        print(row, flush=True)
    #sys.stdout.flush()
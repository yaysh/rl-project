import matplotlib.pyplot as plt
from IPython import display

def show_state(observation, env_id, step=0, info=""):
    plt.figure(3)
    plt.clf()
    plt.imshow(observation, cmap='gray')
    plt.title("%s | Step: %d %s" % (env_id, step, info))
    plt.axis('off')

    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.close()
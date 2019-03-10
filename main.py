import tensorflow as tf  
config = tf.ConfigProto()  
config.gpu_options.allow_growth = True  
session = tf.Session(config=config)  

from ai_image_preprocess import preprocess
import ai_state as state_util
import numpy as np

def step(env, action, state):
    next_frame_1, reward_1, done_1, _ = env.step(action)
    next_frame_2, reward_2, done_2, _ = env.step(action)
    next_state = state_util.update(state, preprocess(next_frame_1), preprocess(next_frame_2))
    return (next_state, int(reward_1 + reward_2), done_1 or done_2)

# import ai_display as display
import ai_state as state_util
from ai_logger import Logger
import time
from file_handler import FileHandler
file_handler = FileHandler()
print(file_handler.steps)


import matplotlib.pyplot as plt


def train(env, agent, n_episodes=100000, model_name="model.h5", save_interval=25):
    logger = Logger(10, "episode | states | score | step time | epi time | epsilon")
    backup_save_interval = 28
    import os 
    if os.path.isfile(model_name):
        agent.load_model(model_name)
        steps = file_handler.steps
        agent.epsilon = file_handler.epsilon
        agent.epsilon_decay = file_handler.decay_rate
        rewards = file_handler.rewards
        epsilons = file_handler.epsilon
    else:
        agent.new_model()
        epsilon = agent.epsilon
        epsilon_decay = agent.epsilon_decay
        steps = 0
        rewards = []
        epsilons = []
        file_handler.write_to_file(epsilon_decay, steps, epsilon, rewards, epsilons)

    for episode in range(n_episodes):
        
        frame = env.reset()
        state = state_util.create(preprocess(frame))
        score = 0
        
        start_time = time.time()
        done = False
        t = 0
        
        while not done:
            # display.show_state(state, env.spec.id, t, score)
            #env.render()
            t+=1
            action = agent.act(state)
            next_state, reward, done = step(env, action, state)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            score += reward
            # if done: 
            #     break
        steps += 1
        rewards.append(score)
        epsilons.append(agent.epsilon)
        agent.replay(batch_size=128, score=score, epsilon=agent.epsilon)   
        
        if (episode+1) % 8000 == 0:
            plt.plot(rewards)
            plt.show()
        duration = time.time() - start_time
        # logger.log("{:>7d} | {:>6d} | {:>5d} | {:>9.5f} | {:>8.5f} | {:>7.5f}"
        #        .format(episode+1, t, score, duration/t, duration, agent.epsilon))
        print("{:>7d} | {:>6d} | {:>5d} | {:>9.5f} | {:>8.5f} | {:>7.5f}"
               .format(episode+1, t, score, duration/t, duration, agent.epsilon))
        print(np.min(agent.q), np.max(agent.q))
                
        if episode % save_interval == 0:
            # Save number of steps and epsilon and epsilon decay rate to text file
            print("Saving model, please don't interrupt")
            epsilon = agent.epsilon
            epsilon_decay = agent.epsilon_decay
            file_handler.write_to_file(epsilon_decay, steps, epsilon, rewards, epsilons)
            agent.save_model(model_name)
            print("Model has been saved")
        elif episode % backup_save_interval == 0:
            print("Saving backup, please don't interrupt")
            epsilon = agent.epsilon
            epsilon_decay = agent.epsilon_decay
            file_handler.write_to_file(epsilon_decay, steps, epsilon, rewards, epsilons)
            agent.save_model('with-graph-backup.h5')
            print("Model has been saved")
            
    print("Saving model, please don't interrupt")
    agent.save_model(model_name)
    epsilon = agent.epsilon
    epsilon_decay = agent.epsilon_decay
    file_handler.write_to_file(epsilon_decay, steps, epsilon, epsilons)
    print("Model has been saved")
        

def calc_dimensions(env):
    n_actions = env.action_space.n
    obs_shape = env.observation_space.shape
    height = obs_shape[0]//2; width = obs_shape[1]//2; n_frames = 4
    state_shape = (height, width, n_frames)
    return (state_shape, n_actions)


from ai_agent import Agent
import gym

env = gym.make("BreakoutDeterministic-v4")
state_shape, n_actions = calc_dimensions(env)

agent = Agent(state_shape, n_actions)
model_name = "with-graph.h5"
train(env, agent, model_name=model_name)

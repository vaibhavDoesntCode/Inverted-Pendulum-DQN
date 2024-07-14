
import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
import random

# Created the environment
env = gym.make('CartPole-v1')

# Function to create the model
def create_model(state_shape, num_actions):
    model = models.Sequential()
    model.add(layers.Dense(24, activation='relu', input_shape=state_shape))
    model.add(layers.Dense(24, activation='relu'))
    model.add(layers.Dense(num_actions, activation='linear'))
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mse')
    return model

# Define hyperparameters
num_actions = env.action_space.n
state_shape = env.observation_space.shape
model = create_model(state_shape, num_actions)
target_model = create_model(state_shape, num_actions)
target_model.set_weights(model.get_weights())

gamma = 0.99  
epsilon = 1.0  
epsilon_min = 0.1
epsilon_decay = 0.995
learning_rate = 0.001
batch_size = 32
memory = []
max_memory_length = 2000
target_update_freq = 50

# Function to remember experiences
def remember(state, action, reward, next_state, done):
    if len(memory) >= max_memory_length:
        memory.pop(0)
    memory.append((state, action, reward, next_state, done))

# Function to perform experience replay
def replay():
    if len(memory) < batch_size:
        return
    
    minibatch = random.sample(memory, batch_size)
    
    for state, action, reward, next_state, done in minibatch:
        target = reward
        if not done:
            next_state = np.reshape(next_state, [1, state_shape[0]]) 
            target += gamma * np.amax(target_model.predict(next_state)[0])
        
        state = np.reshape(state, [1, state_shape[0]]) 
        target_f = model.predict(state)
        target_f[0][action] = target
        model.fit(state, target_f, epochs=1, verbose=0)

# Training loop
for episode in range(1000):
    state = env.reset()
    
    state = np.reshape(state[0], [1, state_shape[0]])
    total_reward = 0
    for time in range(500):
        env.render()
        if np.random.rand() <= epsilon:
            action = np.random.choice(num_actions)
        else:
            action = np.argmax(model.predict(state)[0])
        
        next_state, reward, done, _, info = env.step(action)
        next_state = np.reshape(next_state, [1, state_shape[0]])  # Ensure shape of next_state is correct
        
        remember(state, action, reward, next_state, done)
        
        state = next_state
        total_reward += reward
        
        if done:
            print(f"Episode: {episode}/{1000}, Score: {total_reward}, Epsilon: {epsilon:.2}")
            break
        
        replay()
    
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
    
    if episode % target_update_freq == 0:
        target_model.set_weights(model.get_weights())

# Evaluate the model
state = env.reset()

state = np.reshape(state[0], [1, state_shape[0]])
total_reward = 0
while True:
    env.render()
    action = np.argmax(model.predict(state)[0])
    next_state, reward, done, _ = env.step(action)
    next_state = np.reshape(next_state, [1, state_shape[0]])
    state = next_state
    total_reward += reward
    if done:
        print(f"Total reward: {total_reward}")
        break

# Close the environment
env.close()
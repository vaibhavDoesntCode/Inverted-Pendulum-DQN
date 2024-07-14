# Finsearch Project
## Stock trading and optimization using RL and Deep Learning
## Assignment 1- Inverted Pendulum Problem solved using DQN
This project implements a Deep Q-Learning algorithm to train an agent to balance a pole on a cart using [OpenAI's Gym environment CartPole-v1.](https://www.gymlibrary.dev/index.html)
### Setup
1. Environment Setup: Make sure you have Python 3.x installed along with necessary packages:
~~~sh
python install -r requirements.txt
~~~
OR
~~~pip
pip install gym tensorflow
~~~
2. **Running the Code:** Execute the main script cartpole_dqn.py to start training the agent:
~~~python
python inverted_pendulum.py
~~~
### Files
1. inverted_pendulum.py: The main script containing the Deep Q-Learning algorithm implementation
2. requirements.txt: A text file that contains the requirements for the script to run
3. README.md: This file, provides an overview of the project and instructions.
### Implementation
#### Model Architecture
The neural network model used for Q-learning:

**Input layer:** Dense layer with ReLU activation.
**Hidden layers:** Two Dense layers with 24 units each and ReLU activation.
**Output layer:** Dense layer with linear activation corresponding to the number of actions.
#### Hyperparameters
gamma: Discount factor for future rewards.
epsilon: Exploration-exploitation trade-off parameter.
learning_rate: Learning rate for the Adam optimizer.
batch_size: Batch size for experience replay.
max_memory_length: Maximum length of the memory replay buffer.
target_update_freq: Frequency of updating the target network weights.
#### Training
The agent is trained using experience replay and periodically updates the target network to stabilize learning.

#### Evaluation
After training, the trained model is evaluated by running it on the environment without exploration


PS: We all worked together on a Google Colab, which is why only one of us is pushing it to GitHub on the last day.

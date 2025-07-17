# DRL experiment
This set of experiments explores the fundamental concepts in RL and DRL, as well as their applications in various simulated environments. Throughout the making this repo, I learned the theory of RL and DRL from the [RL Course by David Silver](https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ&index=1) and Sutton & Barto's RL book ([Reinforcement Learning: An Introduction, 2018](https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf))

## Structure
### gym_exercise folder
This folder contains experiments with RL and DRL experiments in various gymnasium environments, such as FrozenLake-v1, CartPole-v1, InvertedPendulum-v5,...

#### Frozen Lake 
This environment serves as a learning and testing ground for both traditional RL and DRL algorithms. The algorithms that I tested in this environment include both value-based and policy-based approaches
- Monte Carlo 
- Temporal Difference methods, currently only using TD(0) with both SARSA and Q-Learning update.
- In the future, I plan to extend this to TD(n) and TD($\lambda$) to explore the bias-variance trade-off between the two extremes of MC and TD(0)
- REINFORCE (only when is_slipper=False)

#### CartPole
This environment was chosen as the next step in the DRL learning roadmap. Compared to `FrozenLake`, this environment has a continuous observation space (states are bounded real values) and a discrete action space. Therefore, I used this environment to reinforce my knowledge in value function approximation using neural network approximations. Each algorithm tested (listed below) was trained using one starting hyperparameter set. They are then trained in a grid search fashion to investigate the effect of each hyperparameter on the training performance. The algorithms that I trained in the `CartPole-v1` environment are:
- `DQN` (Deep Q Network), where I learned how to store past transitions in a replay buffer, from which data batches are randomly sampled to train the value function NN
- `DDQN` (Double DQN), which is an extension of DQN that reduces the overestimation during Q-learning bootstrap step. This algorithm uses a frozen (less-frequently updated) value approximator to approximate the value of an optimal action selected by the current (frequently-updated) policy.
- `REINFORCE`
- `A2C` (Advantaged Actor Critic)

#### InvertedPendulum
This environment is a step above the CartPole environment, where both the observation and action spaces are continuous. 
- `PPO` (Proximal Policy Optimization)

### RL folder
(currently on pause to focus on gym_exercise - May 2025)
This folder contains customized code with rudimentary implementation of classical environments, such as frozen lake for learning Monte Carlo method

## Requirements
Setup a virtual environment (venv) and install the following dependencies in the venv
'pip install gymnasium'

Installing pytorch (with or without CUDA) - [PyTorch documentation](https://pytorch.org/get-started/locally/)

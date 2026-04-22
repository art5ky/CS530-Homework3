"""
Rutgers University, Spring 2026
CS530, Homework 3 - DQN Agent and Discretized VI Agent
Written by: Arthur Levitsky
"""

import argparse
import gymnasium as gym
import random
import numpy as np
import torch 
import torch.nn as nn
from collections import deque

# Shared tunable hyperparameters for both DQN and VI agent.
NUM_ACTIONS = 5
GAMMA = 0.99

# Tunable hyperparameters specific to DQN agent.
SYNC_TARGET_DQN_RATE = 10 # Number of episodes until syncing target network with policy network. 
REPLAY_BUFFER_SIZE = 100000
BATCH_SIZE = 64
EPSILON = 1
EPSILON_DECAY = 0.95 # How decay rate for epsilon per episode.
EPSILON_MIN = 0.01 # Lower bound for epsilon.

# Tunable hyperparameters specific to VI agent.
NUM_POS = 100
NUM_VEL = 100
THETA = 1e-4

# The DQN network that takes in the observations and outputs Q-values for the discrete actions. 
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    def forward(self, x):
        return self.net(x)

def modifyReward(next_state):
    """
    For calculating DQN reward, take the absolute value of velocity.
    This incentivizes the agent to maintain the highest velocity possible at each step.
    Implicitly, it will force the agent out of the valley.
    """
    _, velocity = next_state
    return abs(velocity)

def discretizeState(state, env):
    """
    Discretize the state. Map the continuous state to a (pos_idx, vel_idx) grid index.
    """
    pos_low, vel_low = env.observation_space.low
    pos_high, vel_high = env.observation_space.high
    position, velocity = state

    pos_idx = int(np.digitize(position, np.linspace(pos_low, pos_high, NUM_POS)))
    vel_idx = int(np.digitize(velocity, np.linspace(vel_low, vel_high, NUM_VEL)))

    pos_idx = np.clip(pos_idx, 0, NUM_POS - 1)
    vel_idx = np.clip(vel_idx, 0, NUM_VEL - 1)

    return (pos_idx, vel_idx)

def createTransitionTable(env):
        """
        State transition table created using the environment. This is used for Value Iteration.
        """

        pos_low, vel_low = env.observation_space.low
        pos_high, vel_high = env.observation_space.high
        discrete_actions = np.linspace(env.action_space.low, env.action_space.high, NUM_ACTIONS)
        discrete_pos = np.linspace(pos_low,  pos_high,  NUM_POS)
        discrete_vel = np.linspace(vel_low,  vel_high,  NUM_VEL)

        transition_table = {}

        print("Building state transition table...")
        for i, position in enumerate(discrete_pos):
            for j, velocity in enumerate(discrete_vel):
                for k, action in enumerate(discrete_actions):

                    env.reset()
                    env.unwrapped.state = np.array([position, velocity])
                    next_state, env_reward, terminated, truncated, info = env.step(action)
                    next_i, next_j = discretizeState(next_state, env)
                    done = terminated or truncated

                    # Agent cannot depend on environment rewards alone. We can use modifyReward() function here too like in DQN.
                    shaped_reward = env_reward + modifyReward(next_state)
                    transition_table[(i, j, k)] = (next_i, next_j, shaped_reward, done)

        env.close()
        return transition_table

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithm",
        required=False,
        type=str,
        default='dqn',
        choices=['dqn', 'discretized'],
        help="Policy search algorithm.")
    parser.add_argument(
        "--record",
        required=False,
        default=False,
        action='store_true',
        help="Record the end result as a video.")
    parser.add_argument(
        "--episodes",
        required=False,
        type=int,
        default=100,
        help="The number of episodes to run")

    args = parser.parse_args()

    max_steps = 4000
    env = gym.make('MountainCarContinuous-v0', max_episode_steps=max_steps)

    # Continuous actions representing directional force applied to car.
    print(f"Action space: {env.action_space}")
    # We know the vehicle position and speed.
    print(f"Observation space: {env.observation_space}")

    # By default, program will run the DQN approach.
    if args.algorithm == 'dqn':

        # For example, with ACTION_BINS = 5, we have [[-1.0], [-0.5], [0.0], [0.5], [1.0]]
        discrete_actions = np.linspace(env.action_space.low, env.action_space.high, NUM_ACTIONS)
        num_actions = len(discrete_actions)
        num_obs = env.observation_space.shape[0]

        policy_dqn = DQN(state_dim=num_obs, action_dim=num_actions)
        target_dqn = DQN(state_dim=num_obs, action_dim=num_actions)

        target_dqn.load_state_dict(policy_dqn.state_dict())
        target_dqn.eval() # Target network will not be training.

        optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()

        # Hyperparameter and replay buffer initialization
        replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)
        batch_size = BATCH_SIZE
        gamma = GAMMA
        epsilon = EPSILON
        epsilon_decay = EPSILON_DECAY
        epsilon_min = EPSILON_MIN
        sync_target_dqn_rate = SYNC_TARGET_DQN_RATE

        for episode in range(args.episodes):
            # Reset the environment to put the agent into an initial state
            state, info = env.reset()

            # Run the simulation
            finished = False

            sim_steps = 0
            episode_reward = 0.0

            while not(finished):
                sim_steps += 1

                # Run epsilon-greedy policy to obtain action on policy network. 
                if random.random() < epsilon:
                    action_idx = random.randint(0, num_actions - 1)
                else:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    with torch.no_grad():
                        q_values = policy_dqn(state_tensor)
                    action_idx = torch.argmax(q_values).item()
                
                action = discrete_actions[action_idx]
                next_state, env_reward, terminated, truncated, info = env.step(action)
                episode_reward += env_reward
                
                reward = modifyReward(next_state)

                # Storing the experience in the replay buffer.
                replay_buffer.append((state, action_idx, reward, next_state, terminated))

                # If we have enough experiences in replay buffer, begin training.
                if len(replay_buffer) >= batch_size:
                    batch = random.sample(replay_buffer, batch_size)
                    states, actions_batch, rewards, next_states, dones = zip(*batch)

                    states_tensor = torch.FloatTensor(np.array(states))
                    actions_tensor = torch.LongTensor(actions_batch).unsqueeze(1)
                    rewards_tensor = torch.FloatTensor(rewards)
                    
                    # Q-values obtained from policy network. 
                    policy_q_values = policy_dqn(states_tensor).gather(1, actions_tensor).squeeze()

                    # Obtain next batch of states Q-values using target network. Filter out next states that are 'None'.
                    next_q_values = torch.zeros(batch_size)
                    filtered_next_states_mask = torch.tensor([s is not None for s in next_states], dtype=torch.bool)
                    filtered_next_states_tensor = torch.FloatTensor(np.array([s for s in next_states if s is not None]))

                    with torch.no_grad(): 
                        next_q_values[filtered_next_states_mask] = target_dqn(filtered_next_states_tensor).max(1)[0]

                    # Update target Q-values using DQL formula.
                    target_q_values = rewards_tensor + (gamma * next_q_values)

                    # Calculate loss and backpropogate on policy network.
                    loss = loss_fn(policy_q_values, target_q_values)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if terminated:
                    next_state = None

                # Update the state
                state = next_state

                finished = terminated or truncated

            # Decay epsilon per episode to decrease exploration.
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            
            # Periodically sync target network.
            if episode % sync_target_dqn_rate == 0:
                target_dqn.load_state_dict(policy_dqn.state_dict())

            print(f"Episode {episode + 1}/{args.episodes} | Steps: {sim_steps} | Total Reward: {episode_reward:.2f} | Epsilon: {epsilon:.3f}")
        
        env.close()

    # run discrete VI agent by using argument --algorithm discretized
    elif args.algorithm == 'discretized': 

        # Initialize value and policy tables.
        V = np.zeros((NUM_POS, NUM_VEL))
        policy = np.zeros((NUM_POS, NUM_VEL), dtype=int) # will hold the index to the next best action.    
        transition_table = createTransitionTable(env)

        iteration = 0
        while True: 
            delta = 0
            V_new = np.copy(V)

            for i in range(NUM_POS):
                for j in range(NUM_VEL):
                    action_values = []
                    for k in range(NUM_ACTIONS):
                        next_i, next_j, reward, done = transition_table[(i, j, k)]

                        if done: # if next state is terminal, take immediate reward.
                            q = reward
                        else: 
                            q = reward + GAMMA * V[next_i, next_j]
                        
                        action_values.append(q)

                    best_action_idx = int(np.argmax(action_values))
                    V_new[i, j] = action_values[best_action_idx]
                    policy[i, j] = best_action_idx

                    delta = max(delta, abs(V_new[i, j] - V[i, j]))
            V = V_new

            iteration += 1
            if iteration % 10 == 0: 
                print(f"Iteration {iteration} | max delta = {delta:.6f}")

            if delta < THETA: 
                print(f"Converged after {iteration} iterations.")
                break
                

    # Play with policy. If using --record argument, video will be created in directory of file.
    if args.record:
        env = gym.make('MountainCarContinuous-v0', max_episode_steps=4000, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(
            env,
            #episode_trigger=lambda num: num % 2 == 0,
            video_folder="./",
            name_prefix=f"mountain-car-{args.algorithm}",
        )
    else:
        env = gym.make('MountainCarContinuous-v0', max_episode_steps=4000, render_mode="human")
    
    state, info = env.reset()
    # Run the simulation
    finished = False

    while not(finished):
        if args.algorithm == 'discretized':
            # Get the next action from policy using VI
            discrete_actions = np.linspace(env.action_space.low, env.action_space.high, NUM_ACTIONS)

            i, j = discretizeState(state, env)
            action_idx = policy[i, j]
            action = discrete_actions[action_idx]

        else:
            # Get the next action from your DQN
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = policy_dqn(state_tensor)
            
            action_idx = torch.argmax(q_values).item()
            action = discrete_actions[action_idx]

        # Take the action
        next_state, reward, terminated, truncated, info = env.step(action)

        # Update the state
        state = next_state

        finished = terminated or truncated
    env.close()
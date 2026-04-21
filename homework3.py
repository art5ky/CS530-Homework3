"""
Rutgers University, Spring 2026
CS530, Homework 3 - DQN Agent and Discretized Agent
Written by: Arthur Levitsky
"""

import argparse
import gymnasium as gym
import random
import numpy as np
import torch 
import torch.nn as nn
from collections import deque

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
    For calculating DQN reward, utilizing the mechanical energy formula.
    The agent is encouraged to stay out of the bottom of the valley through potential energy. 
    The agent is also encouraged to increase speed through it's actions and momentum via kinetic energy.
    If agent is at position 0.45 (the flag), give a big reward to signify that reaching the flag is the ultimate objective.
    """
    position, velocity = next_state

    if position >= 0.45:
        return 100.0

    # We scale reward from velocity higher to make sure that the agent doesn't just prioritize being on the slopes of the hills. 
    return abs(velocity) * 20 + abs(position + 0.5)

def discretizeState(state, env, pos_bins=20, vel_bins=20):
    """
    Discretize the state. Used with value iteration.
    """
    pos_low, vel_low = env.observation_space.low
    pos_high, vel_high = env.observation_space.high
    position, velocity = state

    pos_idx = int(np.digitize(position, np.linspace(pos_low, pos_high, pos_bins)))
    vel_idx = int(np.digitize(velocity, np.linspace(vel_low, vel_high, vel_bins)))

    return (pos_idx, vel_idx)

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

        discrete_actions = [[-1.0], [-0.5], [0.0], [0.5], [1.0]] # Convert continuous actions into three discrete actions.
        num_actions = len(discrete_actions)

        policy_dqn = DQN(state_dim=2, action_dim=num_actions)
        target_dqn = DQN(state_dim=2, action_dim=num_actions)

        target_dqn.load_state_dict(policy_dqn.state_dict())
        target_dqn.eval() # Target network will not be training.

        optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()

        # Hyperparameter and replay buffer initialization
        replay_buffer = deque(maxlen=10000)
        batch_size = 64
        gamma = 0.99
        epsilon = 1.0 # Start with 100% random actions
        epsilon_decay = 0.95 # Multiply epsilon by this after each episode
        epsilon_min = 0.01 # Never explore less than 1% of the time
        sync_target_dqn_rate = 5

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
    
    # run discrete agent by using argument --algorithm discretized
    elif args.algorithm == 'discretized': 
        pos_bins = 10
        vel_bins = 10
        discrete_actions = [-1.0, -0.5, 0.0, 0.5, 1.0]
        num_actions = len(discrete_actions)

        q_table = np.zeros((pos_bins + 1, vel_bins + 1, num_actions))
        flag_state = discretizeState([0.45, 0.0], env, pos_bins, vel_bins)
        flag_pos_idx, flag_vel_idx = flag_state
        q_table[flag_pos_idx, :, :] = 100.0

        # Seed high velocity states on the left hill as valuable
        left_hill_state = discretizeState([-1.0, 0.0], env, pos_bins, vel_bins)
        left_hill_pos_idx = left_hill_state[0]

        # High rightward velocity columns (right half of vel axis)
        q_table[left_hill_pos_idx, vel_bins//2:, :] = 25.0

        max_q = np.max(q_table, axis=2)
        np.savetxt("q_table.csv", max_q, delimiter=',', fmt='%.2f')

        alpha = 0.5 
        gamma = 0.99
        epsilon = 1.0 # Start with 100% random actions
        epsilon_min = 0.01 # Never explore less than 1% of the time
        target_episode = args.episodes * 10 / 2
        epsilon_decay = (epsilon_min / epsilon) ** (1 / target_episode) # Multiply epsilon by this after each episode

        
        for episode in range(args.episodes * 10):
            # Reset the environment to put the agent into an initial state
            state, info = env.reset()

            # Run the simulation
            finished = False
            state = discretizeState(state, env, pos_bins, vel_bins)
            sim_steps = 0
            sim_steps_rate = 500

            while not(finished):
                sim_steps += 1

                if random.random() < epsilon: 
                    action_idx = random.randint(0, num_actions -1)
                else: 
                    action_idx = np.argmax(q_table[state])

                action = [discrete_actions[action_idx]]
                next_state, env_reward, terminated, truncated, info = env.step(action)
                next_state = discretizeState(next_state, env, pos_bins, vel_bins)

                # update Q-table
                best_next_q = 0 if terminated else np.max(q_table[next_state])
                q_table[state][action_idx] += alpha * (env_reward + gamma * best_next_q - q_table[state][action_idx])
                    
                state = next_state
                finished = terminated or truncated

            # Decay epsilon per episode to decrease exploration.
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

            print(f"Episode {episode+1}/{args.episodes * 10} | Steps: {sim_steps} | Epsilon: {epsilon:.3f}")

        max_q = np.max(q_table, axis=2)
        np.savetxt("q_table_end.csv", max_q, delimiter=',', fmt='%.2f')

    env.close()

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
            pos_bins = 10
            vel_bins = 10


            # The discretized model should not require learning, converging instead through value iteration.
            state_d = discretizeState(state, env, pos_bins, vel_bins)
            action = [discrete_actions[np.argmax(q_table[state_d])]]
        else:
            # Get the next action from your DQN
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = policy_dqn(state_tensor)
            
            action_idx = torch.argmax(q_values).item()
            action = discrete_actions[action_idx]

        # Take the action
        #print(f"Taking action {action} from state {state}")
        next_state, reward, terminated, truncated, info = env.step(action)

        # Update the state
        state = next_state

        finished = terminated or truncated
    env.close()
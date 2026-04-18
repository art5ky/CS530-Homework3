import argparse
import gymnasium as gym
import pickle
import random
import numpy as np
import torch 
import torch.nn as nn
from collections import deque

def modifyReward(state, next_state, action, reward):
    """
    Modify the reward so that your DQN policy search succeeds.
    """
    position, velocity = next_state

    if position >= 0.45:
        return 100.0
    
    #currently, reward for kinetic energy and potential energy
    custom_reward = abs(velocity) * 10 + abs(position + 0.5)

    return custom_reward


def discretizeState(state):
    """
    Discretize the state. Used with value iteration.
    """
    # This is part of your homework
    pass

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

    # We can go up, down, left, or right
    print(f"Action space: {env.action_space}")
    # We know the vehicle position and speed
    print(f"Observation space: {env.observation_space}")

    # --- Pre-Training Setup ---
    # 1. Map discrete actions to continuous environment inputs
    discrete_actions = [[-1.0], [0.0], [1.0]] 
    num_actions = len(discrete_actions)

    # 2. Define the Deep Q-Network
    class QNetwork(nn.Module):
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

    # 3. Initialize components
    q_net = QNetwork(state_dim=2, action_dim=num_actions)
    optimizer = torch.optim.Adam(q_net.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    # 4. Hyperparameters and Replay Buffer
    replay_buffer = deque(maxlen=10000)
    batch_size = 64
    gamma = 0.99
    epsilon = 1.0         # Start with 100% random actions
    epsilon_decay = 0.95 # Multiply epsilon by this after each episode
    epsilon_min = 0.01    # Never explore less than 1% of the time
    # --------------------------

    # The discretized, value-iteration solution does not need training.
    if args.algorithm != 'discretized':
        for episode in range(args.episodes):
            # Reset the environment to put the agent into an initial state
            state, info = env.reset()

            # Run the simulation
            finished = False

            sim_steps = 0
            episode_reward = 0.0

            while not(finished):
                sim_steps += 1
                # TODO Get an action from your model
                if random.random() < epsilon:
                    # Explore: pick a random action index
                    action_idx = random.randint(0, num_actions - 1)
                else:
                    # Exploit: use the network to pick the best action
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    with torch.no_grad():
                        q_values = q_net(state_tensor)
                    action_idx = torch.argmax(q_values).item()
                
                # Map the chosen index back to the format the env expects
                action = discrete_actions[action_idx]

                next_state, reward, terminated, truncated, info = env.step(action)

                episode_reward += reward

                if args.algorithm == 'dqn':
                    reward = modifyReward(state, next_state, action, reward)

                # Perform an update step for your deep Q network.
                # TODO
                # 1. Store the experience in the replay buffer
                replay_buffer.append((state, action_idx, reward, next_state, terminated))

                # 2. Only train if we have enough samples in the buffer
                if len(replay_buffer) >= batch_size:
                    batch = random.sample(replay_buffer, batch_size)
                    states, actions_batch, rewards, next_states, dones = zip(*batch)

                    states_tensor = torch.FloatTensor(np.array(states))
                    actions_tensor = torch.LongTensor(actions_batch).unsqueeze(1)
                    rewards_tensor = torch.FloatTensor(rewards)
                    
                    # 3. Handle 'None' next_states based on skeleton code logic
                    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_states)), dtype=torch.bool)
                    non_final_next_states = torch.FloatTensor(np.array([s for s in next_states if s is not None]))

                    # Current Q-values for the actions taken
                    q_values = q_net(states_tensor).gather(1, actions_tensor).squeeze()

                    # Calculate Next Q-values
                    next_q_values = torch.zeros(batch_size)
                    if len(non_final_next_states) > 0:
                        with torch.no_grad():
                            next_q_values[non_final_mask] = q_net(non_final_next_states).max(1)[0]

                    # Target Q-values (bellman equation)
                    target_q_values = rewards_tensor + (gamma * next_q_values)

                    # 4. Calculate Loss and Backpropagate
                    loss = loss_fn(q_values, target_q_values)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if terminated:
                    next_state = None

                # Update the state
                state = next_state

                finished = terminated or truncated

            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            print(f"Episode {episode + 1}/{args.episodes} | Steps: {sim_steps} | Total Reward: {episode_reward:.2f} | Epsilon: {epsilon:.3f}")
    
    env.close()

    # Play with policy
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
            # The discretized model should not require learning, converging instead through value iteration.
            state = discretizeState(state)
            next_state = discretizeState(next_state)
            # TODO Get the next action from the discretized model
            action = [0]
        else:
            # TODO Get the next action from your DQN
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = q_net(state_tensor)
            
            action_idx = torch.argmax(q_values).item()
            action = discrete_actions[action_idx]

        # Take the action
        #print(f"Taking action {action} from state {state}")
        next_state, reward, terminated, truncated, info = env.step(action)

        # Update the state
        state = next_state

        finished = terminated or truncated
    env.close()
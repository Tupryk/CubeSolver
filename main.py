from itertools import count

import matplotlib.pyplot as plt
import numpy as np
import torch

from agent import env, select_action, optimize_model, policy_net, target_net, episode_durations, TAU, device, memory

num_episodes = 5000
total_rewards = []
for i_episode in range(num_episodes):
    
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    total_reward = 0
    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        total_reward += reward
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        memory.push(state, action, next_state, reward)

        state = next_state

        optimize_model()

        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            break

    total_rewards.append(total_reward)
    if (i_episode+1) % 1000 == 0:
        env.render()
        print(f'Episode {i_episode+1}/{num_episodes}, End Reward: {total_reward}.')

print('Complete')

scaler = num_episodes//100
total_rewards = np.array(total_rewards)
total_rewards = np.average(total_rewards.reshape(-1, scaler), axis=1)
episode_durations = np.array(episode_durations)
episode_durations = np.average(episode_durations.reshape(-1, scaler), axis=1)

# Display training results
plt.plot(total_rewards, "g-")
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('End reward')
plt.show()

plt.plot(episode_durations, "r-")
plt.xlabel('Episode')
plt.ylabel('Moves')
plt.title('Time taken')
plt.show()
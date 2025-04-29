import numpy as np
import matplotlib.pyplot as plt

data = np.load("logs/logs_ppo/random_bid_2_cards/evaluations.npz")
timesteps = data["timesteps"]
results = data["results"]  # shape (n_evals, n_eval_episodes)

# Mean and std across evaluation episodes
mean_rewards = results.mean(axis=1)
std_rewards = results.std(axis=1)

# Plot
plt.plot(timesteps, mean_rewards, label="Mean reward")
plt.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.3)
plt.xlabel("Timesteps")
plt.ylabel("Evaluation Reward")
plt.title("Evaluation Rewards Over Time")
plt.legend()
plt.grid(True)
plt.show()
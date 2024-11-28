import numpy as np

class BernoulliBandit:
    def __init__(self, horizon=2000, num_arms=5, prob_min=0.1, prob_max=0.9):
        """
        Parameters:
            horizon: Total number of steps (time horizon).
            num_arms: Number of arms (bandits).
            prob_min: Minimum success probability for each arm.
            prob_max: Maximum success probability for each arm.
        """
        self.T = 0
        self.horizon = horizon
        self.num_arms = num_arms
        self.success_probs = np.random.uniform(prob_min, prob_max, size=num_arms)  # Success probabilities for each arm
        self.total_rewards = np.zeros(num_arms)  # Cumulative reward for each arm
        self.reward_history = []  # Record of rewards at each step
        self.regret_history = []  # Record of regrets at each step
        self.arm_pull_counts = np.zeros(num_arms)  # Count of pulls for each arm
        self.arm_rewards = [[] for _ in range(num_arms)]  # List of rewards for each arm
        self.arm_history = []
        
    def pull_arm(self, arm_index):
        """
        Simulates pulling an arm and returns the reward.
        
        Parameters:
            arm_index: Index of the arm to pull.
        
        Returns:
            reward: 1 if success, 0 otherwise.
        """
        reward = np.random.binomial(1, self.success_probs[arm_index])  # Bernoulli reward
        self.total_rewards[arm_index] += reward
        self.regret_history.append(max(self.success_probs) - self.success_probs[arm_index])
        self.arm_pull_counts[arm_index] += 1
        self.arm_rewards[arm_index].append(reward)
        self.reward_history.append(reward)
        self.T += 1
        self.arm_history.append(arm_index)
        return reward

    def reset(self, prob_min=0.1, prob_max=0.9):
        """
        Resets the bandit to its initial state with new success probabilities.
        """
        self.success_probs = np.random.uniform(prob_min, prob_max, size=self.num_arms)
        self.total_rewards = np.zeros(self.num_arms)
        self.reward_history = []
        self.regret_history = []
        self.arm_history = []
        self.arm_pull_counts = np.zeros(self.num_arms)
        self.arm_rewards = [[] for _ in range(self.num_arms)]
        self.T = 0

    def get_T(self):
        return self.T

    def get_regrets(self):
        return self.regret_history

    def cumulative_rewards(self):
        """
        Returns the cumulative rewards for all arms.
        """
        return self.total_rewards

    def cumulative_regret(self):
        """
        Returns the cumulative regret over all steps.
        """
        return np.cumsum(self.regret_history)

    def success_probabilities(self):
        """
        Returns the true success probabilities of the arms.
        """
        return self.success_probs

    def pulls_per_arm(self):
        """
        Returns the number of times each arm has been pulled.
        """
        return self.arm_pull_counts

    def reward_per_arm(self):
        """
        Returns the list of rewards for each arm.
        """
        return self.arm_rewards

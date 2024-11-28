import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import scipy
import scipy.stats

def random_argmax(arr):
    """
    Returns the index of the maximum value in the array, breaking ties randomly.
    """
    max_indices = np.where(arr == np.max(arr))[0]
    return np.random.choice(max_indices)

def run_algorithm(mab, alg, R=10):
    T = mab.horizon
    regret_list = np.zeros((R, T))  # To store regrets for each repetition
    for r in range(R):
        for t in range(T):
            alg.play_one_step()  # Run one step of the algorithm
        regret_list[r, :] = mab.cumulative_regret()  # Collect regrets
        if r != (R-1):
            alg.reset()  # Reset algorithm for the next repetition
    return regret_list

def plot_regret_single(mab, regrets, alg_name):
    T = mab.horizon
    mean_regret = np.mean(regrets, axis=0)
    lower_bound = np.percentile(regrets, 5, axis=0)
    upper_bound = np.percentile(regrets, 95, axis=0)

    plt.figure(figsize=(8, 6))
    plt.plot(range(T), mean_regret, label="Mean Regret")
    plt.fill_between(
        range(T), lower_bound, upper_bound, color="b", alpha=0.1, label="90% CI"
    )
    plt.xlabel("Step")
    plt.ylabel("Cumulative Regret")
    plt.title(f"Cumulative Regret: {alg_name}")
    plt.axhline(mean_regret[-1], linestyle = '--', color = 'grey', alpha = 0.5)
    plt.text(
        T - 1, mean_regret[-1] + 0.1,  # Position of the label (slightly above the line)
        f"{mean_regret[-1]:.2f}",       # Format the value to 2 decimal places
        color='grey', fontsize=10, ha='center', va='bottom'
    )
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_arm_choices(horizon, arm_history, name):
    """
    Plots the arms chosen at each timestep.

    Parameters:
        horizon (int): The number of timesteps to visualize.
        arm_history (list or array): History of arms chosen, truncated to the last `horizon` steps.
    """
    # Extract the unique arms
    unique_arms = np.unique(arm_history)
    colors = plt.cm.Set1(np.linspace(0, 1, 5))  # Using a colormap
    
    # Create the plot
    plt.figure(figsize=(6, 4))
    
    for arm in unique_arms:
        indices = np.where(arm_history == arm)[0]
        plt.scatter(indices, [arm] * len(indices), 
                    color=colors[arm], s=2, label=f'Arm {arm}', alpha=0.7)
    
    # Formatting the plot
    plt.title('Arm Chosen by {} Over Time'.format(name), fontsize=14)
    plt.xlabel('Timestep $t$', fontsize=12)
    plt.ylabel('Arm', fontsize=12)
    plt.yticks(unique_arms)
    plt.legend(title='Arms', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_ucb(stats, t):
    true_means = stats["true_means"]
    mean_estimates = stats["mean_estimates"]
    ucb = stats["ucb"]
    
    num_arms = len(true_means)
    x = np.arange(num_arms)  # Arm indices

    # Calculate lower confidence bounds
    lcb = 2 * mean_estimates - ucb

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plotting points for true means, mean estimates, and UCB
    plt.scatter(x, true_means, label='True Means', color='black', zorder=3, alpha = 0.2)

    # Plot confidence intervals as one-sided error bars
    plt.errorbar(x, mean_estimates, 
             yerr=ucb - mean_estimates,  # Only upper error
             fmt='o', color='orange', ecolor='gray', capsize=5, label='Upper Confidence Interval', zorder=2)

    for i in range(num_arms):
        plt.text(x[i], ucb[i] + 0.01, f"{ucb[i]:.3f}", 
                 ha='center', va='bottom', fontsize=9, color='grey', weight='bold')

    # Add labels, legend, and title
    plt.xlabel("Arms")
    plt.ylabel("Values")
    plt.title("Steps = {}".format(t))
    plt.xticks(x, [f"Arm {i}" for i in x])
    plt.legend()
 # Adjust the y-axis for better visibility
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    
    # Show the plot
    plt.show()

def plot_beta_distribution(alpha, beta_param, num_points=1000):
    """
    Plots the Beta distribution for given alpha and beta parameters.

    Parameters:
        alpha: Shape parameter (successes) of the Beta distribution.
        beta_param: Shape parameter (failures) of the Beta distribution.
        num_points: Number of points for the plot (default: 1000).
    """
    # Generate x values between 0 and 1
    x = np.linspace(0, 1, num_points)
    # Compute the Beta PDF
    y = beta.pdf(x, alpha, beta_param)

    # Plot the Beta distribution
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, label=f"Beta(α={alpha}, β={beta_param})", color="blue")
    plt.fill_between(x, y, alpha=0.2, color="blue")  # Shaded area under the curve
    #plt.title("Beta Distribution")
    plt.xlabel("x")
    plt.ylabel("Density")
    #plt.legend()
    plt.ylim(0, 7)
    plt.grid(alpha=0.3)
    plt.show()



def plot_thompson(mab, alphas, betas, t):
    # Plot posterior distribution for each arm (Beta distribution)
    plt.figure(figsize = (10,6))
    x = np.linspace(0, 1, 1000)  # Range of theta (success probability)
    
    # Create a color map for arm colors
    colors = plt.cm.Set1(np.linspace(0, 1, mab.num_arms))  # Using a colormap
    
    for arm in range(mab.num_arms):
        # Compute the Beta distribution for each arm
        y = scipy.stats.beta.pdf(x, alphas[arm], betas[arm])
        
        # Plot the Beta distribution line
        plt.plot(x, y, label=f'Arm {arm+1} (α={alphas[arm]}, β={betas[arm]})', color=colors[arm])
        
        # Fill the area under the Beta distribution curve
        plt.fill_between(x, y, alpha=0.1, color=colors[arm])  # alpha for transparency
        
        # Add a dotted line for the true mean (success probability) of each arm
        plt.axvline(mab.success_probs[arm], color=colors[arm], linestyle='dotted', linewidth=2, alpha = 0.7)

    # Set plot properties
    plt.title(f'Posterior Distributions of Arms at Time Step {t+1}')
    plt.xlabel('θ (Success Probability)')
    plt.ylabel('Density')
    plt.legend(loc='upper left')
    #plt.ylim(0,10)
    current_ylim = plt.gca().get_ylim()
    plt.ylim(top=max(10, current_ylim[1]))
    
    # Show plot
    plt.show()

# Example usage:
# You can call the function in your loop like this:
# plot_thompson(mab, alg.alpha, alg.beta, t)
def plot_regret_double(mab, regrets_1, regrets_2, alg_name_1, alg_name_2):
    T = mab.horizon
    
    # Compute mean and confidence intervals for both algorithms
    mean_regret_1 = np.mean(regrets_1, axis=0)
    lower_bound_1 = np.percentile(regrets_1, 5, axis=0)
    upper_bound_1 = np.percentile(regrets_1, 95, axis=0)
    
    mean_regret_2 = np.mean(regrets_2, axis=0)
    lower_bound_2 = np.percentile(regrets_2, 5, axis=0)
    upper_bound_2 = np.percentile(regrets_2, 95, axis=0)

    plt.figure(figsize=(8, 6))
    
    # Plot mean regrets for both algorithms
    plt.plot(range(T), mean_regret_1, label=f"Mean Regret: {alg_name_1}")
    plt.plot(range(T), mean_regret_2, label=f"Mean Regret: {alg_name_2}")

    # Fill the areas for 90% CI for both algorithms
    plt.fill_between(range(T), lower_bound_1, upper_bound_1, color="b", alpha=0.1)
    plt.fill_between(range(T), lower_bound_2, upper_bound_2, color="orange", alpha=0.1)

    # Add labels and title
    plt.xlabel("Step")
    plt.ylabel("Cumulative Regret")
    plt.title(f"Comparison of Cumulative Regret: {alg_name_1} vs {alg_name_2}")
    
    # Add horizontal lines at the final mean regret values for both algorithms
    plt.axhline(mean_regret_1[-1], linestyle='--', color='blue', alpha=0.5)
    plt.axhline(mean_regret_2[-1], linestyle='--', color='orange', alpha=0.5)
    
    # Add numerical labels above the horizontal lines
    plt.text(
        T - 1, mean_regret_1[-1] + 0.1,  # Position of the label for algorithm 1
        f"{mean_regret_1[-1]:.2f}",       # Format the value to 2 decimal places
        color='blue', fontsize=10, ha='center', va='bottom'
    )
    plt.text(
        T - 1, mean_regret_2[-1] + 0.1,  # Position of the label for algorithm 2
        f"{mean_regret_2[-1]:.2f}",       # Format the value to 2 decimal places
        color='orange', fontsize=10, ha='center', va='bottom'
    )

    # Add legend
    plt.legend()
    plt.tight_layout()
    plt.show()
def plot_regret_triple(mab, regrets_1, regrets_2, regrets_3, alg_name_1, alg_name_2, alg_name_3):
    T = mab.horizon
    
    # Compute mean and confidence intervals for all three algorithms
    mean_regret_1 = np.mean(regrets_1, axis=0)
    lower_bound_1 = np.percentile(regrets_1, 5, axis=0)
    upper_bound_1 = np.percentile(regrets_1, 95, axis=0)
    
    mean_regret_2 = np.mean(regrets_2, axis=0)
    lower_bound_2 = np.percentile(regrets_2, 5, axis=0)
    upper_bound_2 = np.percentile(regrets_2, 95, axis=0)
    
    mean_regret_3 = np.mean(regrets_3, axis=0)
    lower_bound_3 = np.percentile(regrets_3, 5, axis=0)
    upper_bound_3 = np.percentile(regrets_3, 95, axis=0)

    plt.figure(figsize=(10, 7))
    
    # Plot mean regrets for all three algorithms
    plt.plot(range(T), mean_regret_1, label=f"Mean Regret: {alg_name_1}", color="blue")
    plt.plot(range(T), mean_regret_2, label=f"Mean Regret: {alg_name_2}", color="orange")
    plt.plot(range(T), mean_regret_3, label=f"Mean Regret: {alg_name_3}", color="green")

    # Fill the areas for 90% CI for all three algorithms
    plt.fill_between(range(T), lower_bound_1, upper_bound_1, color="blue", alpha=0.1)
    plt.fill_between(range(T), lower_bound_2, upper_bound_2, color="orange", alpha=0.1)
    plt.fill_between(range(T), lower_bound_3, upper_bound_3, color="green", alpha=0.1)

    # Add labels and title
    plt.xlabel("Step")
    plt.ylabel("Cumulative Regret")
    plt.title(f"Comparison of Cumulative Regret: {alg_name_1} vs {alg_name_2} vs {alg_name_3}")
    
    # Add horizontal lines at the final mean regret values for all algorithms
    plt.axhline(mean_regret_1[-1], linestyle='--', color='blue', alpha=0.5)
    plt.axhline(mean_regret_2[-1], linestyle='--', color='orange', alpha=0.5)
    plt.axhline(mean_regret_3[-1], linestyle='--', color='green', alpha=0.5)
    
    # Add numerical labels above the horizontal lines
    plt.text(
        T - 1, mean_regret_1[-1] + 0.1,  # Position of the label for algorithm 1
        f"{mean_regret_1[-1]:.2f}",       # Format the value to 2 decimal places
        color='blue', fontsize=10, ha='center', va='bottom'
    )
    plt.text(
        T - 1, mean_regret_2[-1] + 0.1,  # Position of the label for algorithm 2
        f"{mean_regret_2[-1]:.2f}",       # Format the value to 2 decimal places
        color='orange', fontsize=10, ha='center', va='bottom'
    )
    plt.text(
        T - 1, mean_regret_3[-1] + 0.1,  # Position of the label for algorithm 3
        f"{mean_regret_3[-1]:.2f}",       # Format the value to 2 decimal places
        color='green', fontsize=10, ha='center', va='bottom'
    )

    # Add legend
    plt.legend()
    plt.tight_layout()
    plt.show()



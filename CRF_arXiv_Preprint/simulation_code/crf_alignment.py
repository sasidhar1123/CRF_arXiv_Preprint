import numpy as np
import matplotlib.pyplot as plt
import time


np.random.seed(42)           # For reproducibility
N_COMPONENTS = 1000          # Number of AI components (e.g., neural network layers)
GAMMA = 1e-8                 # Resonance sensitivity parameter
LEARNING_RATE = 1000         # Optimization step size
MAX_ITERATIONS = 500         # Maximum optimization steps
CONVERGENCE_THRESH = 1e-4    # Stopping criterion (gradient magnitude)
INITIAL_MEAN_DELAY = 20      # Mean processing delay (ms)
INITIAL_STD_DEV = 10         # Initial delay variability (ms)


 
def compute_resonance(delays):
    n = len(delays)
    pairwise_diff = delays.reshape(1, -1) - delays.reshape(-1, 1)
    sum_sq_diff = np.sum(np.triu(pairwise_diff**2, k=1))
    return np.exp(-GAMMA * sum_sq_diff)

def compute_gradient(delays, resonance):
    n = len(delays)
    # Vectorized gradient calculation
    return -2 * GAMMA * resonance * (n * delays - np.sum(delays))


# Initialize random delays
delays = np.random.normal(INITIAL_MEAN_DELAY, INITIAL_STD_DEV, N_COMPONENTS)
initial_delays = delays.copy()

# Tracking variables
resonance_history = []
start_time = time.time()

# Optimization loop
for step in range(MAX_ITERATIONS):
    # Calculate current resonance
    R = compute_resonance(delays)
    resonance_history.append(R)
    
    # Compute alignment forces
    gradient = compute_gradient(delays, R)
    gradient_norm = np.linalg.norm(gradient)
     
    # Check convergence
    if gradient_norm < CONVERGENCE_THRESH:
        print(f"Converged at step {step} with R= {R:.6f}")
        break
    
    # Update delays (gradient ASCENT to maximize resonance)
    delays += LEARNING_RATE * gradient
    
    # Progress reporting
    if step % 50 == 0:
        print(f"Step {step:4d}: R = {R:.6f}, |∇| = {gradient_norm:.6f}")

# Simulation metrics
final_mean = np.mean(delays)
final_std = np.std(delays)
sim_time = time.time() - start_time


# VISUALIZATION

plt.figure(figsize=(12, 5))

# Resonance convergence plot
plt.subplot(1, 2, 1)
plt.plot(resonance_history, 'b-', linewidth=2)
plt.xlabel("Optimization Step", fontsize=12)
plt.ylabel(r"Resonance" , fontsize=12)
plt.title("CRF Alignment Convergence", fontsize=14)
plt.grid(alpha=0.2)
plt.ylim(0, 1.05)

# Delay distribution comparison
plt.subplot(1, 2, 2)
plt.hist(initial_delays, bins=30, alpha=0.6, label="Initial", color='red')
plt.hist(delays, bins=30, alpha=0.6, label="Aligned", color='green')
plt.axvline(INITIAL_MEAN_DELAY, color='k', linestyle='--', label='Target')
plt.xlabel("Processing Delay (ms)", fontsize=12)
plt.ylabel("Component Count", fontsize=12)
plt.title("Delay Distribution", fontsize=14)
plt.legend()
plt.grid(alpha=0.2)

plt.tight_layout()
plt.savefig('crf_alignment_results.png', dpi=300, bbox_inches='tight')


# RESULTS SUMMARY

print("CRF ALIGNMENT SIMULATION RESULTS")
print(f"Components:           {N_COMPONENTS}")
print(f"Final resonance (R):  {resonance_history[-1]:.6f}")
print(f"Initial mean delay:   {np.mean(initial_delays):.2f} ± {np.std(initial_delays):.2f} ms")
print(f"Final mean delay:     {final_mean:.2f} ± {final_std:.2f} ms")
print(f"Variance reduction:   {np.var(initial_delays)/np.var(delays):.1f}x")
print(f"Optimization time:    {sim_time:.2f} seconds")
print("="*60)






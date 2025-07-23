import numpy as np
import matplotlib.pyplot as plt
import time


np.random.seed(42)  # For reproducible results
N_COMPONENTS = 50   # Number of AI components (e.g., neural layers)
SIMULATION_STEPS = 500  # Time steps to simulate
DRIFT_NOISE = 0.015  # System noise level (milliseconds)
ALERT_THRESHOLD = 40  # Drift threshold for alerts


def calculate_drift(delays):
    n = len(delays)
    sum_squared = np.sum(delays ** 2)
    sum_total = np.sum(delays)
    return n * sum_squared - sum_total ** 2


print("=" * 70)
print("CRF DRIFT DETECTION SIMULATION")
print("=" * 70)
print(f"System Components: {N_COMPONENTS}")
print(f"Simulation Steps: {SIMULATION_STEPS}")
print(f"Process Noise: {DRIFT_NOISE} ms per step")
print(f"Alert Threshold: {ALERT_THRESHOLD}")
print("=" * 70 + "\n")

delays = np.full(N_COMPONENTS, 10.0)  # All components start at 10ms
drift_history = np.zeros(SIMULATION_STEPS)  # Store drift values
alert_history = np.zeros(SIMULATION_STEPS, dtype=bool)  # Track alerts


start_time = time.time()
first_alert_step = None

for step in range(SIMULATION_STEPS):
    # Add random noise to simulate real-world drift
    delays += np.random.normal(0, DRIFT_NOISE, N_COMPONENTS)
    
    # Calculate current drift level
    current_drift = calculate_drift(delays)
    drift_history[step] = current_drift
    
    # Check if drift exceeds threshold
    if current_drift > ALERT_THRESHOLD:
        alert_history[step] = True
        
        # Record first alert after initial warm-up
        if first_alert_step is None and step > 20:
            first_alert_step = step
            print(f"FIRST ALERT at step {step}: Drift = {current_drift:.2f}")


plt.figure(figsize=(10, 6))
plt.plot(drift_history, 'b-', linewidth=1.5, alpha=0.8, label='Drift Level')

# Add threshold line and alert region
plt.axhline(ALERT_THRESHOLD, color='r', linestyle='--', linewidth=2, 
            label=f'Alert Threshold ({ALERT_THRESHOLD})')

if np.any(alert_history):
    plt.fill_between(range(SIMULATION_STEPS), ALERT_THRESHOLD, drift_history,
                    where=(drift_history > ALERT_THRESHOLD),
                    color='red', alpha=0.15, label='Alert Region')

# Add first alert annotation
if first_alert_step is not None:
    plt.annotate(f'First Alert: Step {first_alert_step}', 
                 xy=(first_alert_step, ALERT_THRESHOLD * 1.1),
                 xytext=(first_alert_step + 30, ALERT_THRESHOLD * 4),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=12,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray"))


plt.title("CRF Drift Detection Performance", fontsize=16, pad=15)
plt.xlabel("Time Step", fontsize=14, labelpad=10)
plt.ylabel(r"Drift Metric: $\sum_{i<j}(\tau_i - \tau_j)^2$", 
           fontsize=14, labelpad=10)
plt.legend(loc='upper left', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

plt.savefig('crf_drift_detection.pdf', format='pdf', bbox_inches='tight')


max_drift = np.max(drift_history)
final_drift = drift_history[-1]
alert_count = np.sum(alert_history)
execution_time = time.time() - start_time

print("\n" + "=" * 70)
print("SIMULATION RESULTS")
print("=" * 70)
print(f"Maximum Drift: {max_drift:.2f}")
print(f"Final Drift: {final_drift:.2f}")
print(f"Alert Steps: {alert_count} ({alert_count/SIMULATION_STEPS:.1%} of time)")
print(f"First Alert: Step {first_alert_step}" if first_alert_step else "No alerts triggered")
print(f"Execution Time: {execution_time:.4f} seconds")
print(f"Figure saved to: crf_drift_detection.pdf")
print("=" * 70)


# Test with simple case to verify formula
test_delays = np.array([1.0, 2.0, 3.0])
expected = (1-2)**2 + (1-3)**2 + (2-3)**2  # Manual calculation
calculated = calculate_drift(test_delays)   # Function calculation

print("\nMATHEMATICAL VALIDATION:")
print(f"Expected Value: {expected:.1f}")
print(f"Calculated Value: {calculated:.1f}")
print("Status: VALID" if np.isclose(expected, calculated) else "Status: INVALID")
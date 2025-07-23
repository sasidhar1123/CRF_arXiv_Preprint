import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.metrics import mean_squared_error
import time
np.random.seed(42)       # Consistent results
num_systems = 1000       # More systems = better accuracy
num_delays = 100         # Components per AI system
compression_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # 10-90%


def calculate_alignment(delays):
    total_diff = 0
    for i in range(len(delays)):
        for j in range(i+1, len(delays)):
            total_diff += (delays[i] - delays[j])**2
    return np.exp(-0.00001 * total_diff)  # Adjusted sensitivity

def compress_delays(delays, keep_ratio):
    """Keep most variable delays, mean for others"""
    keep_count = max(1, int(len(delays) * keep_ratio))
    
    # Find most variable delays (biggest differences from mean)
    mean_delay = np.mean(delays)
    deviations = np.abs(delays - mean_delay)
    important = np.argsort(deviations)[-keep_count:]
    
    # Reconstruct with kept delays + mean for others
    reconstructed = np.full_like(delays, mean_delay)
    reconstructed[important] = delays[important]
    
    return reconstructed


print("Running improved CRF compression test...")
start_time = time.time()

alignment_errors = []

for ratio in compression_levels:
    ratio_errors = []
    
    for _ in range(num_systems):
        # Realistic delays (gamma distribution)
        delays = 10 + 5 * np.random.randn(num_delays)  # Normal distribution
        
        # Calculate original alignment
        original_alignment = calculate_alignment(delays)
        
        # Compress and reconstruct
        reconstructed = compress_delays(delays, ratio)
        new_alignment = calculate_alignment(reconstructed)
        
        # Record error
        ratio_errors.append(np.abs(original_alignment - new_alignment))
    
    alignment_errors.append(np.mean(ratio_errors))

# Find best compression level
best_ratio = compression_levels[np.argmin(alignment_errors)]
best_error = np.min(alignment_errors)


plt.figure(figsize=(9, 6))
plt.plot(np.array(compression_levels)*100, alignment_errors, 'bo-', 
         markersize=8, linewidth=2.5)
plt.axvline(best_ratio*100, color='red', linestyle='--', linewidth=2,
            label=f'Optimal: {best_ratio*100:.0f}%')
plt.title("CRF Compression Performance", fontsize=16)
plt.xlabel("Percentage of Delays Kept", fontsize=14)
plt.ylabel("Alignment Error", fontsize=14)
plt.grid(alpha=0.3)
plt.legend(fontsize=12)

# Add physics explanation
plt.annotate('Low Compression: Too much information loss', 
             xy=(15, max(alignment_errors)*0.9),
             xytext=(10, max(alignment_errors)*0.7),
             arrowprops=dict(arrowstyle="->"))
plt.annotate('High Compression: Diminishing returns', 
             xy=(80, max(alignment_errors)*0.8),
             xytext=(60, max(alignment_errors)*0.6),
             arrowprops=dict(arrowstyle="->"))

plt.tight_layout()
plt.savefig('crf_simple_results.png', dpi=300)


print(f"\n=== RESULTS ===")
print(f"Best compression: {best_ratio*100:.0f}%")
print(f"Minimum error:    {best_error:.6f}")
print(f"Total systems:    {num_systems}")
print(f"Compute time:     {time.time()-start_time:.2f} seconds")
print(f"Results saved to crf_simple_results.png")
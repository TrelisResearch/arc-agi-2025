import matplotlib.pyplot as plt
import numpy as np

# Data for c3171 checkpoint
attempts = [1, 8, 256]
scores = [5.1, 15.1, 43.8]
std_devs = [0.8, 0.4, None]  # No std dev for 256 attempts (only 1 run)

# Convert to log base 2
log_attempts = np.log2(attempts)

# Create the plot
plt.figure(figsize=(10, 6))

# Plot with error bars where available
for i, (x, y, std) in enumerate(zip(log_attempts, scores, std_devs)):
    if std is not None:
        plt.errorbar(x, y, yerr=std, fmt='o', capsize=5, capthick=2, 
                    markersize=8, color='darkblue', ecolor='gray', alpha=0.8)
    else:
        plt.plot(x, y, 'o', markersize=8, color='darkblue', alpha=0.8)

# Connect points with a line
plt.plot(log_attempts, scores, '-', color='darkblue', alpha=0.5, linewidth=2)

# Add labels for each point
for x, y, attempts_val, score in zip(log_attempts, scores, attempts, scores):
    plt.annotate(f'{attempts_val} attempts\n{score}%', 
                xy=(x, y), 
                xytext=(10, 10), 
                textcoords='offset points',
                fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

# Formatting
plt.xlabel('Log₂(Number of Attempts)', fontsize=12)
plt.ylabel('Pass@2 Score (%)', fontsize=12)
plt.title('Scaling of Model Performance: Trelis/Qwen3-4B c3171 Checkpoint\n(ARC-AGI-1 Evaluation Set)', 
          fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, linestyle='--')

# Set x-axis ticks to show both log and actual values
ax = plt.gca()
ax.set_xticks(log_attempts)
ax.set_xticklabels([f'{int(2**x)}\n(2^{int(x)})' for x in log_attempts])

# Add note about pending 64 attempts
plt.text(0.02, 0.98, 'Note: 64 attempts data pending', 
         transform=ax.transAxes, 
         fontsize=10, 
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Set y-axis limits with some padding
plt.ylim(0, 50)

# Add a trend line (exponential fit in log space)
z = np.polyfit(log_attempts, scores, 1)
p = np.poly1d(z)
x_smooth = np.linspace(min(log_attempts), max(log_attempts), 100)
plt.plot(x_smooth, p(x_smooth), '--', color='red', alpha=0.5, 
         label=f'Trend: y = {z[0]:.2f}x + {z[1]:.2f}')

plt.legend(loc='lower right')

plt.tight_layout()
plt.savefig('/Users/ronanmcgovern/TR/arc-agi-2025/llm_python/experimental/checkpoint_c3171_scaling.png', 
            dpi=300, bbox_inches='tight')
plt.show()

print("Plot saved to experimental/checkpoint_c3171_scaling.png")
print(f"\nScaling results for c3171 checkpoint:")
print(f"1 attempt:   {scores[0]}% ± {std_devs[0]}%")
print(f"8 attempts:  {scores[1]}% ± {std_devs[1]}%") 
print(f"256 attempts: {scores[2]}% (single run)")
print(f"64 attempts: Data pending...")
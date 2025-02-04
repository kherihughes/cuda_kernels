import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# Data for Q1
# Update with latest benchmark data
q1_data = {
    "K (millions)": [1, 5, 10, 50, 100],
    "Time (CPU only)": [0.00186641, 0.00809323, 0.0201013, 0.0862944, 0.171453]
}

q2_data = {
    "K (millions)": [1, 5, 10, 50, 100],
    "Time (1 block, 1 thread)": [64.2683, 294.264, 592.683, 3001.25, 6023.8],
    "Time (1 block, 256 threads)": [1.23597, 5.93408, 11.4565, 58.2943, 116.725],
    "Time (multi-blocks, 256 threads)": [0.0256, 0.09728, 0.183296, 0.88576, 1.75206]
}

q3_data = {
    "K (millions)": [1, 5, 10, 50, 100],
    "Time (1 block, 1 thread)": [217.324, 1089.12, 2177.61, 10902.2, 21802.5],
    "Time (1 block, 256 threads)": [6.29248, 29.3622, 58.7049, 295.064, 592.132],
    "Time (multi-blocks, 256 threads)": [2.52723, 12.2501, 24.4132, 121.703, 243.837]
}

# Creating tables for Q1, Q2, Q3
q1_df = pd.DataFrame(q1_data)
q2_df = pd.DataFrame(q2_data)
q3_df = pd.DataFrame(q3_data)

# Plotting for Q4
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
k_values = q2_data["K (millions)"]

# Without Unified Memory plot
ax[0].plot(k_values, q1_data["Time (CPU only)"], label="CPU only", marker='o')
ax[0].plot(k_values, q2_data["Time (1 block, 1 thread)"], label="1 block, 1 thread", marker='s')
ax[0].plot(k_values, q2_data["Time (1 block, 256 threads)"], label="1 block, 256 threads", marker='^')
ax[0].plot(k_values, q2_data["Time (multi-blocks, 256 threads)"], label="Multi-blocks, 256 threads", marker='x')
ax[0].set_title("Q4 - Without Unified Memory", fontsize=14)
ax[0].set_xlabel("K (millions)", fontsize=12)
ax[0].set_ylabel("Time (ms)", fontsize=12)
ax[0].set_xscale("log")
ax[0].set_yscale("log")
ax[0].grid(True, which='both', linestyle='--', linewidth=0.5)
ax[0].legend()

# With Unified Memory plot
ax[1].plot(k_values, q1_data["Time (CPU only)"], label="CPU only", marker='o')
ax[1].plot(k_values, q3_data["Time (1 block, 1 thread)"], label="1 block, 1 thread", marker='s')
ax[1].plot(k_values, q3_data["Time (1 block, 256 threads)"], label="1 block, 256 threads", marker='^')
ax[1].plot(k_values, q3_data["Time (multi-blocks, 256 threads)"], label="Multi-blocks, 256 threads", marker='x')
ax[1].set_title("Q4 - With Unified Memory", fontsize=14)
ax[1].set_xlabel("K (millions)", fontsize=12)
ax[1].set_ylabel("Time (ms)", fontsize=12)
ax[1].set_xscale("log")
ax[1].set_yscale("log")
ax[1].grid(True, which='both', linestyle='--', linewidth=0.5)
ax[1].legend()

plt.tight_layout()
plt.show()

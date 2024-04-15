import matplotlib.pyplot as plt

# Sample accuracy values for each configuration across 5 epochs
# Please replace these sample values with your actual accuracies
acc_standard_config = [0.991, 0.996, 0.998, 0.999, 0.996]  # Standard configuration
acc_noise_level_10 = [0.822, 0.847, 0.851, 0.775, 0.743]      # Noise addition with noise level of 10%
acc_noise_level_20 = [0.419, 0.494, 0.594, 0.687, 0.836]     # Noise addition with noise level of 20%
acc_noise_rate_10 = [0.928, 0.970, 0.967, 0.992, 0.986]  # Label noise addition with 10 percent noise rate
acc_noise_rate_20 = [0.739, 0.558, 0.799, 0.790, 0.789]     # Label noise addition with 20 percent noise rate
acc_combined = [0.549, 0.396, 0.456, 0.564, 0.566]  # Combined noise addition

# Sample loss values for each configuration across 5 epochs
# Please replace these sample values with your actual losses
loss_standard_config = [0.839, 0.837,0.833, 0.833, 0.836]   # Standard configuration
loss_noise_level_10 = [0.861, 0.846, 0.836, 0.841, 0.832]
loss_noise_level_20 = [0.857, 0.852, 0.846, 0.846, 0.842]
loss_noise_rate_10 = [0.943, 0.937, 0.933, 0.925, 0.933]
loss_noise_rate_20 = [0.999, 0.991, 0.993, 0.987, 0.992]
loss_combined = [0.935, 0.932, 0.922, 0.921, 0.927]

# Epochs (1-5)
epochs = list(range(1, 6))

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(epochs, acc_standard_config, marker='o', linestyle='-', label='Standard Configuration')
plt.plot(epochs, acc_noise_level_10, marker='x', linestyle='-', label='Noise addition with noise level of 10%')
plt.plot(epochs, acc_noise_level_20, marker='^', linestyle='-', label='Noise addition with noise level of 20%')
plt.plot(epochs, acc_noise_rate_10, marker='s', linestyle='-', label='Label noise addition with 10% noise rate')
plt.plot(epochs, acc_noise_rate_20, marker='^', linestyle='-', label='Label noise addition with 20% noise rate')
plt.plot(epochs, acc_combined, marker='x', linestyle='-', label='Combined noise addition')

# Setting x-axis ticks to show only whole numbers
plt.xticks(epochs)

# Adding labels and title
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model Accuracy with Dataset Alterations')
plt.legend()
plt.grid(True)

# Display the plot
plt.show()




# Epochs (1-5)
epochs = list(range(1, 6))

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(epochs, loss_standard_config, marker='o', linestyle='-', label='Standard Configuration')
plt.plot(epochs, loss_noise_level_10, marker='s', linestyle='-', label='Noise addition with noise level of 10%')
plt.plot(epochs, loss_noise_level_20, marker='^', linestyle='-', label='Noise addition with noise level of 20%')
plt.plot(epochs, loss_noise_rate_10, marker='x', linestyle='-', label='Label noise addition with 10% noise rate')
plt.plot(epochs, loss_noise_rate_20, marker='^', linestyle='-', label='Label noise addition with 20% noise rate')
plt.plot(epochs, loss_combined, marker='x', linestyle='-', label='Combined noise addition')

# Setting x-axis ticks to show only whole numbers
plt.xticks(epochs)

# Adding labels and title
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Model Loss with Dataset Alterations')
plt.legend()
plt.grid(True)

# Display the plot
plt.show()


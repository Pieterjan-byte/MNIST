import matplotlib.pyplot as plt

# Sample accuracy values for each configuration across 5 epochs
# Replace these sample values with your actual data
accuracy_baseline = [0.997, 0.992, 0.996, 0.991, 0.981]  # Baseline
accuracy_input_noise = [0.807, 0.905, 0.844, 0.927, 0.907]  # Input Noise Addition
accuracy_label_noise = [0.891, 0.993, 0.998, 0.870, 0.967]  # Label Noise Addition
accuracy_combined_noise = [0.752, 0.658, 0.685, 0.639, 0.517]  # Combined Noise Addition

# Epochs (1-5)
epochs = list(range(1, 6))

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(epochs, accuracy_baseline, marker='o', linestyle='-', label='Baseline')
plt.plot(epochs, accuracy_input_noise, marker='s', linestyle='-', label='Input Noise')
plt.plot(epochs, accuracy_label_noise, marker='^', linestyle='-', label='Label Noise')
plt.plot(epochs, accuracy_combined_noise, marker='x', linestyle='-', label='Combined Noise')

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


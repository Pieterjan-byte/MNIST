import matplotlib.pyplot as plt

# Epochs (1-5)
epochs = list(range(1, 6))

# Sample loss values for each configuration across 5 epochs
# Replace these sample values with your actual data
loss_baseline = [0.470, 0.455, 0.458, 0.455, 0.459]  # Baseline
loss_input_noise = [0.469, 0.470, 0.465, 0.467, 0.458]  # Input Noise Addition
loss_label_noise = [0.530, 0.519, 0.521, 0.518, 0.518]  # Label Noise Addition
loss_combined_noise = [0.520, 0.517, 0.521, 0.511, 0.522]  # Combined Noise Addition

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(epochs, loss_baseline, marker='o', linestyle='-', label='Baseline')
plt.plot(epochs, loss_input_noise, marker='s', linestyle='-', label='Input Noise')
plt.plot(epochs, loss_label_noise, marker='^', linestyle='-', label='Label Noise')
plt.plot(epochs, loss_combined_noise, marker='x', linestyle='-', label='Combined Noise')

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


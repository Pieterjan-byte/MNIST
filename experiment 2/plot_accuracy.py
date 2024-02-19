import matplotlib.pyplot as plt

# Sample accuracy values for each configuration across 5 epochs
# Please replace these sample values with your actual accuracies
acc_standard_config = [0.991, 0.952, 0.999, 0.996, 0.999]  # Standard configuration
acc_n_classes_4 = [0.892, 0.920, 0.894, 0.912, 0.942]      # n_classes = 4
acc_n_addition_3 = [0.987, 0.995, 0.997, 0.950, 0.999]     # n_addition = 3
acc_classes_digits_4_3 = [0.741, 0.904, 0.888, 0.893, 0.914]  # n_classes = 4, n_addition = 3

# Epochs (1-5)
epochs = list(range(1, 6))

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(epochs, acc_standard_config, marker='o', linestyle='-', label='Standard Configuration')
plt.plot(epochs, acc_n_classes_4, marker='s', linestyle='-', label='n_classes = 4')
plt.plot(epochs, acc_n_addition_3, marker='^', linestyle='-', label='n_addition = 3')
plt.plot(epochs, acc_classes_digits_4_3, marker='x', linestyle='-', label='n_classes = 4, n_addition = 3')

# Setting x-axis ticks to show only whole numbers
plt.xticks(epochs)

# Adding labels and title
plt.xlabel('Epochs')
plt.ylabel('Test Accuracy')
plt.title('Accuracy Evolution During Training Across Configurations')
plt.legend()
plt.grid(True)

# Display the plot
plt.show()


# Sample loss values for each configuration across 5 epochs
# Please replace these sample values with your actual losses
loss_standard_config = [0.472, 0.465, 0.456, 0.461, 0.462]   # Standard configuration
loss_n_classes_4 = [1.100, 1.035, 1.023, 1.022, 1.013]     # n_classes = 4
loss_n_addition_3 = [0.635, 0.625, 0.633, 0.625, 0.624]      # n_addition = 3
loss_classes_digits_4_3 = [1.379, 1.230, 1.212, 1.206, 1.194]  # n_classes = 4, n_addition = 3

# Epochs (1-5)
epochs = list(range(1, 6))

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(epochs, loss_standard_config, marker='o', linestyle='-', label='Standard Configuration')
plt.plot(epochs, loss_n_classes_4, marker='s', linestyle='-', label='n_classes = 4')
plt.plot(epochs, loss_n_addition_3, marker='^', linestyle='-', label='n_addition = 3')
plt.plot(epochs, loss_classes_digits_4_3, marker='x', linestyle='-', label='n_classes = 4, n_addition = 3')

# Setting x-axis ticks to show only whole numbers
plt.xticks(epochs)

# Adding labels and title
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Evolution During Training Across Configurations')
plt.legend()
plt.grid(True)

# Display the plot
plt.show()


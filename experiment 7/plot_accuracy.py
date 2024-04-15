import matplotlib.pyplot as plt

# Sample accuracy values for each configuration across 5 epochs
# Please replace these sample values with your actual accuracies
acc_standard_config = [0.991, 0.996, 0.998, 0.999, 0.996]  # Standard configuration
acc_2_hidden_layer = [0.822, 0.847, 0.851, 0.775, 0.743]
acc_3_hidden_layer = [0.419, 0.494, 0.594, 0.687, 0.836]


# Sample loss values for each configuration across 5 epochs
# Please replace these sample values with your actual losses
loss_standard_config = [0.839, 0.837,0.833, 0.833, 0.836]   # Standard configuration
loss_2_hidden_layer = [0.861, 0.846, 0.836, 0.841, 0.832]
loss_3_hidden_layer = [0.857, 0.852, 0.846, 0.846, 0.842]


# Epochs (1-5)
epochs = list(range(1, 6))

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(epochs, acc_standard_config, marker='o', linestyle='-', label='Standard Configuration')
plt.plot(epochs, acc_2_hidden_layer, marker='x', linestyle='-', label='2 hidden layers')
plt.plot(epochs, acc_3_hidden_layer, marker='^', linestyle='-', label='3 hidden layers')


# Setting x-axis ticks to show only whole numbers
plt.xticks(epochs)

# Adding labels and title
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend()
plt.grid(True)

# Display the plot
plt.show()




# Epochs (1-5)
epochs = list(range(1, 6))

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(epochs, loss_standard_config, marker='o', linestyle='-', label='Standard Configuration')
plt.plot(epochs, loss_2_hidden_layer, marker='s', linestyle='-', label='2 hidden layers')
plt.plot(epochs, loss_3_hidden_layer, marker='^', linestyle='-', label='3 hidden layers')


# Setting x-axis ticks to show only whole numbers
plt.xticks(epochs)

# Adding labels and title
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend()
plt.grid(True)

# Display the plot
plt.show()


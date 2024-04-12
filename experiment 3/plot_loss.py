import matplotlib.pyplot as plt

# Epochs (1-3)
epochs = list(range(1, 6))

# Sample loss values for each semantic across 3 epochs
# Replace these values with your actual data
loss_sum_product = [1.166, 1.086, 1.076, 1.069, 1.060]  # SumProductSemiring
loss_lukasievicz = [100.0, 100.0, 100.0, 100.0, 100.0]  # LukasieviczTNorm
loss_godel = [1.345, 1.320, 1.314, 1.316, 1.316]        # GodelTNorm
loss_product = [1.369, 1.296, 1.284, 1.280, 1.285]      # ProductTNorm

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(epochs, loss_sum_product, marker='o', linestyle='-', label='SumProductSemiring')
plt.plot(epochs, loss_lukasievicz, marker='s', linestyle='-', label='LukasieviczTNorm')
plt.plot(epochs, loss_godel, marker='^', linestyle='-', label='GodelTNorm')
plt.plot(epochs, loss_product, marker='x', linestyle='-', label='ProductTNorm')

# Setting x-axis ticks to show only whole numbers
plt.xticks(epochs)

# Adding labels and title
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Comparison Across Semantics')
plt.legend()
plt.grid(True)

# Display the plot
plt.show()

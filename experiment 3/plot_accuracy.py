import matplotlib.pyplot as plt

# Sample accuracy values for each semantic across 3 epochs
# Replace these values with your actual data
accuracy_sum_product = [ 0 , 0.90, 0.92, 0.90, 0.92]  # SumProductSemiring
accuracy_lukasievicz = [0.032, 0.032, 0.032, 0.032, 0.032]  # LukasieviczTNorm
accuracy_godel = [0.281, 0.322, 0.340, 0.290, 0.379]        # GodelTNorm
accuracy_product = [0.529, 0.562, 0.610, 0.605, 0.615]      # ProductTNorm

# Epochs (1-3)
epochs = list(range(1, 4))

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(epochs, accuracy_sum_product, marker='o', linestyle='-', label='SumProductSemiring')
plt.plot(epochs, accuracy_lukasievicz, marker='s', linestyle='-', label='LukasieviczTNorm')
plt.plot(epochs, accuracy_godel, marker='^', linestyle='-', label='GodelTNorm')
plt.plot(epochs, accuracy_product, marker='x', linestyle='-', label='ProductTNorm')

# Adding labels and title
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison Across Semantics')
plt.legend()
plt.grid(True)

# Display the plot
plt.show()

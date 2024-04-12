import matplotlib.pyplot as plt

def plot_digitscale_step():
    # Read the log file and extract relevant information
    with open('digitsscale_2.log', 'r') as file:
        lines = file.readlines()

    # Extracting elapsed times from log entries
    elapsed_times = [float(line.split(' ')[-4]) for line in lines if 'The function took' in line]
    total_time = sum(elapsed_times) # total time for one setting (training + validation)
    print(total_time)

    # Create a plot
    plt.plot(elapsed_times, marker='o', linestyle='-')
    plt.xlabel('Function Calls')
    plt.ylabel('Elapsed Time (seconds)')
    plt.title('Execution Time of the Function Over Multiple Calls')
    plt.show()

def plot_digitscale_total():
    # Your list of values
    y_values = [0, 0, 34.1894, 56.5348, 849.1140, 5000]

    # Generate x values starting from zero with the same length as y_values
    x_values = list(range(len(y_values)))

    # Plotting the data
    plt.plot(x_values, y_values, linestyle='-')

    # Setting the range of the y-axis
    plt.ylim(0, 1500)

    plt.xticks(range(len(y_values)))

    plt.grid(True)

    # Adding labels and title
    plt.xlabel('Number of digits')
    plt.ylabel('Time (s)')
    plt.title('Scaling of MNISTAddition task with increasing digits')

    # Display the plot
    plt.show()

#plot_digitscale_step
plot_digitscale_total()
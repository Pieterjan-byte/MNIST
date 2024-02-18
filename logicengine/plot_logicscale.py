import matplotlib.pyplot as plt

def plot_logicscale_step():
    # Read the log file and extract relevant information
    with open('logicscale_11.log', 'r') as file:
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

def plot_logicscale_total():
    # Your list of values
    y_values = [0, 0.2376, 3.8531, 10.7715, 22.5603, 51.0763, 137.3049, 203.6008, 369.4450, 572.9217, 858.1767]

    # Generate x values starting from zero with the same length as y_values
    x_values = list(range(len(y_values)))

    # Plotting the data
    plt.plot(x_values, y_values, linestyle='-')

    # Adding labels and title
    plt.xlabel('Number of classes')
    plt.ylabel('Time (s)')
    plt.title('Scaling of logic engine')

    #plt.xticks(range(len(y_values)))

    plt.grid(True)

    # Display the plot
    plt.show()

#plot_logicscale_step()
plot_logicscale_total()
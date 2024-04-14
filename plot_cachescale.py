import matplotlib.pyplot as plt

def plot_logicscale_step():
    # Read the log file and extract relevant information
    with open('t_double_cache.log', 'r') as file:
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
    x_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y_data = [0, 44.5829, 78.4911, 144.6647, 248.8494, 401.0412, 625.6398, 906.5131, 1283.7103, 1865.9296]
    yy_data = [0, 28.6888, 48.7352, 78.9612, 121.7147, 161.0464, 225.8019, 312.2290, 414.1815, 539.4101]
    yyy_data = [0, 24.5941, 40.6174, 69.6882, 112.9669, 162.3049, 217.3319, 299.0830, 380.0930, 480.8399] 

    # Generate x values starting from zero with the same length as y_values
    # x_values = list(range(len(y_values)))

    # Plotting the data
    plt.plot(x_data, y_data, linestyle='-', label="No cache", color="blue")
    plt.plot(x_data, yy_data, linestyle='-', label="First Cache", color="red")
    plt.plot(x_data, yyy_data, linestyle='-', label="Second Cache", color="green")

    # Adding labels and title
    plt.xlabel('Number of classes')
    plt.ylabel('Time (s)')
    # plt.title('Utilization of cache in logic engine')

    #plt.xticks(range(len(y_values)))

    plt.grid(True)
    plt.legend()

    # Display the plot
    plt.show()

# plot_logicscale_step()
plot_logicscale_total()
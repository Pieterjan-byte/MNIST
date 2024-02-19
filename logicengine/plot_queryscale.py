import matplotlib.pyplot as plt

def plot_logicscale_step():
    # Read the log file and extract relevant information
    with open('queryscale_2.log', 'r') as file:
        lines = file.readlines()

    # Extracting elapsed times from log entries
    elapsed_queries = [float(line.split(' ')[-4]) for line in lines if 'The function took' in line]
    total_query = sum(elapsed_queries) # total time for one setting (training + validation)
    print(total_query)

    # Create a plot
    plt.plot(elapsed_queries, marker='o', linestyle='-')
    plt.xlabel('Function Calls')
    plt.ylabel('Elapsed Time (seconds)')
    plt.title('Execution Time of the Function Over Multiple Calls')
    plt.show()

def plot_logicscale_total():
    # Your list of values
    y_values = [0, 3579, 9887, 17816, 27819, 39571, 52581, 68461, 86340, 105289, 127432]

    amount_examples = [0, 1480, 3166, 4655, 6189, 7649, 9004, 10548, 12050, 13901, 15000]

    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    amount_substitutions = [0, 16792, 278464, 457704, 1074560, 2069200, 3498336, 4898392, 8287232, 11694094, 16102400]

    # Generate x values starting from zero with the same length as y_values
    x_values = list(range(len(y_values)))

    # Plotting the data
    #plt.plot(x_values, y_values, linestyle='-', color="blue", label="Queries")
    #plt.plot(x_values, amount_examples, linestyle='-', color="red", label="Training examples")
    plt.plot(x, amount_substitutions, linestyle="-", label="Substitutions")

    # Adding labels and title
    plt.xlabel('Number of classes')
    plt.ylabel('Total amount')
    plt.title('Scaling of logic engine')

    #plt.xticks(range(len(y_values)))

    plt.grid(True)
    plt.legend()

    # Display the plot
    plt.show()

plot_logicscale_step()
#plot_logicscale_total()
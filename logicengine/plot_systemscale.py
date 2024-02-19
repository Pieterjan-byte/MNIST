import matplotlib.pyplot as plt

def plot_logicscale_step():
    # Read the log file and extract relevant information
    with open('add_queryscale_4.log', 'r') as file:
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
    queries_examples = [0, 0, 9887, 7553, 6446, 5218]

    amount_examples = [0, 0, 3166, 2111, 1583, 1266]  

    x = [0, 1, 2, 3, 4, 5]
    amount_substitutions = [0, 0, 139232, 1271808, 18341888, 289567390]

    # Generate x values starting from zero with the same length as y_values
    x_values = list(range(len(queries_examples)))

    # Plotting the data
    #plt.plot(x_values, queries_examples, linestyle='-', color="blue", label="Queries")
    #plt.plot(x_values, amount_examples, linestyle='-', color="red", label="Training examples")
    plt.plot(x, amount_substitutions, linestyle="-", label="Substitutions")

    # Adding labels and title
    plt.xlabel('Number of digits')
    plt.ylabel('Total amount')
    plt.title('Scaling of logic engine')

    plt.xticks(range(len(queries_examples)))

    plt.xlim(2, max(x_values))
    #plt.ylim(0, 20000000)

    plt.grid(True)
    plt.legend()

    # Display the plot
    plt.show()

#plot_logicscale_step()
plot_logicscale_total()
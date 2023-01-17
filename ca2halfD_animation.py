import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def animate_heat_map(filename, data, num_grids, num_iterations):
    fig = plt.figure()
    # ax = sns.heatmap(data[0], vmin=0, vmax=3)

    def init():
        plt.clf()

    def animate(i, data):
        plt.clf()
        plt.title(f'Grid no. {int(np.floor(i/num_iterations))} iteration {i%num_iterations}')
        sns.heatmap(data[i], vmin=0, vmax=3)

    print("animating")
    anim = animation.FuncAnimation(fig, animate, fargs=[data], init_func=init, interval=1, frames=len(data), repeat=False)
    print("writing gif")
    writer = animation.PillowWriter(fps=8)
    anim.save(filename, writer=writer)


# Define a function that maps values to the desired range
def map_values(x: float) -> int:
    return min(max(int(x), 0), 3)


if __name__ == "__main__":

    n = 30  # Set the size of the grid
    num_iterations = 6
    num_grids = 50
    # Set the range of possible weights and biases
    weight_range = (-2, 2)
    bias_range = (0, 0)

    fig = plt.figure()
    data = []
    # Generate the specified number of grids
    for i in range(num_grids):
        print(f'Grid no. {i}')
        weights = np.random.uniform(weight_range[0], weight_range[1], size=(7,7))
        bias = np.random.uniform(bias_range[0], bias_range[1])

        # Initialize the grid with random values
        # grid = np.random.randint(2, size=(n,n), dtype=int)
        grid = np.zeros((n,n), dtype=int)
        # for p in range(1):
        #     randx = np.random.randint(n)
        #     randy = np.random.randint(n)
        #     grid[randx,randy] = 1
        grid[int(np.ceil(n/2)),int(np.ceil(n/2))] = 1
        data.append(grid)
        # Set random weights and biases
        # weights = np.random.uniform(factor*weight_range[0], factor*weight_range[1], size=(7,7))
        # bias = np.random.uniform(factor*bias_range[0], factor*bias_range[1])
        for j in range(num_iterations):
            # Create an empty grid to store the new values
            new_grid = np.zeros((n,n))
            # Iterate over each cell in the grid
            for k in range(3,n-3):
                for ll in range(3,n-3):
                    # Apply the NCA rule to update the cell
                    new_grid[k,ll] = np.sum(weights * grid[k-3:k+4,ll-3:ll+4]) + bias
            # Map the values in the grid to the desired range
            grid = np.vectorize(map_values)(new_grid)
            data.append(grid)
    animate_heat_map(f'test_.gif', data, num_grids, num_iterations)

import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def animate_heat_map(filename, data, j):
    fig = plt.figure()
    # ax = sns.heatmap(data[0], vmin=0, vmax=3)

    def init():
        plt.clf()

    def animate(i, j, data):
        plt.clf()
        plt.title(f'Grid no. {j} iteration {i}')
        ax = sns.heatmap(data[i], vmin=0, vmax=3)

    anim = animation.FuncAnimation(fig, animate, fargs=(j, data), init_func=init, interval=1, frames=10, repeat=False)
    writer = animation.PillowWriter(fps=4)
    anim.save(filename, writer=writer)

# Define a function that maps values to the desired range
def map_values(x: float) -> int:
    return min(max(int(x), 0), 3)
        

if __name__ == "__main__":

    # Set the size of the grid
    n = 10

    # Set the number of iterations
    num_iterations = 10

    # Set the number of grids to generate
    num_grids = 2

    # Set the range of possible weights and biases
    weight_range = (-3, 3)
    bias_range = (3, 3)


    # Generate the specified number of grids
    fig = plt.figure()
    for i in range(num_grids):
        print(f'Grid no. {i}')

        # Initialize the grid with random values
        # grid = np.random.randint(2, size=(n,n), dtype=int)
        grid = np.zeros((n,n), dtype=int)
        # for p in range(5):
        #     randx = np.random.randint(n)
        #     randy = np.random.randint(n)
        #     grid[randx,randy] = 1
        grid[int(np.ceil(n/2)),int(np.ceil(n/2))] = 1  
        # Set random weights and biases
        weights = np.random.uniform(weight_range[0], weight_range[1], size=(7,7))
        bias = np.random.uniform(bias_range[0], bias_range[1])

        data = [grid]
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

        animate_heat_map("test.gif", data, i)
    
        

        # # Iterate over the number of iterations
        # for j in range(num_iterations):
        #     # print(grid)
        #     if im is None:
        #         im = plt.imshow(grid, cmap='hot', interpolation='nearest', vmin=0, vmax=3)
        #     else:
        #         im.set_data(grid)
        #         im.set_cmap('hot')
        #     plt.title(f'Grid no. {i} iteration {j} max val {np.max(grid)}')
        #     plt.pause(0.00001)

        #     # print(j)
        #     # Create an empty grid to store the new values
        #     new_grid = np.zeros((n,n))
        #     # Iterate over each cell in the grid
        #     for k in range(3,n-3):
        #         for ll in range(3,n-3):
        #             # Apply the NCA rule to update the cell
        #             new_grid[k,ll] = np.sum(weights * grid[k-3:k+4,ll-3:ll+4]) + bias
        #     # Map the values in the grid to the desired range
        #     grid = np.vectorize(map_values)(new_grid)

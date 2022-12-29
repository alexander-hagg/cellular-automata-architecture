import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# Set the size of the grid
n = 30

# Set the number of iterations
num_iterations = 100

# Set the number of grids to generate
num_grids = 10

# Set the range of possible weights and biases
weight_range = (-1, 1)
bias_range = (-1, 1)


# Define a function that maps values to the desired range
def map_values(x: float) -> int:
    return min(max(int(x), 0), 3)


# Generate the specified number of grids
for i in range(num_grids):
    # Initialize the grid with random values
    # grid = np.random.randint(2, size=(n,n), dtype=int)
    grid = np.zeros((n,n), dtype=int)
    randx = np.random.randint(n)
    randy = np.random.randint(n)
    # grid[int(np.ceil(n/2)),int(np.ceil(n/2))] = 1
    grid[randx,randy] = 1
    # Set random weights and biases
    weights = np.random.uniform(weight_range[0], weight_range[1], size=(3,3))
    bias = np.random.uniform(bias_range[0], bias_range[1])
    # Iterate over the number of iterations
    for j in range(num_iterations):
        # Create an empty grid to store the new values
        new_grid = np.zeros((n,n))
        # Iterate over each cell in the grid
        for k in range(1,n-1):
            for ll in range(1,n-1):
                # Apply the NCA rule to update the cell
                new_grid[k,ll] = np.sum(weights * grid[k-1:k+2,ll-1:ll+2]) + bias
        # Map the values in the grid to the desired range
    grid = np.vectorize(map_values)(new_grid)
    # print(new_grid)
    fig = plt.figure()
    plt.imshow(grid, cmap='hot', interpolation='nearest')
    plt.show()

    # Visualize the grid as 3D voxels
    # ax = fig.add_subplot(111, projection='3d')
    # x, y = np.meshgrid(range(n), range(n))
    # filled = np.ones((n-1,n-1,max(grid.flatten())+1))
    # print(f'filled: {filled.shape}')
    # ax.voxels(x, y, grid, filled=filled, edgecolor='k')
    # print(grid)
    # ax.voxels(grid, edgecolor='k')
    # ax.set_aspect('equal', 'box')
    # plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Set the size of the grid
n = 30

# Set the number of iterations
num_iterations = 50

# Set the number of grids to generate
num_grids = 10

# Set the range of possible weights and biases
weight_range = (-1, 1)
bias_range = (-1, 1)


# Define a function that maps values to the desired range
def map_values(x: float) -> int:
    return min(max(int(x), 0), 3)


# Generate the specified number of grids
fig = plt.figure()
im = None
for i in range(num_grids):
    print(f'Grid no. {i}')
    # Initialize the grid with random values
    # grid = np.random.randint(2, size=(n,n), dtype=int)
    grid = np.zeros((n,n), dtype=int)
    for p in range(5):
        randx = np.random.randint(n)
        randy = np.random.randint(n)
        grid[randx,randy] = 1
    # grid[int(np.ceil(n/2)),int(np.ceil(n/2))] = 1
      
    # Set random weights and biases
    weights = np.random.uniform(weight_range[0], weight_range[1], size=(5,5))
    bias = np.random.uniform(bias_range[0], bias_range[1])
    # Iterate over the number of iterations
    for j in range(num_iterations):
        # print(grid)
        if im is None:
            im = plt.imshow(grid, cmap='hot', interpolation='nearest')
        else:
            im.set_data(grid)
            im.set_cmap('hot')
        plt.title(f'Grid no. {i} iteration {j}')
        plt.pause(0.0001)

        # print(j)
        # Create an empty grid to store the new values
        new_grid = np.zeros((n,n))
        # Iterate over each cell in the grid
        for k in range(2,n-2):
            for ll in range(2,n-2):
                # Apply the NCA rule to update the cell
                new_grid[k,ll] = np.sum(weights * grid[k-2:k+3,ll-2:ll+3]) + bias
        # Map the values in the grid to the desired range
        grid = np.vectorize(map_values)(new_grid)
# plt.show()

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

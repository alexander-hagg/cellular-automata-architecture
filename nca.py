import numpy as np
import matplotlib.pyplot as plt

# Set the size of the grid
n = 100

# Set the number of iterations
num_iterations = 10

# Set the number of grids to generate
num_grids = 5

# Set the range of possible weights and biases
weight_range = (-1, 1)
bias_range = (-1, 1)

# Define a function that maps values to the desired range
def map_values(x):
  return min(max(int(x), 0), 3)

# Generate the specified number of grids
for i in range(num_grids):
  # Initialize the grid with random values
  grid = np.random.randint(2, size=(n,n))
  # Set random weights and biases
  weights = np.random.uniform(weight_range[0], weight_range[1], size=(3,3))
  bias = np.random.uniform(bias_range[0], bias_range[1])
  # Iterate over the number of iterations
  for j in range(num_iterations):
    # Create an empty grid to store the new values
    new_grid = np.empty((n,n))
    # Iterate over each cell in the grid
    for k in range(1,n-1):
      for l in range(1,n-1):
        # Apply the NCA rule to update the cell
        new_grid[k,l] = np.sum(weights * grid[k-1:k+2,l-1:l+2]) + bias
    # Map the values in the grid to the desired range
    grid = np.vectorize(map_values)(new_grid)
  # Plot the resulting grid
  plt.imshow(grid, cmap='binary')
  plt.show()

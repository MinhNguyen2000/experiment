import numpy as np
import matplotlib.pyplot as plt
from frozen_lake_plot import plot_grid

# Initialize environment parameters
grid_size = 4
holes = [(1, 1), (1, 3), (2, 3), (3, 0)]  # Bad states
goal = (3, 3)  # Terminal state
start = (0, 0)

# Example 2: Random values for demonstration
V_rand = {(i,j): np.random.rand() for i in range(grid_size) for j in range(grid_size)}
Q_rand = {(i,j): {a: np.random.rand() for a in ['up','down','left','right']} 
           for i in range(grid_size) for j in range(grid_size)}

plot_grid(V_rand, Q_rand, "Random Initialized Values")
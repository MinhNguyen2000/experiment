import numpy as np
import matplotlib.pyplot as plt

# Initialize environment parameters
grid_size = 4
holes = [(1, 1), (1, 3), (2, 3), (3, 0)]  # Bad states
goal = (3, 3)  # Terminal state
start = (0, 0)

# Initialize value tables
V = {(i, j): 0 for i in range(grid_size) for j in range(grid_size)}
Q = {(i, j): {'up': 0, 'down': 0, 'left': 0, 'right': 0} 
      for i in range(grid_size) for j in range(grid_size)}

def plot_grid(V, Q, title="Frozen Lake"):
    fig, ax = plt.subplots()
    ax.set_xticks(np.arange(grid_size+1))
    ax.set_yticks(np.arange(grid_size+1))
    ax.grid(True)
    ax.invert_yaxis()  # To match matrix coordinates (0,0 at top-left)
    ax.set_aspect('equal')

    # Draw cells
    for row in range(grid_size):
        for col in range(grid_size):
            # Background color
            if (row, col) == goal:
                bg_color = 'lightgreen'
            elif (row, col) in holes:
                bg_color = 'salmon'
            elif (row, col) == start:
                bg_color = 'lightblue'
            else:
                bg_color = 'white'
            
            ax.add_patch(plt.Rectangle((col, row), 1, 1, facecolor=bg_color, edgecolor='black'))
            
            # Text: State value (center) and Q-values (directions)
            ax.text(col + 0.5, row + 0.5, f"{V[(row, col)]:.2f}", 
                    ha='center', va='center')
            ax.text(col + 0.05, row + 0.5, f"{Q[(row, col)]['left']:.2f}", fontsize=8, rotation=-90, ha='left', va='center')
            ax.text(col + 0.95, row + 0.5, f"{Q[(row, col)]['right']:.2f}", fontsize=8, rotation=90, ha='right', va='center')
            ax.text(col + 0.5, row + 0.2, f"{Q[(row, col)]['up']:.2f}", fontsize=8, ha='center', va='center')
            ax.text(col + 0.5, row + 0.8, f"{Q[(row, col)]['down']:.2f}", fontsize=8, ha='center', va='center')


    plt.title(title)
    plt.show()

# Example 1: Show initialized values (all zeros)
# plot_grid(V, Q, "Initial State Values")

# Example 2: Random values for demonstration
V_rand = {(i,j): np.random.rand() for i in range(4) for j in range(4)}
Q_rand = {(i,j): {a: np.random.rand() for a in ['up','down','left','right']} 
           for i in range(4) for j in range(4)}
plot_grid(V_rand, Q_rand, "Random Initialized Values")
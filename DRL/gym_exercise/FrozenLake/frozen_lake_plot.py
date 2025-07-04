import numpy as np
import matplotlib.pyplot as plt

# Initialize environment parameters
grid_size = 4
n_states = 16
n_actions = 4
holes = [(1, 1), (1, 3), (2, 3), (3, 0)]  # Bad states
goal = (3, 3)  # Terminal state
start = (0, 0)


def plot_grid(grid_size, Q, title="Frozen Lake"):
    ''' Function to plot a simple square GridWorld with the state-action values in each square 
    
    Inputs: 
    - grid_size: the number of square in each edge
    - Q: the array of state-action values, where the rows represent the action-space and the columns represents the state-space.
         the array has size (n_actions x n_states)
    '''
    fig, ax = plt.subplots()
    fig_size = 8
    font_size = 10
    fig.set_figheight(fig_size); fig.set_figwidth(fig_size)
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
            # ax.text(col + 0.5, row + 0.5, f"{V[(row, col)]:.2f}", ha='center', va='center')
            ax.text(col + 0.5, row + 0.2, f"{Q[3,row*grid_size+col]:.3f}", fontsize=font_size, ha='center', va='center')
            ax.text(col + 0.95, row + 0.5, f"{Q[2,row*grid_size+col]:.3f}", fontsize=font_size, rotation=90, ha='right', va='center')            
            ax.text(col + 0.5, row + 0.8, f"{Q[1,row*grid_size+col]:.3f}", fontsize=font_size, ha='center', va='center')
            ax.text(col + 0.05, row + 0.5, f"{Q[0,row*grid_size+col]:.3f}", fontsize=font_size, rotation=-90, ha='left', va='center')

            # Visualize greedy policy
            state = row * grid_size + col
            
            # Action directions (for arrows) - [up, right, down, left]
            arrow_directions = [(0, -0.05), (0.05, 0), (0, 0.05), (-0.05, 0)]
            arrow_directions.reverse()
            arrow_colors = ['red', 'red', 'red', 'red']

            # Greedy policy arrow (only if not terminal state)
            if (row, col) not in holes and (row, col) != goal:
                best_action = np.argmax(Q[:, state])
                dx, dy = arrow_directions[best_action]
                ax.arrow(col + 0.5, row + 0.5, dx, dy, 
                        head_width=0.2, head_length=0.2, 
                        fc=arrow_colors[best_action], ec=arrow_colors[best_action])

    plt.title(title)
    plt.show()

# Example 1: Show initialized values (all zeros)
# plot_grid(V, Q, "Initial State Values")

# Example 2: Random values for demonstration
# V_rand = {(i,j): np.random.rand() for i in range(4) for j in range(4)}
# Q_rand = {(i,j): {a: np.random.rand() for a in ['up','down','left','right']} 
#            for i in range(4) for j in range(4)}
# plot_grid(V_rand, Q_rand, "Random Initialized Values")
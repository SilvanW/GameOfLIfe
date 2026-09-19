import matplotlib.pyplot as plt
import numpy as np

from src.rules import apply_rules

"""
Rules
	- cell c is 0 (dead)
		- if 3 neighbours are 1 then => 1
	- call c is 1 (alive)
		- if 2 or 3 alive neighbours -> then 1 (no change)
		- else: -> 0 (dead)
"""

grid = np.zeros((50, 50), dtype=np.int8)
grid_copy = np.zeros_like(grid)

num_generations = 1


# TODO: fine a nice way to add those programatically
# Still life
grid[5, 5] = 1
grid[5, 6] = 1
grid[6, 5] = 1
grid[6, 6] = 1

# Flipper
grid[20, 20] = 1
grid[20, 21] = 1
grid[20, 22] = 1

plt.imshow(grid)
plt.title("Game of Life Initial State")
plt.show()

for generation in range(num_generations):
    apply_rules(grid, grid_copy)

    grid = grid_copy.copy()
    grid_copy[:, :] = 0

print(np.unique(grid))

plt.imshow(grid)
plt.title("Game of Life Result")
plt.show()

import matplotlib.pyplot as plt
import numpy as np

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


MOORE_CELL_OFFSETS: list[tuple[int, int]] = [
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
]

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
    for (j, i), value in np.ndenumerate(grid[1:-1, 1:-1]):
        # Required due to grid slicing in enumerate
        j += 1
        i += 1

        if value == 0:
            num_alive = 0
            for dy, dx in MOORE_CELL_OFFSETS:
                if grid[j + dy, i + dx] == 1:
                    num_alive += 1

            if num_alive == 3:
                grid_copy[j, i] = 1

        if value == 1:
            num_alive = 0
            for dy, dx in MOORE_CELL_OFFSETS:
                if grid[j + dy, i + dx] == 1:
                    num_alive += 1

            if num_alive == 2 or num_alive == 3:
                grid_copy[j, i] = 1
            else:
                grid_copy[j, i] = 0

    grid = grid_copy.copy()
    grid_copy[:, :] = 0

print(np.unique(grid))

plt.imshow(grid)
plt.title("Game of Life Result")
plt.show()

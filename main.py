import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from tqdm import tqdm

from src.rules import apply_rules, apply_rules_numba

"""
Rules
	- cell c is 0 (dead)
		- if 3 neighbours are 1 then => 1
	- call c is 1 (alive)
		- if 2 or 3 alive neighbours -> then 1 (no change)
		- else: -> 0 (dead)
"""

grid = np.zeros((500, 500), dtype=np.int8)
grid_copy = np.zeros_like(grid)

num_generations = 1000


# TODO: fine a nice way to add those programatically
# Still life
# grid[5, 5] = 1
# grid[5, 6] = 1
# grid[6, 5] = 1
# grid[6, 6] = 1

# Flipper
# grid[20, 20] = 1
# grid[20, 21] = 1
# grid[20, 22] = 1

# Acorn
grid[250, 250] = 1
grid[250, 251] = 1
grid[248, 251] = 1
grid[249, 253] = 1
grid[250, 254:257] = 1

plt.imshow(grid)
plt.title("Game of Life Initial State")
plt.show()

# Set up writer
fps = 10
writer = FFMpegWriter(fps=fps)

fig, ax = plt.subplots()
ax.set_title("Game of Life")
im = ax.imshow(grid)

# Create video
with writer.saving(fig, "output.mp4", dpi=600):
    writer.grab_frame()
    start = time.perf_counter()
    for generation in tqdm(range(num_generations)):
        # apply_rules(grid, grid_copy)
        apply_rules_numba(grid, grid_copy)

        grid = grid_copy.copy()
        grid_copy[:, :] = 0

        im.set_data(grid)  # Update the data
        writer.grab_frame()  # Capture the frame

    end = time.perf_counter()

print(f"Elapsed: {end - start:.6f} s")

plt.imshow(grid)
plt.title("Game of Life Result")
plt.show()

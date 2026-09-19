import numpy as np
from numba import jit

MOORE_CELL_OFFSETS: np.ndarray = np.array(
    [
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1),
        (-1, 0),
        (-1, 1),
    ],
    dtype=np.int8,
)


def apply_rules(grid: np.ndarray, grid_copy: np.ndarray) -> None:
    rows, cols = grid.shape
    for j in range(1, rows - 1):
        for i in range(1, cols - 1):
            value = grid[j, i]

            num_alive = 0
            for dy, dx in MOORE_CELL_OFFSETS:
                if grid[j + dy, i + dx] == 1:
                    num_alive += 1

            if value == 0 and num_alive == 3:
                grid_copy[j, i] = 1

            if value == 1:
                if num_alive == 2 or num_alive == 3:
                    grid_copy[j, i] = 1
                else:
                    grid_copy[j, i] = 0


@jit(nopython=True)
def apply_rules_numba(grid: np.ndarray, grid_copy: np.ndarray) -> None:
    rows, cols = grid.shape
    for j in range(1, rows - 1):
        for i in range(1, cols - 1):
            value = grid[j, i]

            num_alive = 0
            for dy, dx in MOORE_CELL_OFFSETS:
                if grid[j + dy, i + dx] == 1:
                    num_alive += 1

            if value == 0 and num_alive == 3:
                grid_copy[j, i] = 1

            if value == 1:
                if num_alive == 2 or num_alive == 3:
                    grid_copy[j, i] = 1
                else:
                    grid_copy[j, i] = 0

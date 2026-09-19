import numpy as np

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


def apply_rules(grid: np.ndarray, grid_copy: np.ndarray) -> None:
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

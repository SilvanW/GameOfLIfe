import numpy as np
import torch
import torch.nn.functional as F
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

            # Tolist avoids OverflowError due to int8 np array
            for dy, dx in MOORE_CELL_OFFSETS.tolist():
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


DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print(DEVICE)

conv_kernel: torch.Tensor = (
    torch.tensor([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=torch.float32)
    .view(
        1,
        1,
        3,
        3,  # 1 output channel, 1 input channel and 3x3 in shape
    )
    .to(DEVICE)
)


def apply_rules_conv(grid: torch.Tensor, grid_copy: torch.Tensor) -> None:

    # Padding ignores the most outer cells
    conv_result = F.conv2d(grid, conv_kernel, padding=1)

    alive = grid == 1
    new_grid = (
        (conv_result == 3) | (alive & ((conv_result == 2) | (conv_result == 3)))
    ).float()

    grid_copy[:] = new_grid

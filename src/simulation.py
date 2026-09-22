from enum import Enum
from typing import Protocol

import numpy as np
import torch
import torch.nn.functional as F
from numba import jit
from tqdm import tqdm

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


class RuleImplementation(Enum):
    PYTHON = "python"
    NUMBA = "numba"
    PYTORCH = "pytorch"


class RuleNumpy(Protocol):
    def __call__(self, grid: np.ndarray, grid_copy: np.ndarray) -> None: ...


class RuleTorch(Protocol):
    def __call__(self, grid: torch.Tensor, grid_copy: torch.Tensor) -> None: ...


RULE_IMPLEMENTATIONS: dict[RuleImplementation, RuleNumpy | RuleTorch] = {
    RuleImplementation.PYTHON: apply_rules,
    RuleImplementation.NUMBA: apply_rules_numba,
    RuleImplementation.PYTORCH: apply_rules_conv,
}


def get_rules(rule_implementation: RuleImplementation) -> RuleNumpy | RuleTorch:
    rules = RULE_IMPLEMENTATIONS.get(rule_implementation)

    if rules is None:
        raise NotImplementedError(f"Rule {rule_implementation} is not implemented.")

    return rules


def _simulate_numpy(
    grid: np.ndarray, rules: RuleNumpy, n_generations: int
) -> np.ndarray:
    grid_copy = np.zeros_like(grid)
    for _ in tqdm(range(n_generations)):
        rules(grid, grid_copy)

        grid = grid_copy.copy()
        grid_copy[:, :] = 0

    return grid


def _simulate_pytorch(
    grid: torch.Tensor, rules: RuleTorch, n_generations: int
) -> torch.Tensor:
    grid_copy = torch.zeros_like(grid)
    for _ in tqdm(range(n_generations)):
        rules(grid, grid_copy)
        grid.copy_(grid_copy)

    return grid


def simulate(
    rule_implementation: RuleImplementation, n_generations: int, grid: np.ndarray
) -> np.ndarray:
    """Simulate game of life for n_generations using the provided grid and the selected rule implementation.

    Args:
        rule_implementation (RuleImplementation): Implementation of the Rules to be used
        n_generations (int): Number of generations to simulated
        grid (np.ndarray): The grid to simulate off of

    Returns:
        np.ndarray: Resulting grid after n_generations
    """
    apply_rules = get_rules(rule_implementation)

    # Simulate with torch
    if rule_implementation == RuleImplementation.PYTORCH:
        grid_torch = torch.from_numpy(grid).float().unsqueeze(0).to(DEVICE)
        result = _simulate_pytorch(
            grid=grid_torch, rules=apply_rules, n_generations=n_generations
        )
        return result.squeeze(0).cpu().numpy()

    # Simulate with numpy
    result = _simulate_numpy(grid=grid, rules=apply_rules, n_generations=n_generations)
    return result

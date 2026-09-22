from enum import Enum

import numpy as np

STILL_LIFE = np.array([[1, 1], [1, 1]])

FLIPPER = np.array([[1], [1], [1]])

ACORN = np.array(
    [
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [1, 1, 0, 0, 1, 1, 1],
    ]
)


class Pattern(Enum):
    STILL_LIFE = "still_life"
    FLIPPER = "flipper"
    ACORN = "acorn"


PATTERNS: dict[Pattern, np.ndarray] = {
    Pattern.STILL_LIFE: STILL_LIFE,
    Pattern.FLIPPER: FLIPPER,
    Pattern.ACORN: ACORN,
}


def get_pattern(pattern: Pattern) -> np.ndarray:
    """Get the selected Pattern as a np array to be stamped into the grid

    Args:
        pattern (Pattern): The Pattern selection

    Raises:
        NotImplementedError: Pattern <pattern name> is not implemented

    Returns:
        np.ndarray: Selected Pattern
    """
    retrieved_pattern = PATTERNS.get(pattern)

    if retrieved_pattern is None:
        raise NotImplementedError(f"Pattern {pattern.value} is not implemented")

    return retrieved_pattern


def apply_pattern(pattern: np.ndarray, grid: np.ndarray) -> None:
    """Applies (stamps) the pattern in to the center of the grid

    Args:
        pattern (np.ndarray): The pattern to be applied into the grid
        grid (np.ndarray): The grid the pattern should be applied to

    Raises:
        ValueError: Only 2d arrays are supported
        ValueError: Pattern cannot be larger than grid
    """
    if pattern.ndim != 2 or grid.ndim != 2:
        raise ValueError("Only 2d arrays are supported")

    if pattern.shape[0] >= grid.shape[0] or pattern.shape[1] >= grid.shape[1]:
        raise ValueError("Pattern cannot be larger than grid")

    grid_center_y, grid_center_x = grid.shape[0] // 2, grid.shape[1] // 2

    pattern_height, pattern_width = pattern.shape

    pattern_start_y = grid_center_y - pattern_height // 2
    pattern_start_x = grid_center_x - pattern_width // 2

    grid[
        pattern_start_y : pattern_start_y + pattern_height,
        pattern_start_x : pattern_start_x + pattern_width,
    ] = pattern

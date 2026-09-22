# Game of Life

Game of Life Implementation for UZH ESC413.

## Getting Started

```shell
uv run main.py
```

## Implementations

Currently there are the three following implementations of the game of life rules.

- Python: Pure Python Loop over Numpy arrays
- Numba: Same as Python with additional numba `jit` decorator
- Pytorch: 2D Conv Kernel running on mps or cpu

The user is asked about which implementation should be used via the cli.

## Patterns

The patterns currently implemented in `pattern.py` are the following.

- Still Life
- Flipper
- Acorn

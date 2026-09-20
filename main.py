import time

import matplotlib.pyplot as plt
import numpy as np
import questionary
import torch

from src.simulation import (
    RuleImplementation,
    simulate,
)


def main():
    # TODO: let user define grid size
    grid: np.ndarray | torch.Tensor = np.zeros((500, 500), dtype=np.int8)

    rule_implementation: RuleImplementation = questionary.select(
        "Which rule implementation should be used for Simulation?",
        choices=[
            questionary.Choice(title=rule.value, value=rule)
            for rule in RuleImplementation
        ],
    ).ask()

    num_generations = int(
        questionary.text("How many generations should be simulated?").ask()
    )

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

    start = time.perf_counter()

    result = simulate(rule_implementation, num_generations, grid)

    end = time.perf_counter()

    print(f"Elapsed: {end - start:.6f} s")

    # TODO: somehow support video export as well (use yield and export thread)

    plt.imshow(result)
    plt.title("Game of Life Result")
    plt.show()


if __name__ == "__main__":
    main()

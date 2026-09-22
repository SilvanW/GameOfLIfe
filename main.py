import time

import matplotlib.pyplot as plt
import numpy as np
import questionary
import torch

from src.patterns import Pattern, apply_pattern, get_pattern
from src.simulation import (
    RuleImplementation,
    simulate,
)


def main():
    # TODO: let user define grid size
    grid: np.ndarray | torch.Tensor = np.zeros((50, 50), dtype=np.int8)

    rule_implementation: RuleImplementation = questionary.select(
        "Which rule implementation should be used for Simulation?",
        choices=[
            questionary.Choice(title=rule.value, value=rule)
            for rule in RuleImplementation
        ],
    ).ask()

    pattern_selection: Pattern = questionary.select(
        "Which pattern should be used for Simulation?",
        choices=[
            questionary.Choice(title=pattern.value, value=pattern)
            for pattern in Pattern
        ],
    ).ask()

    num_generations = int(
        questionary.text("How many generations should be simulated?").ask()
    )

    pattern = get_pattern(pattern_selection)

    apply_pattern(pattern, grid)

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

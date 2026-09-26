import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pydantic import BaseModel, PositiveFloat

from src.patterns import Pattern
from src.simulation import RuleImplementation


class Score(BaseModel):
    grid_dimension: int
    rule_implementation: RuleImplementation
    pattern: Pattern
    n_generations: int
    execution_time_seconds: PositiveFloat


def ensure_score_file(file_path: Path) -> None:
    if not os.path.exists(file_path):
        df = pd.DataFrame(
            columns=[
                "grid_dimension",
                "rule_implementation",
                "pattern",
                "n_generations",
                "execution_time_seconds",
            ]
        )
        df.to_csv(file_path, index=False)


def add_score(file_path: Path, score: Score) -> None:
    df = pd.read_csv(file_path)

    new_row = pd.DataFrame([score.model_dump(mode="json")])
    df = pd.concat([df, new_row], ignore_index=True)

    df.to_csv(file_path, index=False)


def plot_mean_execution_time(file_path, grid_dimension, pattern, n_generations):
    df = pd.read_csv(file_path)
    df = df[
        (df["grid_dimension"] == grid_dimension)
        & (df["pattern"] == pattern.value)
        & (df["n_generations"] == n_generations)
    ]

    sns.barplot(data=df, x="rule_implementation", y="execution_time_seconds")
    plt.title("Rule Implementation Execution Time")
    plt.yscale("log")
    plt.show()

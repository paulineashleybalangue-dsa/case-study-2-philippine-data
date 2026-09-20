import time

import numpy as np
import pandas as pd


def calculate_loop(values: np.ndarray) -> float:
    """Calculate the total using a Python loop."""

    total = 0.0

    for value in values:
        total += value * 1.10

    return total


def calculate_vectorized(values: np.ndarray) -> float:
    """Calculate the total using NumPy vectorization."""

    return np.sum(values * 1.10)


def run_numpy_comparison(
    data: pd.DataFrame,
    sample_size: int = 10_000,
    runs: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare loop and vectorized NumPy calculations."""

    values = data["dutiablevaluephp"].to_numpy(dtype=float)

    rng = np.random.default_rng(seed)

    sample_indices = rng.choice(
        len(values),
        size=min(sample_size, len(values)),
        replace=False,
    )

    sample = values[sample_indices]

    # Boolean mask.
    positive_mask = sample > 0

    # Apply the mask to keep positive values.
    sample = sample[positive_mask]

    # Run once to verify that both methods produce the same result.
    loop_result = calculate_loop(sample)
    vectorized_result = calculate_vectorized(sample)

    numpy_equal = bool(
        np.isclose(
            loop_result,
            vectorized_result,
        )
    )

    if not numpy_equal:
        raise ValueError(
            "Loop and vectorized calculations do not agree."
        )

    loop_times = []
    vectorized_times = []

    for _ in range(runs):
        start = time.perf_counter()
        calculate_loop(sample)
        loop_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        calculate_vectorized(sample)
        vectorized_times.append(time.perf_counter() - start)

    results = pd.DataFrame(
        {
            "method": [
                "Python loop",
                "NumPy vectorized",
            ],
            "result": [
                loop_result,
                vectorized_result,
            ],
            "median_time_seconds": [
                np.median(loop_times),
                np.median(vectorized_times),
            ],
        }
    )

    results.attrs["numpy_equal"] = numpy_equal

    return results
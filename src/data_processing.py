from pathlib import Path
from typing import Iterator

import pandas as pd


def validate_required_columns(
    file_path: Path,
    required_columns: set[str],
) -> None:
    """Check that the CSV contains all required columns."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {file_path}"
        )

    header = pd.read_csv(file_path, nrows=0)
    actual_columns = set(header.columns)

    missing_columns = required_columns - actual_columns

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Missing required column(s): {missing}"
        )


def load_csv_chunks(
    file_path: Path,
    required_columns: set[str],
    chunksize: int = 100_000,
) -> Iterator[pd.DataFrame]:
    """Load the CSV in chunks after checking required columns."""

    validate_required_columns(file_path, required_columns)

    return pd.read_csv(
        file_path,
        usecols=list(required_columns),
        chunksize=chunksize,
        encoding="cp1252",
    )


def prepare_chunk(
    chunk: pd.DataFrame,
    minimum_value: float = 0,
    high_value_threshold: float = 100,
) -> pd.DataFrame:
    """Filter a chunk and create the required derived columns."""

    # Make a copy so the original chunk is not modified unexpectedly.
    data = chunk.copy()

    # Make sure the numerical measure is numeric.
    data["dutiablevaluephp"] = pd.to_numeric(
        data["dutiablevaluephp"],
        errors="coerce",
    )

    # Two-condition filter.
    mask = (
        data["tq"].notna()
        & (data["dutiablevaluephp"] > minimum_value)
    )

    # Use .loc to select the valid records.
    selected = data.loc[mask].copy()

    # Keep missing country categories as an explicit group.
    selected["countryorigin_iso3"] = (
        selected["countryorigin_iso3"]
        .fillna("Missing")
        .astype(str)
    )

    # Derived numerical column.
    selected["dutiablevalue_million_php"] = (
        selected["dutiablevaluephp"] / 1_000_000
    )

    # Derived category/flag column.
    selected["value_band"] = (
        selected["dutiablevalue_million_php"]
        .ge(high_value_threshold)
        .map({
            True: "High",
            False: "Standard",
        })
    )

    return selected
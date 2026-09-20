from pathlib import Path
import hashlib

import pandas as pd


def inspect_raw_data(
    file_path: Path,
    chunksize: int = 100_000,
) -> dict[str, int | float]:
    """Independently calculate raw row and measure totals."""

    total_rows = 0
    total_sum = 0.0
    selected_sum = 0.0
    missing_measure = 0
    selected_rows = 0
    excluded_rows = 0

    for chunk in pd.read_csv(
        file_path,
        usecols=["tq", "dutiablevaluephp"],
        chunksize=chunksize,
        encoding="cp1252",
        low_memory=False,
    ):
        values = pd.to_numeric(
            chunk["dutiablevaluephp"],
            errors="coerce",
        )

        total_rows += len(chunk)
        total_sum += float(values.sum())
        missing_measure += int(values.isna().sum())

        selected_mask = (
            chunk["tq"].notna()
            & (values > 0)
        )

        selected_rows += int(selected_mask.sum())
        excluded_rows += int((~selected_mask).sum())

        selected_sum += float(
            values.loc[selected_mask].sum()
        )

    return {
        "raw_rows": total_rows,
        "raw_sum": total_sum,
        "selected_sum": selected_sum,
        "missing_measure": missing_measure,
        "selected_rows": selected_rows,
        "excluded_rows": excluded_rows,
    }


def inspect_dataset(
    file_path: Path,
    chunksize: int = 100_000,
) -> pd.DataFrame:
    """Inspect column types and missing values in the full CSV."""

    missing_counts: dict[str, int] = {}
    column_types: dict[str, str] = {}

    first_chunk = True

    for chunk in pd.read_csv(
        file_path,
        chunksize=chunksize,
        encoding="cp1252",
        low_memory=False,
    ):
        if first_chunk:
            column_types = {
                column: str(dtype)
                for column, dtype in chunk.dtypes.items()
            }

            missing_counts = {
                column: 0
                for column in chunk.columns
            }

            first_chunk = False

        for column in chunk.columns:
            missing_counts[column] += int(
                chunk[column].isna().sum()
            )

    inspection = pd.DataFrame(
        {
            "column": list(column_types.keys()),
            "dtype": list(column_types.values()),
            "missing_count": [
                missing_counts[column]
                for column in column_types.keys()
            ],
        }
    )

    return inspection


def calculate_file_sha256(
    file_path: Path,
) -> str:
    """Calculate the SHA-256 hash of the raw CSV file."""

    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            sha256.update(block)

    return sha256.hexdigest()


def create_validation_results(
    raw_stats: dict[str, int | float],
    selected_data: pd.DataFrame,
    grouped: pd.DataFrame,
    grouped_two: pd.DataFrame,
    pivot: pd.DataFrame,
    numpy_equal: bool,
    actual_sha256: str,
    expected_raw_rows: int = 2_236_612,
    expected_raw_sum: float = 3_587_267_375_257,
    expected_sha256: str = (
        "b3b5a3a95340179a716a05611d51ad4906484d38363d1ac36494a404c04e4370"
    ),
    tolerance: float = 1.00,
) -> pd.DataFrame:
    """Create independent validation checks."""

    selected_rows = len(selected_data)

    selected_sum = float(
        selected_data["dutiablevaluephp"].sum()
    )

    grouped_row_count = int(
        grouped["row_count"].sum()
    )

    grouped_sum = float(
        grouped["measure_sum"].sum()
    )

    grouped_two_sum = float(
        grouped_two["measure_sum"].sum()
    )

    pivot_data = pivot.set_index(
        "countryorigin_iso3"
    )

    if "Total" in pivot_data.index:
        pivot_data = pivot_data.drop(
            index=["Total"]
        )

    if "Total" in pivot_data.columns:
        pivot_data = pivot_data.drop(
            columns=["Total"]
        )

    pivot_sum = float(
        pivot_data.select_dtypes(include="number")
        .fillna(0)
        .to_numpy()
        .sum()
    )

    # Check that the groups used for the bar chart
    # agree with the grouped summary.
    top10_summary = (
        grouped.sort_values(
            "measure_sum",
            ascending=False,
        )
        .head(10)
    )

    top10_countries = set(
        top10_summary["countryorigin_iso3"]
    )

    top10_plot_sum = float(
        selected_data.loc[
            selected_data["countryorigin_iso3"].isin(
                top10_countries
            ),
            "dutiablevaluephp",
        ].sum()
    )

    top10_summary_sum = float(
        top10_summary["measure_sum"].sum()
    )

    plot_top10_match = (
        abs(
            top10_plot_sum - top10_summary_sum
        )
        <= tolerance
    )

    # The heatmap is created from the pivot interior,
    # excluding the Total row and Total column.
    heatmap_interior_sum = pivot_sum

    heatmap_matches_pivot = (
        abs(
            heatmap_interior_sum - selected_sum
        )
        <= tolerance
    )

    checks = [
        {
            "check": "raw_row_count",
            "expected": expected_raw_rows,
            "actual": raw_stats["raw_rows"],
            "tolerance": 0,
            "pass": raw_stats["raw_rows"] == expected_raw_rows,
        },
        {
            "check": "raw_dutiablevaluephp_sum",
            "expected": expected_raw_sum,
            "actual": raw_stats["raw_sum"],
            "tolerance": tolerance,
            "pass": abs(
                raw_stats["raw_sum"] - expected_raw_sum
            ) <= tolerance,
        },
        {
            "check": "missing_dutiablevaluephp",
            "expected": 0,
            "actual": raw_stats["missing_measure"],
            "tolerance": 0,
            "pass": raw_stats["missing_measure"] == 0,
        },
        {
            "check": "raw_rows_partition",
            "expected": raw_stats["raw_rows"],
            "actual": (
                raw_stats["selected_rows"]
                + raw_stats["excluded_rows"]
            ),
            "tolerance": 0,
            "pass": (
                raw_stats["raw_rows"]
                == raw_stats["selected_rows"]
                + raw_stats["excluded_rows"]
            ),
        },
        {
            "check": "selected_row_count",
            "expected": raw_stats["selected_rows"],
            "actual": selected_rows,
            "tolerance": 0,
            "pass": selected_rows == raw_stats["selected_rows"],
        },
        {
            "check": "selected_measure_sum",
            "expected": raw_stats["selected_sum"],
            "actual": selected_sum,
            "tolerance": tolerance,
            "pass": abs(
                selected_sum - raw_stats["selected_sum"]
            ) <= tolerance,
        },
        {
            "check": "grouped_row_counts",
            "expected": selected_rows,
            "actual": grouped_row_count,
            "tolerance": 0,
            "pass": selected_rows == grouped_row_count,
        },
        {
            "check": "grouped_sum",
            "expected": selected_sum,
            "actual": grouped_sum,
            "tolerance": tolerance,
            "pass": abs(
                selected_sum - grouped_sum
            ) <= tolerance,
        },
        {
            "check": "grouped_two_sum",
            "expected": selected_sum,
            "actual": grouped_two_sum,
            "tolerance": tolerance,
            "pass": abs(
                selected_sum - grouped_two_sum
            ) <= tolerance,
        },
        {
            "check": "pivot_interior_sum",
            "expected": selected_sum,
            "actual": pivot_sum,
            "tolerance": tolerance,
            "pass": abs(
                selected_sum - pivot_sum
            ) <= tolerance,
        },
        {
            "check": "top10_plot_values",
            "expected": top10_summary_sum,
            "actual": top10_plot_sum,
            "tolerance": tolerance,
            "pass": plot_top10_match,
        },
        {
            "check": "heatmap_plot_values",
            "expected": selected_sum,
            "actual": heatmap_interior_sum,
            "tolerance": tolerance,
            "pass": heatmap_matches_pivot,
        },
        {
            "check": "numpy_loop_vectorized",
            "expected": True,
            "actual": numpy_equal,
            "tolerance": 0,
            "pass": numpy_equal,
        },
        {
            "check": "raw_file_sha256",
            "expected": expected_sha256,
            "actual": actual_sha256,
            "tolerance": 0,
            "pass": actual_sha256 == expected_sha256,
        },
    ]

    return pd.DataFrame(checks)


def save_validation(
    validation: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save validation results and stop if a check fails."""

    validation.to_csv(
        output_path,
        index=False,
    )

    failed = validation.loc[
        ~validation["pass"]
    ]

    if not failed.empty:
        print()
        print("VALIDATION FAILED")
        print(failed.to_string(index=False))
        raise SystemExit(1)

    print()
    print("All validation checks passed.")
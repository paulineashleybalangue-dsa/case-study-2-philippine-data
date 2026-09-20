import pandas as pd
from config import CONFIG, REQUIRED_COLUMNS
from src.data_processing import load_csv_chunks, prepare_chunk
from src.analysis import CustomsAnalyzer
from src.numpy_analysis import run_numpy_comparison
from src.visualization import create_bar_plot, create_heatmap
from src.validation import (
    inspect_raw_data,
    inspect_dataset,
    calculate_file_sha256,
    create_validation_results,
    save_validation,
)

def main() -> None:
    """Run the Philippine Customs data analysis."""

    output_dir = CONFIG["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    inspection = inspect_dataset(
        CONFIG["input_path"],
        CONFIG["chunksize"],
    )

    inspection.to_csv(
        output_dir / "inspection.csv",
        index=False,
    )

    raw_stats = inspect_raw_data(
        CONFIG["input_path"],
        CONFIG["chunksize"],
    )

    analyzer = CustomsAnalyzer(
        output_dir=output_dir,
        group_columns=CONFIG["group_columns"],
        measure_column=CONFIG["measure_column"],
    )

    chunks = load_csv_chunks(
        CONFIG["input_path"],
        REQUIRED_COLUMNS,
        CONFIG["chunksize"],
    )

    chunk_number = 0

    for chunk in chunks:
        chunk_number += 1
        print(f"Processing chunk {chunk_number}...")

        rows_before = len(chunk)

        selected = prepare_chunk(
            chunk,
            CONFIG["minimum_dutiable_value_php"],
            CONFIG["high_value_threshold_million_php"],
        )

        rows_after = len(selected)

        analyzer.add_chunk(
            selected,
            rows_before,
            rows_after,
        )
        
    data = analyzer.combine_chunks()

    grouped = analyzer.create_grouped_summary(data)
    grouped_two = analyzer.create_two_category_summary(data)
    pivot = analyzer.create_pivot(grouped_two)
    top10 = analyzer.create_top10(grouped)

    numpy_results = run_numpy_comparison(data)

    analyzer.add_audit_record(
        step="numpy",
        operation="Compare loop and vectorized calculation",
        rule="Fixed-seed sample with median of 5 runs",
        rows_before=len(data),
        rows_after=len(data),
    )

    numpy_results.to_csv(
        output_dir / "numpy_comparison.csv",
        index=False,
    )

    actual_sha256 = calculate_file_sha256(
        CONFIG["input_path"]
    )

    validation = create_validation_results(
        raw_stats=raw_stats,
        selected_data=data,
        grouped=grouped,
        grouped_two=grouped_two,
        pivot=pivot,
        numpy_equal=numpy_results.attrs["numpy_equal"],
        actual_sha256=actual_sha256,

    )

    save_validation(
        validation,
        output_dir / "validation.csv",
    )

    create_bar_plot(
        top10,
        output_dir / "bar.png",
    )

    create_heatmap(
        pivot,
        output_dir / "heatmap.png",
    )

    analyzer.add_audit_record(
        step="plots",
        operation="Create bar chart and heatmap",
        rule="Use values from summary tables",
        rows_before=len(data),
        rows_after=len(data),
    )

    grouped.to_csv(output_dir / "grouped.csv", index=False)
    grouped_two.to_csv(output_dir / "grouped_two.csv", index=False)
    pivot.to_csv(output_dir / "pivot.csv", index=False)
    top10.to_csv(output_dir / "top10.csv", index=False)

    audit_log = pd.DataFrame(analyzer.audit_records)

    audit_log.to_csv(
        output_dir / "audit_log.csv",
        index=False,
    )

    print()
    print("Summary tables created successfully.")
    
    print()
    print("Plots created successfully:")
    print("-", output_dir / "bar.png")
    print("-", output_dir / "heatmap.png")

    print()
    print("NumPy comparison:")
    print(numpy_results.to_string(index=False))

if __name__ == "__main__":
    main()
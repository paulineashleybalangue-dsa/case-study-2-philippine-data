from pathlib import Path


# Required columns from the Philippine Customs 2015 dataset
REQUIRED_COLUMNS = {
    "countryorigin_iso3",
    "tq",
    "dutiablevaluephp",
}


# Main project configuration
CONFIG = {
    "input_path": Path("data/2015.csv"),
    "output_dir": Path("outputs"),
    "group_columns": ["countryorigin_iso3", "tq"],
    "measure_column": "dutiablevaluephp",
    "chunksize": 100_000,

    # Filtering rules
    "require_tq": True,
    "minimum_dutiable_value_php": 0,

    # Derived-column rule
    "high_value_threshold_million_php": 100,
}
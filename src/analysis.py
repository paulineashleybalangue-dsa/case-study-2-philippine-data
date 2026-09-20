from pathlib import Path

import pandas as pd


class CustomsAnalyzer:
    """Analyze Philippine Customs 2015 data."""

    def __init__(
        self,
        output_dir: Path,
        group_columns: list[str],
        measure_column: str,
    ) -> None:
        """Initialize the analyzer configuration."""

        self.output_dir = output_dir
        self.group_columns = group_columns
        self.measure_column = measure_column

        self.processed_chunks: list[pd.DataFrame] = []
        self.audit_records: list[dict] = []

    def add_audit_record(
        self,
        step: str,
        operation: str,
        rule: str,
        rows_before: int,
        rows_after: int,
    ) -> None:
        """Add one processing step to the audit log."""

        self.audit_records.append(
            {
                "step": step,
                "operation": operation,
                "rule": rule,
                "rows_before": rows_before,
                "rows_after": rows_after,
            }
        )

    def add_chunk(
        self,
        chunk: pd.DataFrame,
        rows_before: int,
        rows_after: int,
    ) -> None:
        """Store a processed chunk and record its audit information."""

        self.processed_chunks.append(chunk)

        self.add_audit_record(
            step="filter",
            operation="Apply two-condition filter",
            rule=(
                "tq is not missing AND "
                "dutiablevaluephp > 0"
            ),
            rows_before=rows_before,
            rows_after=rows_after,
        )

    def combine_chunks(self) -> pd.DataFrame:
        """Combine all processed chunks into one DataFrame."""

        if not self.processed_chunks:
            raise ValueError("No processed data available.")

        data = pd.concat(
            self.processed_chunks,
            ignore_index=True,
        )

        self.add_audit_record(
            step="combine",
            operation="Combine processed chunks",
            rule="Concatenate all filtered chunks",
            rows_before=sum(
                len(chunk)
                for chunk in self.processed_chunks
            ),
            rows_after=len(data),
        )

        return data

    def create_grouped_summary(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create the summary grouped by country of origin."""

        grouped = (
            data.groupby(
                self.group_columns[0],
                dropna=False,
            )[self.measure_column]
            .agg(
                row_count="size",
                valid_measure_count="count",
                measure_sum="sum",
                measure_mean="mean",
            )
            .reset_index()
        )

        self.add_audit_record(
            step="summary",
            operation="Create grouped summary",
            rule="Group by country of origin",
            rows_before=len(data),
            rows_after=len(grouped),
        )

        return grouped

    def create_two_category_summary(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create the summary grouped by country and quarter."""

        grouped_two = (
            data.groupby(
                self.group_columns,
                dropna=False,
            )[self.measure_column]
            .agg(
                row_count="size",
                measure_sum="sum",
            )
            .reset_index()
        )

        self.add_audit_record(
            step="summary",
            operation="Create two-category summary",
            rule="Group by country of origin and quarter",
            rows_before=len(data),
            rows_after=len(grouped_two),
        )

        return grouped_two

    def create_pivot(
        self,
        grouped_two: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create a country-by-quarter measure-sum pivot table."""

        pivot = grouped_two.pivot_table(
            index=self.group_columns[0],
            columns=self.group_columns[1],
            values="measure_sum",
            aggfunc="sum",
            margins=True,
            margins_name="Total",
        )

        pivot = pivot.reset_index()

        self.add_audit_record(
            step="summary",
            operation="Create pivot table",
            rule="Country by quarter with total margins",
            rows_before=len(grouped_two),
            rows_after=len(pivot),
        )

        return pivot

    def create_top10(
        self,
        grouped: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return the ten groups with the largest measure sum."""

        top10 = (
            grouped.sort_values(
                "measure_sum",
                ascending=False,
            )
            .head(10)
            .reset_index(drop=True)
        )

        self.add_audit_record(
            step="summary",
            operation="Create top-10 summary",
            rule="Keep ten largest groups by measure sum",
            rows_before=len(grouped),
            rows_after=len(top10),
        )

        return top10
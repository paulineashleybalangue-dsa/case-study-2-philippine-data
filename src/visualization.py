from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def create_bar_plot(
    top10: pd.DataFrame,
    output_path: Path,
) -> None:
    """Create a bar chart showing the top groups by total dutiable value."""

    plot_data = top10.sort_values(
        "measure_sum",
        ascending=True,
    )

    plt.figure(figsize=(10, 6))

    plt.barh(
        plot_data["countryorigin_iso3"],
        plot_data["measure_sum"],
    )

    plt.title(
        "Top 10 Countries by Total Dutiable Value"
    )

    plt.xlabel(
        "Total Dutiable Value (PHP)"
    )

    plt.ylabel(
        "Country of Origin"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def create_heatmap(
    pivot: pd.DataFrame,
    output_path: Path,
) -> None:
    """Create a heatmap of dutiable value by country and quarter."""

    heatmap_data = pivot.set_index(
        "countryorigin_iso3"
    )

    # Remove the Total column so margins are not plotted.
    if "Total" in heatmap_data.columns:
        heatmap_data = heatmap_data.drop(
            columns=["Total"]
        )

    # Remove the Total row if present.
    if "Total" in heatmap_data.index:
        heatmap_data = heatmap_data.drop(
            index=["Total"]
        )

    plt.figure(
        figsize=(12, max(8, len(heatmap_data) * 0.2))
    )

    sns.heatmap(
        heatmap_data,
        cmap="YlGnBu",
        linewidths=0.2,
        cbar_kws={
            "label": "Total Dutiable Value (PHP)"
        },
    )

    plt.title(
        "Dutiable Value by Country of Origin and Quarter"
    )

    plt.xlabel("Quarter")
    plt.ylabel("Country of Origin")

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def make_plots_for_run(run_dir: Path) -> list[Path]:
    summary_path = run_dir / "metrics_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Metrics summary not found: {summary_path}")
    rows = [
        row
        for row in csv.DictReader(summary_path.open("r", encoding="utf-8"))
        if (
            row.get("category") == "__all__"
            and row.get("formality", "__all__") == "__all__"
            and row.get("value") not in {"", None}
        )
    ]
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    output_paths: list[Path] = []
    metrics = sorted({row["metric"] for row in rows})
    for metric in metrics:
        output_paths.append(_barplot_metric(rows, metric, figures_dir))

    diversity_metrics = [
        row
        for row in rows
        if row["metric"] in {"jaccard", "self_bleu", "distinct_1", "distinct_2"}
    ]
    if diversity_metrics:
        output_paths.append(_overview_plot(diversity_metrics, figures_dir))
    return output_paths


def _barplot_metric(rows: list[dict[str, Any]], metric: str, figures_dir: Path) -> Path:
    metric_rows = [row for row in rows if row["metric"] == metric]
    labels = [f"{row['model_id']}\n{row['condition']}" for row in metric_rows]
    values = [float(row["value"]) for row in metric_rows]

    width = max(8, len(labels) * 0.75)
    fig, ax = plt.subplots(figsize=(width, 4.8))
    colors = [_condition_color(row["condition"]) for row in metric_rows]
    ax.bar(range(len(values)), values, color=colors)
    ax.set_title(f"{metric} por modelo e condicao")
    ax.set_ylabel(metric)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output = figures_dir / f"{metric}.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def _overview_plot(rows: list[dict[str, Any]], figures_dir: Path) -> Path:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["condition"]].append(float(row["value"]))
    labels = sorted(grouped)
    values = [sum(grouped[label]) / len(grouped[label]) for label in labels]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(labels, values, marker="o", color="#2563eb", linewidth=2)
    ax.set_title("Resumo simples de metricas de diversidade por condicao")
    ax.set_ylabel("media simples")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output = figures_dir / "diversity_overview.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def _condition_color(condition: str) -> str:
    if condition == "baseline":
        return "#64748b"
    if condition == "vs_intermediate":
        return "#0f766e"
    if condition == "vs_aggressive":
        return "#b45309"
    return "#4f46e5"

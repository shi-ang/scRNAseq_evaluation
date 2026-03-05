from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("paper")
sns.set_style("ticks")

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "figure.dpi": 300,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "Helvetica", "sans-serif"],
        "font.size": 12,
        "axes.titlesize": 18.5,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.linewidth": 1.0,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "legend.fontsize": 15,
        "legend.title_fontsize": 15,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)

EXPECTED_MODEL_ORDER = ("Control", "Average", "linearPCA", "scVI")

PAIR_METRIC_GROUPS: tuple[tuple[str, str], ...] = (
    ("pearson", "pearson_degs"),
    ("mae", "mae_degs"),
    ("mse", "mse_degs"),
    ("r2", "r2_degs"),
)
PDS_METRICS: tuple[str, ...] = ("pds_l1", "pds_l2", "pds_cosine")

METRIC_LABELS: dict[str, str] = {
    "pearson": "Pearson",
    "pearson_degs": "Pearson (DEGs)",
    "mae": "MAE",
    "mae_degs": "MAE (DEGs)",
    "mse": "MSE",
    "mse_degs": "MSE (DEGs)",
    "r2": "R2",
    "r2_degs": "R2 (DEGs)",
    "pds_l1": "PDS (L1)",
    "pds_l2": "PDS (L2)",
    "pds_cosine": "PDS (Cosine)",
    "vendi_abs_diff": "|Vendi Pred - Vendi Obs|",
}

# Metric colors (base hue); boxes are drawn as light shade, dots/edges in darker shade.
METRIC_BASE_COLORS: dict[str, str] = {
    "pearson": "#c92a2a",
    "pearson_degs": "#1c7ed6",
    "mae": "#c92a2a",
    "mae_degs": "#1c7ed6",
    "mse": "#c92a2a",
    "mse_degs": "#1c7ed6",
    "r2": "#c92a2a",
    "r2_degs": "#1c7ed6",
    "pds_l1": "#c92a2a",
    "pds_l2": "#1c7ed6",
    "pds_cosine": "#2b8a3e",
    "vendi_abs_diff": "#9c36b5",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot model-comparison boxplots for real experiment results."
    )
    parser.add_argument(
        "--results",
        type=str,
        default=None,
        help=(
            "Path to a results CSV. "
            "If omitted, the latest file in results/real_experiments is used."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save plots. Default: <results_stem>_plots",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI for saved files (default: 300).",
    )
    parser.add_argument(
        "--cluster_gap",
        type=float,
        default=1.2,
        help="Horizontal gap between model clusters (default: 1.2).",
    )
    return parser.parse_args()


def latest_results_file(search_dir: Path) -> Path:
    files = sorted(search_dir.glob("real_experiment_results_*.csv"))
    if not files:
        raise FileNotFoundError(f"No result files found in: {search_dir}")
    return max(files, key=lambda p: p.stat().st_mtime)


def resolve_results_path(results_arg: str | None) -> Path:
    if results_arg:
        path = Path(results_arg)
        if not path.exists():
            raise FileNotFoundError(f"Results file not found: {path}")
        return path
    return latest_results_file(Path("results/real_experiments"))


def load_results(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    drop_cols = [col for col in df.columns if col.endswith("_true_degs")]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        print(f"Dropped *_true_degs columns: {drop_cols}")

    if "status" in df.columns:
        before = len(df)
        df = df[df["status"].astype(str).str.lower() == "success"].copy()
        print(f"Kept successful runs: {len(df)}/{before}")

    if "model" not in df.columns:
        raise KeyError("Results file must include a 'model' column.")
    if df.empty:
        raise ValueError("No rows available to plot after filtering.")
    return df


def coerce_numeric(df: pd.DataFrame, columns: Iterable[str]) -> None:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def get_model_order(df: pd.DataFrame) -> list[str]:
    models = [str(m) for m in df["model"].dropna().unique()]
    preferred = [m for m in EXPECTED_MODEL_ORDER if m in models]
    extras = sorted(m for m in models if m not in EXPECTED_MODEL_ORDER)
    order = [*preferred, *extras]
    if not order:
        raise ValueError("No model names found in the results.")
    return order


def has_valid_metric(df: pd.DataFrame, metric: str) -> bool:
    if metric not in df.columns:
        return False
    values = pd.to_numeric(df[metric], errors="coerce").to_numpy(dtype=float)
    return bool(np.isfinite(values).any())


def lighten(color: tuple[float, float, float], amount: float = 0.6) -> tuple[float, float, float]:
    return tuple((1.0 - amount) * c + amount for c in color)


def darken(color: tuple[float, float, float], amount: float = 0.7) -> tuple[float, float, float]:
    return tuple(c * amount for c in color)


def build_metric_styles(metrics: Sequence[str]) -> dict[str, dict[str, tuple[float, float, float]]]:
    cmap = plt.get_cmap("tab10")
    styles: dict[str, dict[str, tuple[float, float, float]]] = {}
    for idx, metric in enumerate(metrics):
        base = METRIC_BASE_COLORS.get(metric, cmap(idx % 10))
        base_rgb = mcolors.to_rgb(base)
        styles[metric] = {
            "box": lighten(base_rgb, amount=0.62),
            "edge": darken(base_rgb, amount=0.68),
            "dot": darken(base_rgb, amount=0.55),
        }
    return styles


def metric_legend_handles(
    metrics: Sequence[str],
    metric_styles: dict[str, dict[str, tuple[float, float, float]]],
) -> list[Patch]:
    return [
        Patch(
            facecolor=metric_styles[metric]["box"],
            edgecolor=metric_styles[metric]["edge"],
            label=METRIC_LABELS.get(metric, metric),
        )
        for metric in metrics
    ]


def plot_model_grouped_boxplot(
    ax: Axes,
    df: pd.DataFrame,
    metrics: Sequence[str],
    model_order: Sequence[str],
    metric_styles: dict[str, dict[str, tuple[float, float, float]]],
    cluster_gap: float,
    ylabel: str,
) -> None:
    if not metrics:
        ax.text(0.5, 0.5, "No valid metrics to plot", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    n_models = len(model_order)
    n_metrics = len(metrics)
    if n_models == 0:
        raise ValueError("No models available for plotting.")

    rng = np.random.default_rng(0)
    model_centers: list[float] = []
    any_boxes = False

    for model_idx, model in enumerate(model_order):
        cluster_start = model_idx * (n_metrics + cluster_gap)
        cluster_positions: list[float] = []

        for metric_idx, metric in enumerate(metrics):
            position = cluster_start + metric_idx
            values = df.loc[df["model"] == model, metric].dropna().to_numpy(dtype=float)
            if values.size == 0:
                continue

            any_boxes = True
            style = metric_styles[metric]

            boxplot = ax.boxplot(
                values,
                positions=[position],
                widths=0.72,
                patch_artist=True,
                showfliers=False,
                zorder=2,
            )
            for box in boxplot["boxes"]:
                box.set_facecolor(style["box"])
                box.set_edgecolor(style["edge"])
                box.set_linewidth(1.2)
                box.set_alpha(0.9)
            for whisker in boxplot["whiskers"]:
                whisker.set_color(style["edge"])
                whisker.set_linewidth(1.1)
            for cap in boxplot["caps"]:
                cap.set_color(style["edge"])
                cap.set_linewidth(1.1)
            for median in boxplot["medians"]:
                median.set_color(style["dot"])
                median.set_linewidth(1.4)

            jitter = rng.normal(0.0, 0.05, size=values.size)
            ax.scatter(
                np.full(values.size, position) + jitter,
                values,
                color=style["dot"],
                alpha=0.65,
                s=18,
                linewidths=0,
                zorder=3,
            )
            cluster_positions.append(position)

        if cluster_positions:
            model_centers.append(float(np.mean(cluster_positions)))
        else:
            model_centers.append(cluster_start + (n_metrics - 1) / 2.0)

    if not any_boxes:
        ax.text(0.5, 0.5, "No valid values", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    total_width = (n_models - 1) * (n_metrics + cluster_gap) + n_metrics
    ax.set_xlim(-0.9, total_width - 0.1)
    ax.set_xticks(model_centers)
    ax.set_xticklabels(model_order)
    ax.set_xlabel("Model")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)


def save_pair_metric_plot(
    df: pd.DataFrame,
    model_order: Sequence[str],
    pair: tuple[str, str],
    cluster_gap: float,
    output_path: Path,
    dpi: int,
) -> bool:
    metrics = [metric for metric in pair if has_valid_metric(df, metric)]
    if not metrics:
        print(f"Skipping {pair}: no valid values.")
        return False

    metric_styles = build_metric_styles(metrics)
    fig, ax = plt.subplots(figsize=(8.0, 5.8))
    plot_model_grouped_boxplot(
        ax=ax,
        df=df,
        metrics=metrics,
        model_order=model_order,
        metric_styles=metric_styles,
        cluster_gap=cluster_gap,
        ylabel="Score",
    )

    if len(metrics) == 2:
        title = f"{METRIC_LABELS.get(metrics[0], metrics[0])} vs {METRIC_LABELS.get(metrics[1], metrics[1])}"
    else:
        title = METRIC_LABELS.get(metrics[0], metrics[0])
    ax.set_title(title)

    ax.legend(
        handles=metric_legend_handles(metrics, metric_styles),
        title="Metric",
        loc="best",
        frameon=True,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def save_paired_metric_plots(
    df: pd.DataFrame,
    model_order: Sequence[str],
    cluster_gap: float,
    output_dir: Path,
    dpi: int,
) -> list[Path]:
    saved: list[Path] = []
    for left, right in PAIR_METRIC_GROUPS:
        path = output_dir / f"{left}_vs_{right}_boxplot.pdf"
        if save_pair_metric_plot(
            df=df,
            model_order=model_order,
            pair=(left, right),
            cluster_gap=cluster_gap,
            output_path=path,
            dpi=dpi,
        ):
            saved.append(path)
    return saved


def plot_vendi_abs_diff(
    df: pd.DataFrame,
    model_order: Sequence[str],
    cluster_gap: float,
    output_path: Path,
    dpi: int,
) -> bool:
    required = ("vendi_score_pred", "vendi_score_obs")
    if any(col not in df.columns for col in required):
        print("Skipping vendi abs-diff plot: missing vendi_score_pred or vendi_score_obs.")
        return False

    vendi_df = df.copy()
    vendi_df["vendi_abs_diff"] = (vendi_df["vendi_score_pred"] - vendi_df["vendi_score_obs"]).abs()
    if not has_valid_metric(vendi_df, "vendi_abs_diff"):
        print("Skipping vendi abs-diff plot: no valid values.")
        return False

    metric_styles = build_metric_styles(["vendi_abs_diff"])
    fig, ax = plt.subplots(figsize=(8.0, 5.8))
    plot_model_grouped_boxplot(
        ax=ax,
        df=vendi_df,
        metrics=["vendi_abs_diff"],
        model_order=model_order,
        metric_styles=metric_styles,
        cluster_gap=cluster_gap,
        ylabel="Absolute Difference",
    )
    ax.set_title("Vendi Score Absolute Difference by Model")
    ax.legend(
        handles=metric_legend_handles(["vendi_abs_diff"], metric_styles),
        title="Metric",
        loc="best",
        frameon=True,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_pds_metrics(
    df: pd.DataFrame,
    model_order: Sequence[str],
    cluster_gap: float,
    output_path: Path,
    dpi: int,
) -> bool:
    metrics = [metric for metric in PDS_METRICS if has_valid_metric(df, metric)]
    if not metrics:
        print("Skipping PDS plot: no valid PDS metrics found.")
        return False

    metric_styles = build_metric_styles(metrics)
    fig, ax = plt.subplots(figsize=(10.0, 5.8))
    plot_model_grouped_boxplot(
        ax=ax,
        df=df,
        metrics=metrics,
        model_order=model_order,
        metric_styles=metric_styles,
        cluster_gap=cluster_gap,
        ylabel="Score",
    )
    ax.set_title("PDS Metrics by Model")
    ax.legend(
        handles=metric_legend_handles(metrics, metric_styles),
        title="Metric",
        loc="best",
        frameon=True,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def main() -> None:
    args = parse_args()

    results_path = resolve_results_path(args.results)
    print(f"Using results: {results_path}")

    df = load_results(results_path)
    coerce_numeric(
        df,
        {
            *[metric for pair in PAIR_METRIC_GROUPS for metric in pair],
            "vendi_score_pred",
            "vendi_score_obs",
            *PDS_METRICS,
        },
    )

    model_order = get_model_order(df)
    print(f"Models: {model_order}")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = results_path.parent / f"{results_path.stem}_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_pair_paths = save_paired_metric_plots(
        df=df,
        model_order=model_order,
        cluster_gap=float(args.cluster_gap),
        output_dir=output_dir,
        dpi=int(args.dpi),
    )

    vendi_plot_path = output_dir / "vendi_abs_diff_boxplot.pdf"
    pds_plot_path = output_dir / "pds_metrics_boxplots.pdf"

    vendi_saved = plot_vendi_abs_diff(
        df=df,
        model_order=model_order,
        cluster_gap=float(args.cluster_gap),
        output_path=vendi_plot_path,
        dpi=int(args.dpi),
    )
    pds_saved = plot_pds_metrics(
        df=df,
        model_order=model_order,
        cluster_gap=float(args.cluster_gap),
        output_path=pds_plot_path,
        dpi=int(args.dpi),
    )

    for path in saved_pair_paths:
        print(f"Saved: {path}")
    if vendi_saved:
        print(f"Saved: {vendi_plot_path}")
    if pds_saved:
        print(f"Saved: {pds_plot_path}")


if __name__ == "__main__":
    main()

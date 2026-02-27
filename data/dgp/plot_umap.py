from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import anndata as ad
from anndata.experimental import AnnCollection
from umap import UMAP
from .synthetic_two import synthetic_causalDGP
from metrics.reconstruction.vendi_score import vendi_score


def _plot_umap_for_single_perturbation(
    embedding: np.ndarray,
    cell_type: np.ndarray,
    perturbation_id: int,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    color_map = {0: "#1f77b4", 1: "#d62728"}

    fig, ax = plt.subplots(figsize=(9, 7), dpi=150)
    point_colors = [color_map.get(int(ct), "#7f7f7f") for ct in cell_type]
    ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=point_colors,
        marker="o",
        s=18,
        alpha=0.82,
        linewidths=0.0,
        rasterized=True,
    )

    ax.set_title(f"Synthetic UMAP: perturbation {perturbation_id}, color = cell type")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")

    cell_type_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=color_map[0],
            markeredgecolor="none",
            markersize=8,
            label="Type A (0)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=color_map[1],
            markeredgecolor="none",
            markersize=8,
            label="Type B (1)",
        ),
    ]
    ax.legend(handles=cell_type_handles, title="Cell Type", loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic causal DGP data and plot UMAP by cell type per perturbation."
    )
    parser.add_argument("--output-dir", type=str, default="results/other_plots", help="Directory for generated chunks and UMAP figures.")
    parser.add_argument("--G", type=int, default=150, help="Number of genes.")
    parser.add_argument("--N0", type=int, default=2048, help="Number of control cells.")
    parser.add_argument("--Nk", type=int, default=2048, help="Number of cells per perturbation.")
    parser.add_argument("--P", type=int, default=5, help="Number of perturbations.")
    parser.add_argument("--mu_l", type=float, default=1.0, help="Mean of log library size for the synthetic data.")
    parser.add_argument("--swap-fraction", type=float, default=0.2, help="Fraction of A edges rewired to create A_alter.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
    parser.add_argument("--max-cells-per-chunk", type=int, default=1024, help="Chunk size for writing h5ad files.")
    parser.add_argument("--umap-n-neighbors", type=int, default=15, help="UMAP n_neighbors.")
    parser.add_argument("--umap-min-dist", type=float, default=0.15, help="UMAP min_dist.")
    parser.add_argument("--normalized-layer-key", type=str, default="normalized_log1p", help="Layer key used as UMAP input (falls back to .X if missing).")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_params_df = pd.read_csv("results/synthetic_simulations/parameter_estimation/all_fitted_params.csv", index_col=0)
    all_theta = all_params_df['n'].values

    # Keep this demo fixed at P=5 as requested.
    chunk_paths, all_affected_masks = synthetic_causalDGP(
        G=args.G,
        N0=args.N0,
        Nk=args.Nk,
        P=args.P,
        mu_l=args.mu_l,
        all_theta=all_theta,
        swap_fraction=args.swap_fraction,
        seed=args.seed,
        output_dir=str(output_dir),
        mask_method="power-law",
        max_cells_per_chunk=args.max_cells_per_chunk,
        normalize=True,
        normalized_layer_key=args.normalized_layer_key,
        visualize=True
    )
    print(f"Wrote {len(chunk_paths)} chunk files under: {output_dir}")

    adatas = [ad.read_h5ad(path) for path in chunk_paths]
    ac = AnnCollection(adatas, join_vars="inner")
    print(f"Vendi score for the dataset: {vendi_score(ac, layer_key=args.normalized_layer_key)}")

    adata = ad.concat(adatas, axis=0, join="outer", merge="same", index_unique=None)

    required_obs = {"cell_type", "perturbation"}
    missing_obs = required_obs.difference(adata.obs.columns)
    if missing_obs:
        raise ValueError(f"Missing required obs columns: {sorted(missing_obs)}")

    if args.normalized_layer_key in adata.layers:
        print(f"Using layer '{args.normalized_layer_key}' for UMAP input.")
        X = adata.layers[args.normalized_layer_key]
    else:
        print(f"Layer '{args.normalized_layer_key}' not found; falling back to adata.X for UMAP input.")
        X = adata.X
    reducer = UMAP(
        n_components=2,
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
        metric="euclidean",
        random_state=args.seed,
    )
    embedding = reducer.fit_transform(X)

    cell_type = adata.obs["cell_type"].to_numpy(dtype=np.int32, copy=False)
    perturbation = adata.obs["perturbation"].to_numpy(dtype=np.int32, copy=False)
    control_idx = perturbation == -1
    if np.any(control_idx):
        control_fig_path = output_dir / "umap_cell_type_control.png"
        _plot_umap_for_single_perturbation(
            embedding=embedding[control_idx],
            cell_type=cell_type[control_idx],
            perturbation_id=-1,
            output_path=control_fig_path,
        )
        print(f"Saved UMAP plot to: {control_fig_path}")
    else:
        print("No control cells (perturbation == -1) found; skipping control plot.")

    perturbation_values = np.unique(perturbation)
    perturbation_values = perturbation_values[perturbation_values >= 0]

    if perturbation_values.size == 0:
        raise ValueError("No perturbation cells (perturbation >= 0) found to plot.")

    for p in perturbation_values:
        idx = perturbation == int(p)
        fig_path = output_dir / f"umap_cell_type_perturbation_{int(p):02d}.png"
        _plot_umap_for_single_perturbation(
            embedding=embedding[idx],
            cell_type=cell_type[idx],
            perturbation_id=int(p),
            output_path=fig_path,
        )
        print(f"Saved UMAP plot to: {fig_path}")


if __name__ == "__main__":
    main()

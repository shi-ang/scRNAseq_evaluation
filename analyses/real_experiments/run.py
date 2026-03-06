from __future__ import annotations

import argparse
import os
import time
from datetime import datetime
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from analyses.context_splitter import ContextSplitter
from analyses.evaluator import evaluation, get_eval_perturbation_ids
from analyses.synthetic_simulations.util import (
    compute_means_by_perturbation,
    get_pseudobulks_and_degs,
)
from models.linear import PseudoPCALinear
from models.scvi_pert import ScviPerturbation
from util.anndata_util import AnnDataProxy

_MODELS = ["Control", "Average", "linearPCA", "scVI"]
_NORM_LAYER_KEY = "normalized_log1p"


def _encode_perturbations(
    adata: ad.AnnData,
    perturbation_key: str,
    control_label: str,
) -> tuple[ad.AnnData, dict[str, int]]:
    if perturbation_key not in adata.obs.columns:
        raise KeyError(
            f"Missing obs column '{perturbation_key}'. Available columns: {list(adata.obs.columns)}"
        )

    perturbation_labels = adata.obs[perturbation_key].astype(str)
    if control_label not in set(perturbation_labels.unique()):
        raise ValueError(
            f"Control label '{control_label}' not found in '{perturbation_key}'."
        )

    non_control_labels = sorted(label for label in perturbation_labels.unique() if label != control_label)
    mapping = {label: idx for idx, label in enumerate(non_control_labels)}

    encoded = np.full(adata.n_obs, -1, dtype=np.int32)
    non_control_mask = perturbation_labels.values != control_label
    encoded[non_control_mask] = np.array(
        [mapping[label] for label in perturbation_labels.values[non_control_mask]],
        dtype=np.int32,
    )

    adata.obs["perturbation_original"] = perturbation_labels.values
    adata.obs["perturbation"] = encoded
    return adata, mapping


def _ensure_normalized_log1p_layer(
    adata: ad.AnnData,
    output_layer_key: str = _NORM_LAYER_KEY,
    source_layer: str | None = "counts",
    target_sum: float = 1e4,
) -> None:
    """Build normalized+log1p layer once for real datasets if missing."""
    if output_layer_key in adata.layers:
        return

    if source_layer is not None:
        if source_layer not in adata.layers:
            raise KeyError(
                f"Requested source_layer='{source_layer}' not found. Available layers: {list(adata.layers.keys())}"
            )
        source = adata.layers[source_layer]
    else:
        source = adata.X

    tmp = ad.AnnData(X=source.copy())
    sc.pp.normalize_total(tmp, target_sum=float(target_sum))
    sc.pp.log1p(tmp)

    if sparse.issparse(tmp.X):
        adata.layers[output_layer_key] = tmp.X.astype(np.float32)
    else:
        adata.layers[output_layer_key] = np.asarray(tmp.X, dtype=np.float32)


def load_real_dataset(
    dataset_path: str,
    perturbation_key: str,
    cell_line_key: str,
    control_label: str,
) -> tuple[ad.AnnData, dict[str, int]]:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    adata = ad.read_h5ad(dataset_path)
    if cell_line_key not in adata.obs.columns:
        raise KeyError(
            f"Missing obs column '{cell_line_key}'. Available columns: {list(adata.obs.columns)}"
        )

    adata, label_mapping = _encode_perturbations(
        adata=adata,
        perturbation_key=perturbation_key,
        control_label=control_label,
    )
    return adata, label_mapping

def _build_perturbation_label_map(adata: ad.AnnData) -> dict[int, str]:
    pairs = adata.obs[["perturbation", "perturbation_original"]].drop_duplicates()
    return {
        int(pid): str(label)
        for pid, label in zip(pairs["perturbation"].to_numpy(), pairs["perturbation_original"].to_numpy())
    }


def _label_to_target_tokens(label: str) -> tuple[str, ...]:
    return tuple(token for token in str(label).split("+") if token and token != "control")


def _has_known_target(tokens: tuple[str, ...], gene_name_set: set[str]) -> bool:
    return any(token in gene_name_set for token in tokens)


def _ensure_control_cells_for_distribution_eval(
    adata: ad.AnnData,
    obs_eval: ad.AnnData,
    pred_eval: ad.AnnData | None,
    control_indices: np.ndarray,
    obs_layer: str | None,
) -> tuple[ad.AnnData, ad.AnnData | None]:
    if pred_eval is None:
        return obs_eval, pred_eval

    obs_has_control = bool(np.any(obs_eval.obs["perturbation"].to_numpy(dtype=np.int32, copy=False) == -1))
    pred_has_control = bool(np.any(pred_eval.obs["perturbation"].to_numpy(dtype=np.int32, copy=False) == -1))

    if obs_has_control and pred_has_control:
        return obs_eval, pred_eval

    control_adata = adata[control_indices, :].copy()

    if not obs_has_control:
        obs_ctrl = control_adata.copy()
        obs_ctrl.obs_names = [f"{name}__eval_obs_ctrl" for name in obs_ctrl.obs_names]
        obs_eval = ad.concat([obs_eval, obs_ctrl], join="inner", merge="same")

    if not pred_has_control:
        pred_ctrl = control_adata.copy()
        pred_ctrl.obs_names = [f"{name}__eval_pred_ctrl" for name in pred_ctrl.obs_names]

        if obs_layer is None:
            source = pred_ctrl.X
        else:
            if obs_layer not in pred_ctrl.layers:
                raise KeyError(
                    f"obs_layer='{obs_layer}' not found when adding controls to predictions. "
                    f"Available layers: {list(pred_ctrl.layers.keys())}"
                )
            source = pred_ctrl.layers[obs_layer]

        if sparse.issparse(source):
            pred_ctrl.layers["scvi_normalized"] = source.astype(np.float32)
        else:
            pred_ctrl.layers["scvi_normalized"] = np.asarray(source, dtype=np.float32)

        pred_eval = ad.concat([pred_eval, pred_ctrl], join="inner", merge="same")

    return obs_eval, pred_eval


def _dataset_sparsity(adata_obj: ad.AnnData) -> float:
    matrix = adata_obj.X
    total_entries = int(adata_obj.n_obs) * int(adata_obj.n_vars)
    if total_entries == 0:
        return float("nan")

    if sparse.issparse(matrix):
        nonzero = int(matrix.nnz)
    else:
        nonzero = int(np.count_nonzero(np.asarray(matrix)))
    return 1.0 - (nonzero / total_entries)


def run_one_trial(
    adata: ad.AnnData,
    splitter: ContextSplitter,
    trial_id: int,
    cell_line_key: str,
    counts_layer: str | None,
    obs_layer: str | None,
) -> list[dict[str, Any]]:
    train_idx, val_idx, test_idx = splitter.split(seed=trial_id)

    labels = adata.obs["perturbation"].to_numpy(dtype=np.int32, copy=False)
    control_train_idx = train_idx[labels[train_idx] == -1]
    if control_train_idx.size == 0:
        raise ValueError("No control cells found in training split.")

    ad_test_only = adata[test_idx, :]
    test_perturbation_ids = get_eval_perturbation_ids(obs=ad_test_only, control_label=-1, strict_match=False)
    test_perturbation_ids = np.asarray(test_perturbation_ids, dtype=np.int32)
    if test_perturbation_ids.size == 0:
        raise ValueError("No non-control perturbations found in test split.")

    train_val_idx = np.concatenate([train_idx, val_idx])
    test_eval_idx = np.concatenate([control_train_idx, test_idx])
    ad_train_val = adata[train_val_idx, :]
    ad_test_eval = adata[test_eval_idx, :]

    mu_control_train, mu_pool_train, _ = get_pseudobulks_and_degs(
        ac_view=ad_train_val,
        return_degs=False,
        layer_key=obs_layer,
    )
    mu_control_test, mu_pool_test, degs_test = get_pseudobulks_and_degs(
        ac_view=ad_test_eval,
        return_degs=True,
        alpha=0.05,
        method="t-test",
        layer_key=obs_layer,
    )

    mu_obs = compute_means_by_perturbation(
        adata_view=ad_test_eval,
        perturbation_ids=test_perturbation_ids,
        layer_key=obs_layer,
        missing_group_context="evaluation view",
    )
    train_perturbation_ids = get_eval_perturbation_ids(obs=ad_train_val, control_label=-1, strict_match=False)
    train_perturbation_ids = np.asarray(train_perturbation_ids, dtype=np.int32)
    if train_perturbation_ids.size == 0:
        raise ValueError("No non-control perturbations found in train/val split.")

    mu_train = compute_means_by_perturbation(
        adata_view=ad_train_val,
        perturbation_ids=train_perturbation_ids,
        layer_key=obs_layer,
        missing_group_context="evaluation view",
    )
    pert_id_to_label = _build_perturbation_label_map(adata)
    gene_name_set = set(adata.var_names.astype(str))
    train_targets = [_label_to_target_tokens(pert_id_to_label[int(pid)]) for pid in train_perturbation_ids]
    test_targets = [_label_to_target_tokens(pert_id_to_label[int(pid)]) for pid in test_perturbation_ids]

    trial_results: list[dict[str, Any]] = []
    for model in _MODELS:
        start_time = time.time()
        mu_pred = None
        ad_test_pred = None

        if model == "Control":
            mu_pred = np.tile(mu_control_train, (test_perturbation_ids.size, 1))
        elif model == "Average":
            mu_pred = np.tile(mu_pool_train, (test_perturbation_ids.size, 1))
        elif model == "linearPCA":
            # Fallback for unsupported perturbations is Average baseline.
            mu_pred = np.tile(mu_pool_train, (test_perturbation_ids.size, 1))

            train_valid_mask = np.array(
                [_has_known_target(tokens, gene_name_set) for tokens in train_targets],
                dtype=bool,
            )
            test_valid_mask = np.array(
                [_has_known_target(tokens, gene_name_set) for tokens in test_targets],
                dtype=bool,
            )

            if np.any(train_valid_mask) and np.any(test_valid_mask):
                train_targets_valid = [train_targets[i] for i in np.flatnonzero(train_valid_mask)]
                test_targets_valid = [test_targets[i] for i in np.flatnonzero(test_valid_mask)]
                n_train_valid = int(np.sum(train_valid_mask))
                n_test_valid = int(np.sum(test_valid_mask))

                linear_input = np.vstack(
                    [
                        mu_train[train_valid_mask, :],
                        np.zeros((n_test_valid, adata.n_vars), dtype=np.float64),
                    ]
                )
                linear_targets = [*train_targets_valid, *test_targets_valid]
                train_idx_linear = np.arange(n_train_valid, dtype=int)
                test_idx_linear = np.arange(n_train_valid, n_train_valid + n_test_valid, dtype=int)

                linear_model = PseudoPCALinear(
                    pseudobulk=linear_input,
                    target_genes=linear_targets,
                    gene_names=adata.var_names.to_numpy(),
                    train_idx=train_idx_linear,
                    test_idx=test_idx_linear,
                    seed=trial_id,
                )
                mu_pred_valid = linear_model.run()
                mu_pred[test_valid_mask, :] = mu_pred_valid
        elif model == "scVI":
            scvi_model = ScviPerturbation(
                data=adata,
                counts_layer=counts_layer,
                perturbation_key="perturbation",
                cell_line_key=cell_line_key,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                seed=trial_id,
            )
            ad_test_pred = scvi_model.run(
                n_latent=10,
                n_hidden=128,
                n_layers=3,
                gene_likelihood="zinb",
                dispersion="gene",
                max_epochs=800,
                batch_size=512,
                early_stopping=bool(val_idx.size > 0),
                dataloader_num_workers=0,
            )
        else:
            raise NotImplementedError(f"Model '{model}' is not implemented.")

        obs_for_eval, pred_for_eval = _ensure_control_cells_for_distribution_eval(
            adata=adata,
            obs_eval=ad_test_eval,
            pred_eval=ad_test_pred,
            control_indices=control_train_idx,
            obs_layer=obs_layer,
        )

        model_metrics = evaluation(
            pred=pred_for_eval,
            obs=obs_for_eval,
            mu_obs=mu_obs,
            mu_pred=mu_pred,
            mu_control_obs=mu_control_test,
            mu_pool_obs=mu_pool_test,
            true_DEGs=None, # Unknown for real datasets
            DEGs_stats=degs_test,
            perturbation_ids=test_perturbation_ids,
            model=model,
            layer_obs=obs_layer,
            control_label=-1,
        )

        model_metrics.update(
            {
                "model": model,
                "trial_id": int(trial_id),
                "status": "success",
                "execution_time": time.time() - start_time,
            }
        )
        trial_results.append(model_metrics)

    return trial_results


def run_real_experiments(
    dataset_name: str,
    dataset_path: str,
    output_dir: str,
    n_trials: int,
    perturbation_key: str,
    cell_line_key: str,
    control_label: str,
    counts_layer: str | None,
    obs_layer: str | None,
    split_strategy: str,
    norm_target_sum: float,
) -> str:
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = os.path.join(output_dir, f"real_experiment_results_{dataset_name}_{timestamp}.csv")
    error_log_file = os.path.join(output_dir, f"error_log_{dataset_name}_{timestamp}.txt")

    adata, label_mapping = load_real_dataset(
        dataset_path=dataset_path,
        perturbation_key=perturbation_key,
        cell_line_key=cell_line_key,
        control_label=control_label,
    )

    if counts_layer is not None and counts_layer not in adata.layers:
        raise KeyError(
            f"Requested counts_layer='{counts_layer}' not found. Available layers: {list(adata.layers.keys())}"
        )

    if obs_layer is not None and obs_layer not in adata.layers:
        if obs_layer == _NORM_LAYER_KEY:
            source_layer = counts_layer
            if source_layer is None and "counts" in adata.layers:
                source_layer = "counts"
            _ensure_normalized_log1p_layer(
                adata=adata,
                output_layer_key=obs_layer,
                source_layer=source_layer,
                target_sum=norm_target_sum,
            )
        else:
            raise KeyError(
                f"Requested obs_layer='{obs_layer}' not found. Available layers: {list(adata.layers.keys())}"
            )

    print(f"Loaded dataset '{dataset_name}' from {dataset_path}")
    print(f"Shape: cells={adata.n_obs}, genes={adata.n_vars}")
    print(f"Cell lines: {adata.obs[cell_line_key].nunique()} unique ({cell_line_key})")
    print(f"Perturbations (non-control): {len(label_mapping)}")
    print(f"Running {n_trials} trials sequentially (no multiprocessing).")

    n_cell_lines = int(adata.obs[cell_line_key].nunique())
    n_total_perturbations = int(len(label_mapping))
    data_sparsity = _dataset_sparsity(adata)

    metric_columns = [
        "pearson",
        "pearson_degs",
        "mae",
        "mae_degs",
        "mse",
        "mse_degs",
        "r2",
        "r2_degs",
        "parametric_distance",
        "mmd_distance",
        "vendi_score_pred",
        "vendi_score_obs",
        "pds_l1",
        "pds_l2",
        "pds_cosine",
    ]

    # Prepare splitter
    data_proxy = AnnDataProxy(adata.obs)
    if dataset_name == "norman19":
        # Can only do in-context split
        context_mode = None
        holdout_context_values = None
    elif dataset_name == "replogle22":
        if split_strategy == "in-context":
            context_mode = None
            holdout_context_values = None
        elif split_strategy == "cross-context":
            context_mode = "cell"
            holdout_context_values = ["K562"]
        else:
            raise ValueError(f"Invalid split_strategy: {split_strategy}")
    elif dataset_name == "CD4+":
        if split_strategy == "in-context":
            context_mode = None
            holdout_context_values = None
        elif split_strategy == "cross-context":
            context_mode = "donor"
            holdout_context_values = ["D3", "D4"]
        else:
            raise ValueError(f"Invalid split_strategy: {split_strategy}")
    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")
    splitter = ContextSplitter(
        adata=data_proxy,
        train_frac=0.6,
        val_frac=0.2,
        test_frac=0.2,
        context_mode=context_mode,
        perturbation_key=perturbation_key,
        cell_line_key=cell_line_key,
        donor_key="donor",
        holdout_context_values=holdout_context_values,
        control_label=-1,
    )

    all_rows: list[dict[str, Any]] = []
    for trial_id in range(int(n_trials)):
        print(f"Trial {trial_id + 1}/{n_trials})")
        try:
            trial_rows = run_one_trial(
                adata=adata,
                splitter=splitter,
                trial_id=trial_id,
                cell_line_key=cell_line_key,
                counts_layer=counts_layer,
                obs_layer=obs_layer,
            )
            for row in trial_rows:
                row.update(
                    {
                        "dataset": dataset_name,
                        "dataset_path": dataset_path,
                        "n_cells": int(adata.n_obs),
                        "n_genes": int(adata.n_vars),
                        "n_cell_lines": n_cell_lines,
                        "n_total_perturbations": n_total_perturbations,
                        "sparsity": data_sparsity,
                    }
                )
            all_rows.extend(trial_rows)
        except Exception as exc:
            error_row = {
                "dataset": dataset_name,
                "dataset_path": dataset_path,
                "trial_id": int(trial_id),
                "status": "failed",
                "error": str(exc),
                "n_cells": int(adata.n_obs),
                "n_genes": int(adata.n_vars),
                "n_cell_lines": n_cell_lines,
                "n_total_perturbations": n_total_perturbations,
                "sparsity": data_sparsity,
                "execution_time": np.nan,
            }
            for model in _MODELS:
                row = error_row.copy()
                row["model"] = model
                for metric_key in metric_columns:
                    row[metric_key] = np.nan
                all_rows.append(row)

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(csv_file, index=False)

    if "status" in results_df.columns:
        failed_df = results_df[results_df["status"] == "failed"]
        if not failed_df.empty:
            with open(error_log_file, "w", encoding="utf-8") as handle:
                for _, row in failed_df.iterrows():
                    handle.write(f"Trial {int(row['trial_id']) + 1} ({row['model']}) failed\n")
                    handle.write(f"Error: {row.get('error', 'Unknown error')}\n")
                    handle.write("-" * 80 + "\n")
            print(f"Some trials failed. Error log: {error_log_file}")

    success_trials = 0
    if "status" in results_df.columns and "model" in results_df.columns:
        success_trials = results_df[
            (results_df["status"] == "success") & (results_df["model"] == _MODELS[0])
        ].shape[0]
    failed_trials = int(n_trials) - int(success_trials)

    print(f"Done. Results saved to: {csv_file}")
    print(f"Success: {success_trials}/{n_trials} trials")
    print(f"Failed: {failed_trials}/{n_trials} trials")
    return csv_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run real-world perturbation experiments with perturbation-level split by cell type."
    )
    parser.add_argument("--dataset_name", type=str, default="norman19")
    parser.add_argument("--dataset_path", type=str, default="data/norman19/norman19_processed.h5ad")
    parser.add_argument("--output_dir", type=str, default="results/real_experiments")
    parser.add_argument("--n_trials", type=int, default=10)

    parser.add_argument("--perturbation_key", type=str, default="perturbation")
    parser.add_argument("--cell_line_key", type=str, default="cell_line")
    parser.add_argument("--control_label", type=str, default="control")

    parser.add_argument("--counts_layer", type=str, default="counts", help=f"Layer containing raw counts for scVI. Set to 'none' to use adata.X as counts.")
    parser.add_argument("--obs_layer", type=str, default=_NORM_LAYER_KEY, help=f"Layer to use for evaluation and modeling.")
    parser.add_argument("--norm_target_sum", type=float, default=1e4, help="Target sum for normalization when obs_layer is missing and needs to be computed from counts.")

    parser.add_argument("--split_strategy", type=str, default="in-context", choices=["in-context", "cross-context"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    counts_layer = None if str(args.counts_layer).lower() == "none" else args.counts_layer
    obs_layer = None if str(args.obs_layer).lower() == "none" else args.obs_layer

    run_real_experiments(
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        n_trials=int(args.n_trials),
        perturbation_key=args.perturbation_key,
        cell_line_key=args.cell_line_key,
        control_label=args.control_label,
        counts_layer=counts_layer,
        obs_layer=obs_layer,
        split_strategy=args.split_strategy,
        norm_target_sum=float(args.norm_target_sum),
    )


if __name__ == "__main__":
    main()

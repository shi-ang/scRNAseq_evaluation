from __future__ import annotations

from typing import Any

import numpy as np

from metrics.perturbation_effect.pearson import pearson_pert
from metrics.perturbation_effect.perturbation_discrimination_score import pds
from metrics.perturbation_effect.r_square import r2_score_pert
from metrics.reconstruction.distribution_distance import distribution_distance
from metrics.reconstruction.mean_error import mean_error_pert
from metrics.reconstruction.vendi_score import vendi_score
from util.anndata_util import obs_has_key


def _to_vector(x: Any) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim > 1:
        arr = arr.ravel()
    return arr


def _sort_labels(labels: set[Any]) -> np.ndarray:
    try:
        ordered = sorted(labels)
    except TypeError:
        ordered = sorted(labels, key=lambda x: str(x))
    return np.asarray(ordered)


def get_eval_perturbation_ids(
    obs: Any = None,
    pred: Any = None,
    control_label: str | int | float = -1,
    strict_match: bool = True,
) -> np.ndarray:
    """Return sorted non-control perturbation IDs from obs/pred objects."""
    per_source: dict[str, set[Any]] = {}

    for name, data_obj in (("obs", obs), ("pred", pred)):
        if data_obj is None:
            continue
        obs_df = getattr(data_obj, "obs", None)
        if not obs_has_key(obs_df, "perturbation"):
            continue
        labels = np.asarray(data_obj.obs["perturbation"])
        per_source[name] = set(np.unique(labels)) - {control_label}

    if not per_source:
        return np.asarray([], dtype=np.int32)

    if strict_match and len(per_source) >= 2:
        value_sets = list(per_source.values())
        reference = value_sets[0]
        for current in value_sets[1:]:
            if current != reference:
                raise ValueError(
                    "obs and pred have mismatched perturbation labels (excluding control). "
                    f"obs_only={sorted(per_source.get('obs', set()) - per_source.get('pred', set()))}, "
                    f"pred_only={sorted(per_source.get('pred', set()) - per_source.get('obs', set()))}."
                )
        return _sort_labels(reference)

    merged: set[Any] = set()
    for ids in per_source.values():
        merged.update(ids)
    return _sort_labels(merged)


def evaluation(
    pred,
    obs,
    mu_pred,
    mu_obs,
    mu_control_obs,
    mu_pool_obs,
    DEGs_stats,
    perturbation_ids: np.ndarray | None = None,
    model: str = "Average",
    layer_obs: str | None = "normalized_log1p",
    control_label: str | int | float = -1,
    use_pca_for_mmd: bool = True,
):
    """Evaluate predicted perturbation profiles for synthetic or real datasets."""
    if perturbation_ids is None:
        perturbation_ids = get_eval_perturbation_ids(
            obs=obs,
            pred=pred,
            control_label=control_label,
            strict_match=bool(obs is not None and pred is not None),
        )

    n_perts = int(mu_obs.shape[0])
    if perturbation_ids.size == 0:
        raise ValueError("No non-control perturbation IDs found for evaluation.")
    if perturbation_ids.size != n_perts:
        raise ValueError(
            f"Length mismatch: perturbation_ids={perturbation_ids.size}, mu_obs rows={n_perts}."
        )
    if len(DEGs_stats) != n_perts:
        raise ValueError(
            f"Length mismatch: DEGs_stats={len(DEGs_stats)}, expected {n_perts}."
        )

    layer_pred: str | None = None
    if mu_pred is None:
        if pred is None:
            raise ValueError("pred must be provided when mu_pred is None.")
        if "perturbation" not in pred.obs.columns:
            raise KeyError("pred.obs must contain 'perturbation' when mu_pred is None.")

        if model == "scVI":
            layer_pred = "scvi_normalized"
        else:
            raise NotImplementedError(
                f"Model '{model}' is not implemented for on-the-fly pseudobulk calculation when mu_pred is None."
            )

        mu_pred = np.empty((n_perts, pred.shape[1]), dtype=np.float32)
        pred_labels = np.asarray(pred.obs["perturbation"])

        for idx, pert_id in enumerate(perturbation_ids):
            pert_mask = pred_labels == pert_id
            if int(np.sum(pert_mask)) == 0:
                raise ValueError(f"No cells found for perturbation id {pert_id!r} in pred.")
            mu_pred[idx, :] = _to_vector(pred.layers[layer_pred][pert_mask, :].mean(axis=0)).astype(
                np.float32,
                copy=False,
            )

    if pred is not None and obs is not None:
        parametric_distance = distribution_distance(
            obs=obs,
            pred=pred,
            layer_obs=layer_obs,
            layer_pred=layer_pred,
            control_label=control_label,
            method="parametric",
            distribution_form="NB",
            dist_type="JS-divergence",
            use_pca=False,
        )

        obs_has_control = bool(np.any(np.asarray(obs.obs["perturbation"]) == control_label))
        mmd_distance = distribution_distance(
            obs=obs,
            pred=pred,
            layer_obs=layer_obs,
            layer_pred=layer_pred,
            control_label=control_label,
            method="mmd",
            distribution_form="NB",
            use_pca=bool(use_pca_for_mmd and obs_has_control),
        )

        pred_has_control = bool(np.any(np.asarray(pred.obs["perturbation"]) == control_label))
        if pred_has_control:
            vendi_score_pred = vendi_score(
                ac=pred,
                n_pca_components=30,
                layer_key=layer_pred,
                control_label=control_label,
            )
        else:
            vendi_score_pred = np.nan
    else:
        parametric_distance = np.nan
        mmd_distance = np.nan
        vendi_score_pred = np.nan

    pds_l1_score = pds(
        X_obs=mu_obs,
        X_pred=mu_pred,
        reference=mu_control_obs,
        metric="l1",
    )
    pds_l2_score = pds(
        X_obs=mu_obs,
        X_pred=mu_pred,
        reference=mu_control_obs,
        metric="l2",
    )
    pds_cosine_score = pds(
        X_obs=mu_obs,
        X_pred=mu_pred,
        reference=mu_control_obs,
        metric="cosine",
    )

    tracker = {
        "pearson": [],
        "pearson_degs": [],
        "mae": [],
        "mae_degs": [],
        "mse": [],
        "mse_degs": [],
        "r2": [],
        "r2_degs": [],
    }

    for idx in range(n_perts):
        degs = DEGs_stats[idx]
        mu_obs_ptb = mu_obs[idx].astype(np.float32, copy=False)
        mu_pred_ptb = mu_pred[idx].astype(np.float32, copy=False)

        if model != "Control":
            tracker["pearson"].append(
                pearson_pert(mu_obs_ptb, mu_pred_ptb, reference=mu_control_obs)
            )
            tracker["pearson_degs"].append(
                pearson_pert(mu_obs_ptb, mu_pred_ptb, reference=mu_control_obs, DEGs=degs)
            )

        tracker["mae"].append(mean_error_pert(mu_obs_ptb, mu_pred_ptb, type="absolute"))
        tracker["mse"].append(mean_error_pert(mu_obs_ptb, mu_pred_ptb, type="squared"))
        tracker["mae_degs"].append(
            mean_error_pert(mu_obs_ptb, mu_pred_ptb, type="absolute", weights=degs.astype(np.float32))
        )
        tracker["mse_degs"].append(
            mean_error_pert(mu_obs_ptb, mu_pred_ptb, type="squared", weights=degs.astype(np.float32))
        )
        tracker["r2"].append(
            r2_score_pert(mu_obs_ptb, mu_pred_ptb, reference=mu_control_obs)
        )
        tracker["r2_degs"].append(
            r2_score_pert(mu_obs_ptb, mu_pred_ptb, reference=mu_control_obs, weights=degs.astype(np.float32))
        )

    return {
        "pearson": np.nanmedian(tracker["pearson"]) if tracker["pearson"] else np.nan,
        "pearson_degs": np.nanmedian(tracker["pearson_degs"]) if tracker["pearson_degs"] else np.nan,
        "mae": np.nanmedian(tracker["mae"]) if tracker["mae"] else np.nan,
        "mae_degs": np.nanmedian(tracker["mae_degs"]) if tracker["mae_degs"] else np.nan,
        "mse": np.nanmedian(tracker["mse"]) if tracker["mse"] else np.nan,
        "mse_degs": np.nanmedian(tracker["mse_degs"]) if tracker["mse_degs"] else np.nan,
        "r2": np.nanmedian(tracker["r2"]) if tracker["r2"] else np.nan,
        "r2_degs": np.nanmedian(tracker["r2_degs"]) if tracker["r2_degs"] else np.nan,
        "parametric_distance": parametric_distance,
        "mmd_distance": mmd_distance,
        "vendi_score_test": vendi_score_pred,
        "pds_l1": pds_l1_score,
        "pds_l2": pds_l2_score,
        "pds_cosine": pds_cosine_score,
    }

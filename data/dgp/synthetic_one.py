from typing import Tuple
import numpy as np
import pandas as pd

from .util import ChunkedAnnDataWriter, sample_nb_counts


def synthetic_DGP(
    G,   # number of genes
    N0,   # number of control cells
    Nk,     # number of perturbed cells per perturbation
    P,       # number of perturbations
    p_effect,  # a threshold for fraction of genes affected per perturbation
    effect_factor,  # effect factor for affected genes, epsilon in the paper
    B,      # global perturbation bias factor, beta in the paper
    mu_l,   # mean of log library size
    all_theta, # Theta parameter for all cells , size of total number of genes in the real dataset (>= G)
    control_mu, # Control mu parameters, size of total number of genes in the real dataset (>= G)
    pert_mu, # Perturbed mu parameters, size of total number of genes in the real dataset (>= G)
    output_dir, # Directory to persist temporary chunked h5ad files
    seed=None, # Optional for seed,
    max_cells_per_chunk=2048,
    normalize=True, # Whether to normalize before log1p for the persisted layer
    normalized_layer_key: str = "normalized_log1p", # Layer name for normalized/log1p values
) -> Tuple[list[str], list[np.ndarray]]:
    """
    Generate synthetic counts and persist them in chunked .h5ad files.
    Each chunk stores:
      - .X: raw counts
      - .layers[normalized_layer_key]: normalized/log1p representation
    Returns:
      - chunk_paths: list[str], h5ad files containing sparse count chunks
      - all_affected_masks: list[np.ndarray], one mask per perturbation
    """
    rng = np.random.default_rng(42 if seed is None else seed)
    
    # --- Parameter Preparation with assertions ---
    # Assert that inputs are already arrays
    assert isinstance(control_mu, np.ndarray), "control_mu must be a numpy array"
    assert isinstance(pert_mu, np.ndarray), "pert_mu must be a numpy array"
    assert isinstance(all_theta, np.ndarray), "all_theta must be a numpy array"
    # Assert that they have the same length
    assert len(control_mu) == len(all_theta), "control_mu and all_theta must have the same length."
    assert len(control_mu) == len(pert_mu), "control_mu and pert_mu must have the same length."
    # Assert that G is not larger than the provided arrays
    assert len(control_mu) >= G, f"G parameter ({G}) cannot be larger than the length of provided arrays ({len(control_mu)})"
    # --- End of assertions ---
    
    # Sample G elements from control_mu, all_theta, and pert_mu to define the local parameters for selected genes
    indices = rng.choice(len(control_mu), size=G, replace=False)
    local_control_mu = control_mu[indices]
    local_all_theta = all_theta[indices]  # Use the all-cells theta
    local_pert_mu = pert_mu[indices]

    var = pd.DataFrame(index=pd.Index([f"gene_{i}" for i in range(G)], name="gene"))

    all_affected_masks = []
    writer = ChunkedAnnDataWriter(
        output_dir=output_dir,
        var=var,
        max_cells_per_chunk=max_cells_per_chunk,
        normalize=normalize,
        normalized_layer_key=normalized_layer_key,
    )

    # 1. Sample control cells with bias (B, dispersion set to all_theta from all cells, fixed dispersion assumption)
    control_cells_remaining = int(N0)
    while control_cells_remaining > 0:
        current_batch_size = min(max_cells_per_chunk, control_cells_remaining)
        lib_size_control = rng.lognormal(
            mean=mu_l, sigma=0.1714, size=current_batch_size
        )  # 0.1714 from all cells of the Norman19 dataset
        control_counts = sample_nb_counts(
            mean=local_control_mu, l_c=lib_size_control, theta=local_all_theta, rng=rng
        )
        writer.append_counts(control_counts, perturbation_id=-1, cell_line=0)
        control_cells_remaining -= current_batch_size

    # Define global perturbation bias, this is the terms in brackets for eq 2 in the paper
    delta_b = local_pert_mu - local_control_mu
    local_pert_mu_biased = np.clip(local_control_mu + B * delta_b, 0.0, np.inf)

    # 2. For each perturbation generate the cells
    for perturbation_id in range(P):
        # this is eq 4 in the paper
        affected_mask_loop = rng.random(G) < p_effect
        all_affected_masks.append(affected_mask_loop)

        mu_k_loop = local_pert_mu_biased.copy()
        if affected_mask_loop.sum() > 0:
            effect_directions = rng.choice([effect_factor, 1.0/effect_factor], size=affected_mask_loop.sum())   # alpha in Eq 2 and 4
            mu_k_loop[affected_mask_loop] *= effect_directions

        remaining = int(Nk)
        while remaining > 0:
            current_batch_size = min(max_cells_per_chunk, remaining)
            lib_size_pert = rng.lognormal(
                mean=mu_l, sigma=0.1714, size=current_batch_size
            )  # 0.1714 from all cells of the Norman19 dataset
            pert_counts = sample_nb_counts(
                mean=mu_k_loop, l_c=lib_size_pert, theta=local_all_theta, rng=rng
            )
            writer.append_counts(pert_counts, perturbation_id=perturbation_id, cell_line=0)
            remaining -= current_batch_size

    writer.flush()
    return writer.chunk_paths, all_affected_masks

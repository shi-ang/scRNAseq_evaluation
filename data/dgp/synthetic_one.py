from typing import Tuple, List
import numpy as np
from scipy import sparse
import pandas as pd

from .util import ChunkedAnnDataWriter


def _sample_nb_counts(mean, l_c, theta, rng): # theta kept as generic parameter name for this utility function
    """
    Generate individual cell profiles from NB distribution
    Returns an array of shape (len(l_c), G)
    """
    # Ensure mean and theta are numpy arrays for element-wise operations
    mean_arr = np.asarray(mean) # size of genes
    theta_arr = np.asarray(theta) # size of genes
    l_c_arr = np.asarray(l_c) # size of cells

    # Correct mean for library size
    lib_size_corrected_mean = np.outer(l_c_arr, mean_arr)

    # Prevent division by zero or negative p if theta + mean is zero or mean is much larger than theta
    # This can happen if means are very low and theta is also low.
    # Add a small epsilon to the denominator to stabilize.
    # Also ensure p is within (0, 1)
    p_denominator = theta_arr + lib_size_corrected_mean
    p_denominator[p_denominator <= 0] = 1e-9 # Avoid zero or negative denominator
    
    p = theta_arr / p_denominator
    p = np.clip(p, 1e-9, 1 - 1e-9) # Ensure p is in a valid range for negative_binomial

    # Negative binomial expects n (number of successes, our theta) to be > 0.
    # And p (probability of success) to be in [0, 1].
    # If theta contains zeros or negatives, np.random.negative_binomial will fail.
    # Assuming theta values are appropriate (positive).

    predicted_counts = rng.negative_binomial(theta_arr, p)
    return sparse.csr_matrix(predicted_counts)


def synthetic_DGP(
    G=10_000,   # number of genes
    N0=3_000,   # number of control cells
    Nk=150,     # number of perturbed cells per perturbation
    P=50,       # number of perturbations
    p_effect=0.01,  # a threshold for fraction of genes affected per perturbation
    effect_factor=2.0,  # effect factor for affected genes, epsilon in the paper
    B=0.0,      # global perturbation bias factor, beta in the paper
    mu_l=1.0,   # mean of log library size
    all_theta=None, # Theta parameter for all cells , size of total number of genes in the real dataset (>= G)
    control_mu=None, # Control mu parameters, size of total number of genes in the real dataset (>= G)
    pert_mu=None, # Perturbed mu parameters, size of total number of genes in the real dataset (>= G)
    trial_id_for_rng=None, # Optional for seeding RNG per trial,
    output_dir=None, # Directory to persist temporary chunked h5ad files
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
    if trial_id_for_rng is None:
        rng = np.random.default_rng(42)
    else:
        rng = np.random.default_rng(trial_id_for_rng)
    
    # --- Parameter Preparation with assertions ---
    # Assert that control_mu, pert_mu, and all_theta are provided
    assert control_mu is not None, "control_mu must be provided. None value is not allowed."
    assert pert_mu is not None, "pert_mu must be provided. None value is not allowed."
    assert all_theta is not None, "all_theta must be provided. None value is not allowed."
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
    
    # Sample G elements from control_mu and all_theta
    indices = rng.choice(len(control_mu), size=G, replace=False)
    local_control_mu = control_mu[indices]
    local_all_theta = all_theta[indices]  # Use the all-cells theta
    local_pert_mu = pert_mu[indices]

    var = pd.DataFrame(index=pd.Index([f"gene_{i}" for i in range(G)], name="gene"))

    if output_dir is None:
        raise ValueError("output_dir must be provided.")

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
        control_counts = _sample_nb_counts(
            mean=local_control_mu, l_c=lib_size_control, theta=local_all_theta, rng=rng
        )
        writer.append_counts(control_counts, perturbation_id=-1)
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
            pert_counts = _sample_nb_counts(
                mean=mu_k_loop, l_c=lib_size_pert, theta=local_all_theta, rng=rng
            )
            writer.append_counts(pert_counts, perturbation_id=perturbation_id)
            remaining -= current_batch_size

    writer.flush()
    return writer.chunk_paths, all_affected_masks

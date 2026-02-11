from __future__ import annotations
from typing import Tuple
from scipy import sparse
from scipy import stats
import numpy as np
import scanpy as sc
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import IncrementalPCA
from anndata.experimental import AnnCollection


def normalize_and_log1p(matrix, normalize=True, target_sum=1e4):
    """
    Normalize each cell to target_sum (optional) and apply log1p.
    Works on dense and sparse matrices.
    """
    if sparse.issparse(matrix):
        out = matrix.tocsr(copy=True).astype(np.float32)
        if normalize:
            lib_sizes = np.asarray(out.sum(axis=1)).ravel().astype(np.float32, copy=False)
            lib_sizes[lib_sizes == 0.0] = 1.0
            scale = (target_sum / lib_sizes).astype(np.float32, copy=False)
            out = out.multiply(scale[:, None]).tocsr()
        out.data = np.log1p(out.data)
        return out

    out = np.asarray(matrix, dtype=np.float32).copy()
    if normalize:
        lib_sizes = out.sum(axis=1, dtype=np.float32)
        lib_sizes[lib_sizes == 0.0] = 1.0
        out *= (target_sum / lib_sizes)[:, None]
    np.log1p(out, out=out)
    return out


def est_cost(params):
    """
    Estimate cost for the computation, based on the number of cells and genes.
    
    :param params: Dictionary containing parameters including G, N0, Nk, and P
    """
    G, N0, Nk, P = params["G"], params["N0"], params["Nk"], params["P"]
    rows = N0 + P * Nk
    return rows * G


def systematic_variation(
        ptb_shifts: np.ndarray, 
        avg_ptb_shift: np.ndarray
) -> float:
    """
    Calculate the average cosine similarity between perturbation-specific shifts 
    and the average perturbation effect.
    
    :param ptb_shifts: perturbation shifts matrix of shape (n_perturbations, n_genes)
    :param avg_ptb_shift: average perturbation shift vector of shape (n_genes,)
    """
    similarities = cosine_similarity(ptb_shifts, avg_ptb_shift.reshape(1, -1)).flatten()
    return float(np.mean(similarities))


def intra_correlation(
        ptb_shifts: np.ndarray
) -> float:
    """
    Compute mean pairwise Pearson correlation across cells from perturbation-specific shifts.
    """
    corr_matrix = np.corrcoef(ptb_shifts)
    lower_tri_indices = np.tril_indices(corr_matrix.shape[0], k=-1)
    mean_corr = np.mean(corr_matrix[lower_tri_indices])
    return float(mean_corr)


def _pairwise_squared_distances(x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
    """Pairwise squared Euclidean distances computed from dot products."""
    x = np.asarray(x, dtype=np.float64)
    if y is None:
        y = x
    else:
        y = np.asarray(y, dtype=np.float64)

    x_norm = np.sum(x * x, axis=1, keepdims=True)
    y_norm = np.sum(y * y, axis=1, keepdims=True).T
    distances = x_norm + y_norm - 2.0 * (x @ y.T)
    np.maximum(distances, 0.0, out=distances)
    return distances


def _vendi_from_spectrum(eigenvalues: np.ndarray) -> float:
    """Vendi score from spectrum with numerical safety."""
    eig = np.asarray(eigenvalues, dtype=np.float64)
    eig = np.clip(eig, 0.0, None)
    eig_sum = float(eig.sum())
    if eig_sum <= 0.0:
        return 1.0

    eig /= eig_sum
    positive = eig[eig > 0.0]
    if positive.size == 0:
        return 1.0
    entropy = -np.sum(positive * np.log(positive))
    return float(np.exp(entropy))


def _rbf_kernel_mean(x: np.ndarray, y: np.ndarray, sigma: float) -> float:
    """Mean RBF-kernel value over all cross pairs between rows of x and y."""
    dist_sq = _pairwise_squared_distances(x, y)
    kernel = np.exp(-dist_sq / (2.0 * sigma * sigma), dtype=np.float64)
    return float(kernel.mean())


def _median_positive(values: np.ndarray) -> float:
    """Median of strictly positive values, or 1.0 when unavailable."""
    vals = np.asarray(values, dtype=np.float64)
    positive = vals[vals > 0.0]
    if positive.size == 0:
        return 1.0
    return float(np.median(positive))


def _sample_rows_from_grouped_embeddings(
    grouped_embeddings: list[np.ndarray],
    sample_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample rows uniformly without replacement across grouped arrays."""
    counts = np.array([arr.shape[0] for arr in grouped_embeddings], dtype=np.int64)
    total = int(counts.sum())
    if total == 0:
        return np.empty((0, 0), dtype=np.float64)
    if total <= sample_size:
        return np.concatenate(grouped_embeddings, axis=0)

    sampled_global = np.sort(rng.choice(total, size=int(sample_size), replace=False))
    offsets = np.concatenate(([0], np.cumsum(counts)))
    sampled_parts: list[np.ndarray] = []
    for g_idx, group_array in enumerate(grouped_embeddings):
        lo = int(offsets[g_idx])
        hi = int(offsets[g_idx + 1])
        left = int(np.searchsorted(sampled_global, lo, side="left"))
        right = int(np.searchsorted(sampled_global, hi, side="left"))
        if right <= left:
            continue
        local_idx = sampled_global[left:right] - lo
        sampled_parts.append(group_array[local_idx])

    if not sampled_parts:
        return np.empty((0, grouped_embeddings[0].shape[1]), dtype=np.float64)
    return np.concatenate(sampled_parts, axis=0)


def _get_batch_matrix(batch_view, layer_key: str | None = None):
    """Return a batch matrix from X or a requested layer."""
    if layer_key is None:
        return batch_view.X

    layers = getattr(batch_view, "layers", None)
    if layers is None:
        raise KeyError(f"Requested layer '{layer_key}' but no layers are available in the batch view.")

    try:
        return layers[layer_key]
    except KeyError as e:
        available_layers = list(layers.keys()) if hasattr(layers, "keys") else []
        raise KeyError(
            f"Requested layer '{layer_key}' not found in batch view. Available layers: {available_layers}"
        ) from e


def vendi_score(
    ac: AnnCollection,
    ac_batch_size: int = 1024,
    n_pca_components: int = 50,
    sample_size: int = 2000,
    random_state: int = 0,
    layer_key: str | None = None,
) -> float:
    """
    Compute perturbation-level Vendi score using RBF-MMD distances.

    Procedure:
    1) Fit IncrementalPCA on control cells only (perturbation == -1).
    2) Transform all perturbation cells with that same transformation.
    3) Build pairwise perturbation distance matrix via biased MMD^2 (RBF kernel).
    4) Convert distances to similarities K_ij = exp(-MMD^2_ij / (2 * sigma^2)).
    5) Normalize matrix as A = K / n_perturbations.
    6) Eigendecompose A to obtain eigenvalues.
    7) Compute von Neumann entropy H(A) = -sum_i lambda_i log(lambda_i), with 0log0 = 0.
    8) Return Vendi score VS = exp(H(A)).

    Arguments:
    - ac: AnnCollection containing the data, with .obs["perturbation"] indicating control (-1) vs perturbation groups (0, 1, 2, ...).
    - ac_batch_size: number of cells to process at a time when iterating through the AnnCollection (for memory efficiency).
    - n_pca_components: number of PCA components to use for the embedding before MMD; default 50 for a balance of expressiveness and stability.
    - sample_size: number of rows to sample for estimating sigma_rbf and sigma_transform; default 2000 for a balance of speed and stability.
    - random_state: random seed for reproducibility of sampling.
    - layer_key: if provided, use this layer instead of `.X`.
    """
    obs_pert = ac.obs["perturbation"].to_numpy(dtype=np.int32, copy=False)
    control_idx = np.flatnonzero(obs_pert == -1)
    perturbation_ids = np.sort(np.unique(obs_pert[obs_pert != -1])).astype(np.int32, copy=False)
    n_perturbations = perturbation_ids.size

    rng = np.random.default_rng(random_state)

    # Step 1: fit IncrementalPCA on control cells only.
    # We force fit_batch_size == ac_batch_size (no general buffering).
    # Only the potentially short final batch gets special handling.
    ac_control = ac[control_idx]
    n_control = int(control_idx.size)
    n_pca_components = min(n_pca_components, n_control, ac_control.n_vars)
    fit_batch_size = int(ac_batch_size)
    if fit_batch_size < n_pca_components:
        raise ValueError(f"ac_batch_size ({fit_batch_size}) must be >= n_pca_components ({n_pca_components}) for IncrementalPCA partial_fit.")

    pca_model = IncrementalPCA(n_components=n_pca_components, batch_size=fit_batch_size)
    tail_batch: np.ndarray | None = None
    fitted_once = False

    for batch_view, _ in ac_control.iterate_axis(batch_size=fit_batch_size, axis=0, shuffle=False):
        batch = _get_batch_matrix(batch_view, layer_key=layer_key)
        if sparse.issparse(batch):
            batch = batch.toarray()
        batch = np.asarray(batch, dtype=np.float64)
        if batch.shape[0] == 0:
            continue

        # partial_fit requires at least n_components rows; defer a tiny final batch.
        if batch.shape[0] < n_pca_components:
            if tail_batch is None:
                tail_batch = batch
            else:
                tail_batch = np.concatenate([tail_batch, batch], axis=0)
            continue

        pca_model.partial_fit(batch)
        fitted_once = True

    # The leftover tail batch (if any) gets one final partial_fit, with padding if needed to reach n_components.
    if tail_batch is not None and tail_batch.shape[0] > 0:
        if tail_batch.shape[0] < n_pca_components:
            pad_needed = n_pca_components - tail_batch.shape[0]
            pad = tail_batch[np.arange(pad_needed) % tail_batch.shape[0]]
            tail_batch = np.concatenate([tail_batch, pad], axis=0)
        pca_model.partial_fit(tail_batch)
        fitted_once = True

    if not fitted_once:
        raise ValueError("IncrementalPCA failed to fit on control cells.")

    # Step 2: apply the control-fitted IncrementalPCA model to all perturbation cells.
    # Collect one embedding matrix per perturbation ID.
    pert_embeddings_chunks: dict[int, list[np.ndarray]] = {
        int(ptb): [] for ptb in perturbation_ids.tolist()
    }
    for batch_view, _ in ac.iterate_axis(batch_size=int(ac_batch_size), axis=0, shuffle=False):
        batch_labels = batch_view.obs["perturbation"].to_numpy(dtype=np.int32, copy=False)
        pert_mask = batch_labels != -1
        if not np.any(pert_mask):
            continue

        batch = _get_batch_matrix(batch_view, layer_key=layer_key)
        if sparse.issparse(batch):
            batch = batch.toarray()
        batch = np.asarray(batch, dtype=np.float32)
        batch_pert = batch[pert_mask, :]
        batch_labels_pert = batch_labels[pert_mask]

        z_batch = pca_model.transform(batch_pert).astype(np.float64, copy=False)
        for ptb in np.unique(batch_labels_pert):
            ptb_int = int(ptb)
            group_mask = batch_labels_pert == ptb
            pert_embeddings_chunks[ptb_int].append(z_batch[group_mask])

    grouped_embeddings: list[np.ndarray] = []
    for ptb in perturbation_ids.tolist():
        chunks = pert_embeddings_chunks[int(ptb)]
        if not chunks:
            raise ValueError(f"No cells found for perturbation id {int(ptb)}.")
        grouped_embeddings.append(np.concatenate(chunks, axis=0))

    # Step 3 (bandwidth part): calculate sigma for the RBF kernel in MMD.
    # Default uses the median of sampled off-diagonal Euclidean distances.
    sigma_reference = _sample_rows_from_grouped_embeddings(
        grouped_embeddings=grouped_embeddings,
        sample_size=sample_size,
        rng=rng,
    )
    if sigma_reference.shape[0] < 2:
        sigma_rbf = 1.0
    else:
        sigma_dist_sq = _pairwise_squared_distances(sigma_reference)
        tri = np.triu_indices(sigma_dist_sq.shape[0], k=1)
        sigma_dist = np.sqrt(sigma_dist_sq[tri], dtype=np.float64)
        sigma_rbf = _median_positive(sigma_dist)

    # Step 3 (distance matrix part): compute pairwise biased MMD^2 between perturbation groups.
    mmd2 = np.zeros((n_perturbations, n_perturbations), dtype=np.float64)
    self_kernel_means = np.empty(n_perturbations, dtype=np.float64)
    for i in range(n_perturbations):
        self_kernel_means[i] = _rbf_kernel_mean(
            grouped_embeddings[i], grouped_embeddings[i], sigma=sigma_rbf
        )

    for i in range(n_perturbations):
        for j in range(i + 1, n_perturbations):
            cross_mean = _rbf_kernel_mean(
                grouped_embeddings[i], grouped_embeddings[j], sigma=sigma_rbf
            )
            mmd_ij = self_kernel_means[i] + self_kernel_means[j] - 2.0 * cross_mean
            mmd_ij = float(max(mmd_ij, 0.0))
            mmd2[i, j] = mmd_ij
            mmd2[j, i] = mmd_ij

    # Step 4: convert MMD^2 distances into an RBF-like similarity matrix.
    # Default sigma_transform is the median of off-diagonal MMD^2 values.
    tri = np.triu_indices(n_perturbations, k=1)
    sigma_transform = _median_positive(mmd2[tri])
    K = np.exp(
        -mmd2 / (2.0 * sigma_transform * sigma_transform), dtype=np.float64
    )
    np.fill_diagonal(K, 1.0)

    # Step 5: normalize by number of perturbations.
    A = K / float(n_perturbations)
    # Step 6: eigendecomposition of A.
    spectrum = np.linalg.eigvalsh(A)
    return _vendi_from_spectrum(spectrum)     # Step 7 and Step 8 


def sum_and_sumsq(matrix):
    """Return per-gene sums and squared sums for dense/sparse matrices."""
    if sparse.issparse(matrix):
        gene_sum = np.asarray(matrix.sum(axis=0)).ravel().astype(np.float64, copy=False)
        gene_sumsq = np.asarray(matrix.multiply(matrix).sum(axis=0)).ravel().astype(np.float64, copy=False)
        return gene_sum, gene_sumsq

    dense_matrix = np.asarray(matrix, dtype=np.float32)
    gene_sum = dense_matrix.sum(axis=0, dtype=np.float64)
    gene_sumsq = np.square(dense_matrix, dtype=np.float64).sum(axis=0, dtype=np.float64)
    return gene_sum, gene_sumsq


def stratified_split_ac(
    ac: AnnCollection,
    obs_key: str = "perturbation",
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 0,
    shuffle_within_split: bool = True,
) -> Tuple[list[int], list[int], list[int]]:
    """
    Stratified random split of an AnnCollection into train/val/test.

    Splits are at the cell level and stratified by ac.obs[obs_key].
        - Uses only .obs to decide the split; does not touch X.
        - Subsetting an AnnCollection returns an AnnCollectionView.
        - For very small strata, it will prioritize putting at least 1 cell into train
          (and at least 1 into val/test when feasible), while keeping counts consistent.

    Arguments:
        ac: AnnCollection to split
        obs_key: Key in ac.obs to use for stratification
        train_frac: Fraction of data to use for training split
        val_frac: Fraction of data to use for validation split
        test_frac: Fraction of data to use for test split
        seed: Random seed for reproducibility
        shuffle_within_split: Whether to shuffle indices within each split
    Returns:
        idx tuples for train, val, test splits as np.ndarray of ints
    """
    fracs = np.array([train_frac, val_frac, test_frac], dtype=np.float64)
    if np.any(fracs < 0):
        raise ValueError("train/val/test fractions must be non-negative.")
    s = fracs.sum()
    fracs = fracs / s
    train_frac, val_frac, test_frac = fracs.tolist()

    # Pull labels (only obs; should be computationally cheap even for backed datasets)
    try:
        y = np.asarray(ac.obs[obs_key])
    except Exception as e:
        # Fallback: try to read from underlying adatas if exposed
        adatas = getattr(ac, "adatas", None)
        if adatas is None:
            raise KeyError(
                f"Could not access ac.obs['{obs_key}'], and AnnCollection has no .adatas fallback."
            ) from e
        y = np.concatenate([np.asarray(a.obs[obs_key]) for a in adatas], axis=0)

    n = y.shape[0]
    if n == 0:
        raise ValueError("AnnCollection has 0 observations.")

    rng = np.random.default_rng(seed)

    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []

    # get unique labels and their indices, then split each label's indices according to the fractions
    unique_labels = np.unique(y)

    for lab in unique_labels:
        idx = np.flatnonzero(y == lab)
        rng.shuffle(idx)
        m = idx.size
        if m == 0:
            continue

        # Base allocation
        n_train = int(np.floor(train_frac * m))
        n_val = int(np.floor(val_frac * m))
        n_test = m - n_train - n_val

        # Heuristics to avoid empty train when possible
        if m >= 1 and n_train == 0 and train_frac > 0:
            n_train = 1
            # take from the larger of val/test if needed
            if n_val > 0:
                n_val -= 1
            elif n_test > 0:
                n_test -= 1

        # If val requested, try to ensure val has 1 when feasible (m>=2)
        if val_frac > 0 and m >= 2 and n_val == 0:
            n_val = 1
            if n_test > 0:
                n_test -= 1
            elif n_train > 1:
                n_train -= 1

        # If test requested, try to ensure test has 1 when feasible (m>=3)
        if test_frac > 0 and m >= 3 and n_test == 0:
            n_test = 1
            # take from the largest bucket among train/val (keeping train>=1)
            if n_train > n_val and n_train > 1:
                n_train -= 1
            elif n_val > 1:
                n_val -= 1
            elif n_train > 1:
                n_train -= 1
            else:
                # can't satisfy perfectly for this small stratum; revert
                n_test = 0

        # Final safety: make sure counts sum and are non-negative
        if n_train < 0 or n_val < 0 or n_test < 0 or (n_train + n_val + n_test) != m:
            # Reset to simplest valid split: everything to train for this label
            n_train, n_val, n_test = m, 0, 0

        train_idx += idx[:n_train].tolist()
        val_idx += idx[n_train:n_train + n_val].tolist()
        test_idx += idx[n_train + n_val:].tolist()

    train_idx = np.asarray(train_idx, dtype=int)
    val_idx = np.asarray(val_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)

    if shuffle_within_split:
        rng.shuffle(train_idx)
        rng.shuffle(val_idx)
        rng.shuffle(test_idx)

    return train_idx, val_idx, test_idx,


def _fdr_bh(pvals: np.ndarray) -> np.ndarray:
    """
    Benjamini-Hochberg FDR correction. Returns adjusted p-values.

    Arguments:
        pvals: Array of p-values to adjust
    Returns:
        Array of adjusted p-values in the same shape as input
    """
    pvals = np.asarray(pvals, dtype=np.float64)
    n = pvals.size
    if n == 0:
        return pvals

    order = np.argsort(pvals)
    ranked = pvals[order]
    adjusted_ranked = ranked * n / np.arange(1, n + 1, dtype=np.float64)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted_ranked = np.clip(adjusted_ranked, 0.0, 1.0)

    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    return adjusted


def _topk_mask(pvals: np.ndarray, n_top: int) -> np.ndarray:
    """
    Boolean mask of top-k smallest p-values.

    Arguments:
        pvals: Array of p-values
        n_top: Number of top p-values to select

    Returns:
        Boolean mask array where True indicates the top-k smallest p-values
    """
    mask = np.zeros(pvals.shape[0], dtype=bool)
    n_top = int(n_top)
    if n_top <= 0:
        return mask
    n_top = min(n_top, pvals.shape[0])
    if n_top == pvals.shape[0]:
        mask[:] = True
        return mask
    top_idx = np.argpartition(pvals, n_top - 1)[:n_top]
    mask[top_idx] = True
    return mask


def get_pseudobulks_and_degs(
    ac_view,
    ac_batch_size: int = 1024,
    return_degs: bool = True,
    n_degs_per_pert: list[int] | None = None,
    alpha: float = 0.05,
    method: str = "t-test",
    layer_key: str | None = None,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """
    Get differentially expressed genes for each perturbation vs control.

    Arguments:
        ac_view: AnnCollectionView containing the data
        ac_batch_size: Number of cells to process at a time when iterating through the AnnCollectionView (for memory efficiency)
        return_degs: Whether to return DEGs or just pseudobulk means
        n_degs_per_pert: Optional list of number of DEGs to return per perturbation (if None, use all with p<alpha)
        alpha: Significance threshold for calling DEGs when n_degs_per_pert is None
        method: DE testing method to use (e.g., "t-test", "wilcoxon", etc.)
        layer_key: if provided, use this layer instead of `.X`.
    Returns:
        mean control expression vector (shape: n_genes,)
        mean pooled perturbed expression vector (shape: n_genes,)
        List of boolean numpy arrays indicating differentially expressed genes for each perturbation
    """
    n_obs = int(ac_view.n_obs)
    n_genes = int(ac_view.n_vars)
    if n_obs == 0:
        raise ValueError("ac_view has 0 observations.")
    if n_genes == 0:
        raise ValueError("ac_view has 0 genes.")

    obs_pert = ac_view.obs["perturbation"].to_numpy(dtype=np.int32, copy=False)
    observed_pert_ids = np.unique(obs_pert)
    if -1 not in observed_pert_ids:
        raise ValueError("Control group with label -1 is required in ac_view.obs['perturbation'].")

    observed_non_control_ids = observed_pert_ids[observed_pert_ids != -1]
    if observed_non_control_ids.size == 0:
        raise ValueError("No perturbation groups found (labels other than -1).")

    if n_degs_per_pert is not None:
        expected_pert_ids = np.arange(len(n_degs_per_pert), dtype=np.int32)
    else:
        expected_pert_ids = observed_non_control_ids

    # For t-test, the streaming summary-statistics path is both faster and memory-efficient.
    if method == "t-test":
        # Map perturbation IDs to row positions in the accumulators
        id_to_pos = {int(pert_id): idx for idx, pert_id in enumerate(observed_non_control_ids)}
        n_pert_observed = observed_non_control_ids.size

        # Initialize accumulators for sums, squared sums, and counts
        pert_sum = np.zeros((n_pert_observed, n_genes), dtype=np.float32)
        pert_counts = np.zeros(n_pert_observed, dtype=np.int64)
        pert_sumsq = np.zeros((n_pert_observed, n_genes), dtype=np.float32) if return_degs else None

        # Control accumulators
        control_sum = np.zeros(n_genes, dtype=np.float64)
        control_sumsq = np.zeros(n_genes, dtype=np.float64) if return_degs else None
        control_count = 0

        for start in range(0, n_obs, int(ac_batch_size)):
            ac_batch = ac_view[start : start + int(ac_batch_size)]
            batch_counts = _get_batch_matrix(ac_batch, layer_key=layer_key)
            perturbation_ids_batch = ac_batch.obs["perturbation"].to_numpy(dtype=np.int32, copy=False)

            for pert_id in np.unique(perturbation_ids_batch):
                row_mask = perturbation_ids_batch == pert_id
                batch_subset = batch_counts[row_mask]
                subset_count = int(batch_subset.shape[0])
                subset_sum, subset_sumsq = sum_and_sumsq(batch_subset)

                if pert_id == -1:
                    control_sum += subset_sum.astype(np.float64, copy=False)
                    control_count += subset_count
                    if control_sumsq is not None:
                        control_sumsq += subset_sumsq.astype(np.float64, copy=False)
                    continue

                pert_pos = id_to_pos.get(int(pert_id))
                if pert_pos is None:
                    raise ValueError(f"Unexpected perturbation id {pert_id}.")
                pert_sum[pert_pos, :] += subset_sum.astype(np.float32, copy=False)
                pert_counts[pert_pos] += subset_count
                if pert_sumsq is not None:
                    pert_sumsq[pert_pos, :] += subset_sumsq.astype(np.float32, copy=False)

        if control_count == 0:
            raise ValueError("No control cells were found in the data.")

        total_pert_cells = int(pert_counts.sum())
        if total_pert_cells == 0:
            raise ValueError("No perturbation cells were found in the data.")

        mu_control = (control_sum / control_count).astype(np.float32, copy=False)
        mu_pooled = (
            pert_sum.astype(np.float64, copy=False).sum(axis=0) / total_pert_cells
        ).astype(np.float32, copy=False)

        if not return_degs:
            return mu_control, mu_pooled, []

        if control_sumsq is None or pert_sumsq is None:
            raise RuntimeError("Internal error: variance accumulators are missing.")

        var_control = control_sumsq / control_count - np.square(mu_control.astype(np.float64, copy=False))
        var_control = np.clip(var_control, 0.0, None)
        control_std = np.sqrt(var_control)
        control_std[control_std < 1e-6] = 1e-6

        degs: list[np.ndarray] = []
        for pert_id in expected_pert_ids:
            pert_id_int = int(pert_id)
            pert_pos = id_to_pos.get(pert_id_int)
            if pert_pos is None:
                raise ValueError(
                    f"Perturbation id {pert_id_int} is missing in ac_view. "
                    "This usually means the split dropped a perturbation group."
                )
            n_cells_pert = int(pert_counts[pert_pos])
            if n_cells_pert == 0:
                raise ValueError(f"No cells found for perturbation id {pert_id_int}.")

            mu_pert = pert_sum[pert_pos, :].astype(np.float64, copy=False) / n_cells_pert
            var_pert = (
                pert_sumsq[pert_pos, :].astype(np.float64, copy=False) / n_cells_pert
                - np.square(mu_pert)
            )
            var_pert = np.clip(var_pert, 0.0, None)
            std_pert = np.sqrt(var_pert)
            std_pert[std_pert < 1e-6] = 1e-6

            _, pvals = stats.ttest_ind_from_stats(
                mean1=mu_pert,
                std1=std_pert,
                nobs1=n_cells_pert,
                mean2=mu_control,
                std2=control_std,
                nobs2=control_count,
                equal_var=False,
            )
            pvals_adj = _fdr_bh(pvals)

            if n_degs_per_pert is not None:
                if pert_id_int >= len(n_degs_per_pert):
                    raise ValueError(f"n_degs_per_pert has no entry for perturbation id {pert_id_int}.")
                deg_mask = _topk_mask(pvals_adj, n_degs_per_pert[pert_id_int])
            else:
                deg_mask = pvals_adj < float(alpha)
            degs.append(deg_mask)

        return mu_control, mu_pooled, degs

    # Non t-test methods delegate to scanpy's rank_genes_groups for semantic correctness.
    adata = ac_view.to_adata() if hasattr(ac_view, "to_adata") else ac_view.copy()
    if layer_key is not None:
        if layer_key not in adata.layers:
            available_layers = list(adata.layers.keys())
            raise KeyError(
                f"Requested layer '{layer_key}' not found in AnnData. Available layers: {available_layers}"
            )
        adata.X = adata.layers[layer_key]
    obs_pert_adata = adata.obs["perturbation"].to_numpy(dtype=np.int32, copy=False)
    control_mask = obs_pert_adata == -1
    pert_mask = obs_pert_adata != -1
    if int(control_mask.sum()) == 0:
        raise ValueError("No control cells were found in the data.")
    if int(pert_mask.sum()) == 0:
        raise ValueError("No perturbation cells were found in the data.")

    mu_control = np.asarray(adata[control_mask].X.mean(axis=0)).ravel().astype(np.float32, copy=False)
    mu_pooled = np.asarray(adata[pert_mask].X.mean(axis=0)).ravel().astype(np.float32, copy=False)

    if not return_degs:
        return mu_control, mu_pooled, []

    sorted_categories = ["-1"] + [str(int(pid)) for pid in observed_non_control_ids]
    adata.obs["perturbation"] = pd.Categorical(
        obs_pert_adata.astype(str),
        categories=sorted_categories,
        ordered=True,
    )
    sc.tl.rank_genes_groups(
        adata,
        groupby="perturbation",
        reference="-1",
        method=method,
    )

    var_index = pd.Index(adata.var_names)
    rank_names = adata.uns["rank_genes_groups"]["names"]
    rank_pvals_adj = adata.uns["rank_genes_groups"]["pvals_adj"]

    degs: list[np.ndarray] = []
    for pert_id in expected_pert_ids:
        pert_id_int = int(pert_id)
        pert_str = str(pert_id_int)
        if pert_str not in adata.obs["perturbation"].cat.categories:
            raise ValueError(
                f"Perturbation id {pert_id_int} is missing in ac_view. "
                "This usually means the split dropped a perturbation group."
            )

        names_ranked = np.asarray(rank_names[pert_str], dtype=str)
        pvals_adj_ranked = np.asarray(rank_pvals_adj[pert_str], dtype=np.float64)
        gene_idx = var_index.get_indexer(names_ranked)
        if np.any(gene_idx < 0):
            raise ValueError("rank_genes_groups returned genes not found in adata.var_names.")

        pvals_adj = np.ones(adata.n_vars, dtype=np.float64)
        pvals_adj[gene_idx] = pvals_adj_ranked

        if n_degs_per_pert is not None:
            if pert_id_int >= len(n_degs_per_pert):
                raise ValueError(f"n_degs_per_pert has no entry for perturbation id {pert_id_int}.")
            deg_mask = _topk_mask(pvals_adj, n_degs_per_pert[pert_id_int])
        else:
            deg_mask = pvals_adj < float(alpha)
        degs.append(deg_mask)

    return mu_control, mu_pooled, degs

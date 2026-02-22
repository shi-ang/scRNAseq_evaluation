import numpy as np
from anndata import AnnData
from anndata.experimental import AnnCollection

from util.anndata_util import fit_control_incremental_pca, iterate_batches, obs_has_key


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


def vendi_score(
    ac: AnnData | AnnCollection,
    ac_batch_size: int = 1024,
    n_pca_components: int = 50,
    sample_size: int = 2000,
    random_state: int = 0,
    layer_key: str | None = None,
    control_label: str | int | float = -1,
) -> float:
    """
    Compute perturbation-level Vendi score using RBF-MMD distances.

    Procedure:
    1) Fit IncrementalPCA on control cells only (perturbation == control_label).
    2) Transform all perturbation cells with that same transformation.
    3) Build pairwise perturbation distance matrix via biased MMD^2 (RBF kernel).
    4) Convert distances to similarities K_ij = exp(-MMD^2_ij / (2 * sigma^2)).
    5) Normalize matrix as A = K / n_perturbations.
    6) Eigendecompose A to obtain eigenvalues.
    7) Compute von Neumann entropy H(A) = -sum_i lambda_i log(lambda_i), with 0log0 = 0.
    8) Return Vendi score VS = exp(H(A)).

    Arguments:
    - ac: AnnData or AnnCollection containing the data, with .obs["perturbation"] indicating control (-1) vs perturbation groups.
    - ac_batch_size: number of cells to process at a time when iterating through the data object (for memory efficiency).
    - n_pca_components: number of PCA components to use for the embedding before MMD; default 50 for a balance of expressiveness and stability.
    - sample_size: number of rows to sample for estimating sigma_rbf and sigma_transform; default 2000 for a balance of speed and stability.
    - random_state: random seed for reproducibility of sampling.
    - layer_key: if provided, use this layer instead of `.X`.
    - control_label: label in .obs["perturbation"] identifying control cells.
    """
    if not obs_has_key(getattr(ac, "obs", None), "perturbation"):
        raise KeyError("ac.obs must contain a 'perturbation' column.")

    obs_pert = np.asarray(ac.obs["perturbation"])
    perturbation_ids = np.unique(obs_pert[obs_pert != control_label])
    try:
        perturbation_ids = np.asarray(sorted(perturbation_ids.tolist()), dtype=object)
    except TypeError:
        perturbation_ids = np.asarray(
            sorted(perturbation_ids.tolist(), key=lambda x: str(x)),
            dtype=object,
        )
    n_perturbations = perturbation_ids.size
    if n_perturbations == 0:
        return float("nan")

    rng = np.random.default_rng(random_state)

    # Step 1: fit IncrementalPCA on control cells only.
    pca_model = fit_control_incremental_pca(
        data_obj=ac,
        layer_key=layer_key,
        control_label=control_label,
        n_pca_components=n_pca_components,
        batch_size=ac_batch_size,
        obs_key="perturbation",
        data_name="ac",
    )

    # Step 2: apply the control-fitted IncrementalPCA model to all perturbation cells.
    # Collect one embedding matrix per perturbation ID.
    pert_embeddings_chunks: dict[object, list[np.ndarray]] = {
        ptb: [] for ptb in perturbation_ids.tolist()
    }
    for batch_matrix, batch_labels in iterate_batches(
        data_obj=ac,
        layer_key=layer_key,
        batch_size=ac_batch_size,
        obs_key="perturbation",
    ):
        if batch_labels is None:
            continue
        pert_mask = batch_labels != control_label
        if not np.any(pert_mask):
            continue

        batch_pert = np.asarray(batch_matrix[pert_mask, :], dtype=np.float64)
        batch_labels_pert = batch_labels[pert_mask]

        z_batch = pca_model.transform(batch_pert).astype(np.float64, copy=False)
        for ptb in np.unique(batch_labels_pert):
            group_mask = batch_labels_pert == ptb
            pert_embeddings_chunks[ptb].append(z_batch[group_mask])

    grouped_embeddings: list[np.ndarray] = []
    for ptb in perturbation_ids.tolist():
        chunks = pert_embeddings_chunks[ptb]
        if not chunks:
            raise ValueError(f"No cells found for perturbation id {ptb!r}.")
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

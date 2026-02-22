import os
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


def _normalize_and_log1p(matrix, normalize=True, target_sum=1e4):
    """
    Normalize each cell to target_sum and apply log1p.
    Operates on CSR sparse matrices or dense numpy arrays. Returns the same type as input.
    """
    if sparse.issparse(matrix):
        matrix = matrix.tocsr(copy=True)
        if normalize:
            lib_sizes = np.asarray(matrix.sum(axis=1)).ravel()
            # Avoid division by zero for cells with zero total counts.
            lib_sizes[lib_sizes == 0.0] = 1.0
            scale = (target_sum / lib_sizes).astype(np.float32, copy=False)
            matrix = matrix.multiply(scale[:, None]).tocsr()
        matrix.data = np.log1p(matrix.data)
    else:
        matrix = np.asarray(matrix, dtype=np.float32).copy()
        if normalize:
            lib_sizes = matrix.sum(axis=1, dtype=np.float32)
            lib_sizes[lib_sizes == 0.0] = 1.0
            matrix *= (target_sum / lib_sizes)[:, None]
        np.log1p(matrix, out=matrix)
    
    return matrix


def sample_nb_counts(mean, l_c, theta, rng): # theta kept as generic parameter name for this utility function
    """
    Generate individual cell profiles from NB distribution
    Returns an array of shape (len(l_c), G)
    """
    # Ensure mean and theta are numpy arrays for element-wise operations
    mean_arr = np.asarray(mean) # size of genes
    theta_arr = np.asarray(theta) # size of genes
    l_c_arr = np.asarray(l_c) # size of cells

    # Correct mean for library size
    if mean_arr.ndim == 1:
        lib_size_corrected_mean = np.outer(l_c_arr, mean_arr)
    else:
        lib_size_corrected_mean = l_c_arr[:, None] * mean_arr

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


class ChunkedAnnDataWriter:
    """
    Buffer sparse count matrices and persist chunked .h5ad files.
    Optionally stores per-cell metadata such as cell type in .obs.
    """

    def __init__(
        self,
        output_dir: str,
        var: pd.DataFrame,
        max_cells_per_chunk: int = 2048,
        normalize: bool = True,
        normalized_layer_key: str = "normalized_log1p",
    ) -> None:
        if max_cells_per_chunk <= 0:
            raise ValueError("max_cells_per_chunk must be a positive integer.")

        self.output_dir = output_dir
        self.var = var
        self.max_cells_per_chunk = int(max_cells_per_chunk)
        self.normalize = normalize
        self.normalized_layer_key = normalized_layer_key

        os.makedirs(self.output_dir, exist_ok=True)

        self._buffer_X: list[sparse.csr_matrix] = []
        self._buffer_labels: list[np.ndarray] = []
        self._buffer_cell_types: list[np.ndarray] = []
        self._has_cell_type = False
        self._buffer_rows = 0
        self._chunk_idx = 0
        self.chunk_paths: list[str] = []

    @property
    def buffer_rows(self) -> int:
        return self._buffer_rows

    def append_counts(self, counts_block, perturbation_id: int, cell_type=None) -> None:
        if counts_block.size == 0:
            return
        if sparse.issparse(counts_block):
            counts_csr = counts_block.tocsr().astype(np.int32, copy=False)
        else:
            counts_csr = sparse.csr_matrix(np.asarray(counts_block, dtype=np.int32))

        n_rows = counts_csr.shape[0]
        if cell_type is None:
            cell_type_arr = np.full(n_rows, -1, dtype=np.int32)
        elif np.isscalar(cell_type):
            cell_type_arr = np.full(n_rows, int(cell_type), dtype=np.int32)
            self._has_cell_type = True
        else:
            cell_type_arr = np.asarray(cell_type, dtype=np.int32).reshape(-1)
            if cell_type_arr.shape[0] != n_rows:
                raise ValueError(
                    f"cell_type must have length {n_rows}, got {cell_type_arr.shape[0]}"
                )
            self._has_cell_type = True

        self._buffer_X.append(counts_csr)
        self._buffer_labels.append(
            np.full(n_rows, perturbation_id, dtype=np.int32)
        )
        self._buffer_cell_types.append(cell_type_arr)
        self._buffer_rows += n_rows

        if self._buffer_rows >= self.max_cells_per_chunk:
            self.flush()

    def flush(self) -> None:
        if self._buffer_rows == 0:
            return

        chunk_counts = sparse.vstack(self._buffer_X, format="csr", dtype=np.int32)
        chunk_labels = np.concatenate(self._buffer_labels).astype(np.int32, copy=False)
        chunk_cell_types = np.concatenate(self._buffer_cell_types).astype(np.int32, copy=False)
        obs_dict = {"perturbation": chunk_labels}
        if self._has_cell_type:
            obs_dict["cell_type"] = chunk_cell_types
        obs = pd.DataFrame(obs_dict, index=[f"cell_{self._chunk_idx}_{i}" for i in range(chunk_counts.shape[0])])
        chunk_adata = ad.AnnData(X=chunk_counts, obs=obs, var=self.var)
        chunk_adata.layers["counts"] = chunk_counts.copy()
        chunk_adata.layers[self.normalized_layer_key] = _normalize_and_log1p(
            chunk_counts, normalize=self.normalize
        )
        chunk_path = os.path.join(
            self.output_dir, f"synthetic_chunk_{self._chunk_idx:06d}.h5ad"
        )
        chunk_adata.write_h5ad(chunk_path, compression="gzip")
        self.chunk_paths.append(chunk_path)

        self._buffer_X.clear()
        self._buffer_labels.clear()
        self._buffer_cell_types.clear()
        self._buffer_rows = 0
        self._chunk_idx += 1

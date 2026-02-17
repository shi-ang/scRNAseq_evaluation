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


class ChunkedAnnDataWriter:
    """
    Buffer sparse count matrices and persist chunked .h5ad files.
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
        self._buffer_rows = 0
        self._chunk_idx = 0
        self.chunk_paths: list[str] = []

    @property
    def buffer_rows(self) -> int:
        return self._buffer_rows

    def append_counts(self, counts_block, perturbation_id: int) -> None:
        if counts_block.size == 0:
            return
        if sparse.issparse(counts_block):
            counts_csr = counts_block.tocsr().astype(np.int32, copy=False)
        else:
            counts_csr = sparse.csr_matrix(np.asarray(counts_block, dtype=np.int32))

        self._buffer_X.append(counts_csr)
        self._buffer_labels.append(
            np.full(counts_csr.shape[0], perturbation_id, dtype=np.int32)
        )
        self._buffer_rows += counts_csr.shape[0]

        if self._buffer_rows >= self.max_cells_per_chunk:
            self.flush()

    def flush(self) -> None:
        if self._buffer_rows == 0:
            return

        chunk_counts = sparse.vstack(self._buffer_X, format="csr", dtype=np.int32)
        chunk_labels = np.concatenate(self._buffer_labels).astype(np.int32, copy=False)
        obs = pd.DataFrame(
            {"perturbation": chunk_labels},
            index=[f"cell_{self._chunk_idx}_{i}" for i in range(chunk_counts.shape[0])],
        )
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
        self._buffer_rows = 0
        self._chunk_idx += 1

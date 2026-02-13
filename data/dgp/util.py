import numpy as np
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

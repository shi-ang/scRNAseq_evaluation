from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import pandas as pd
from scipy import sparse
import scipy.sparse.linalg as spla
import anndata as ad

from .util import _normalize_and_log1p


def _softplus(x: np.ndarray) -> np.ndarray:
    """
    Stable softplus: log(1+exp(x)). Works for float32/float64 arrays. 
    softplus(x) = log1p(exp(-|x|)) + max(x, 0)
    """
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def _unif_pm(rng: np.random.Generator, low: float, high: float, size) -> np.ndarray:
    """Uniform magnitude in [low, high] times random sign ±1."""
    mag = rng.uniform(low, high, size=size)
    sign = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=size)
    return mag * sign


def _build_A_erdos_renyi(
    G: int,
    rng: np.random.Generator,
    expected_edges_per_gene: int = 10,
) -> sparse.csr_matrix:
    """
    Build sparse A with ~expected_edges_per_gene nonzeros per row (target gene),
    i.e., ~10 regulators per gene.

    A_{i,j} != 0 means gene j directly influences gene i in the linear drift.
    """
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    # We use exactly K edges per row (close to "10 in expectation" but deterministic).
    K = max(0, int(expected_edges_per_gene))

    for i in range(G):
        if K == 0:
            continue
        # choose regulators j != i
        # (for large G, this is fast enough and avoids dense masks)
        regs = rng.choice(G - 1, size=K, replace=False)
        regs = regs + (regs >= i)  # shift up to skip i
        w = _unif_pm(rng, 1.0, 3.0, size=K)
        rows.extend([i] * K)
        cols.extend(regs.tolist())
        data.extend(w.tolist())

    A = sparse.csr_matrix((np.asarray(data, dtype=np.float32),
                       (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32))),
                      shape=(G, G))
    return A


def _build_A_power_law_BA(
    G: int,
    rng: np.random.Generator,
    m: int = 10,
    strength: float = 2.0,
    flip_prob: float = 0.1,
) -> sparse.csr_matrix:
    """
    Barabási–Albert-style preferential attachment with exponent 'strength' (degree**strength),
    ~m links per new node, then flip edge direction with probability flip_prob to create feedback.

    This is O(G^2) in the naive implementation due to computing probabilities each step.
    For G=10_000 it can still be OK on HPC, but Erdős–Rényi is much faster.
    """
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    # Start with a small seed set; typical BA uses m0 >= m
    m0 = max(m + 1, 2)
    degrees = np.zeros(G, dtype=np.float64)

    # Seed: connect nodes [0..m0-1] in a simple chain to initialize degrees
    for i in range(m0 - 1):
        j = i + 1
        degrees[i] += 1
        degrees[j] += 1

        # Default direction older -> newer
        src, dst = i, j
        if rng.random() < flip_prob:
            src, dst = dst, src

        rows.append(dst)
        cols.append(src)
        data.append(float(_unif_pm(rng, 1.0, 3.0, size=()).item()))

    # Grow graph
    for new in range(m0, G):
        # attachment probs proportional to degree**strength (plus tiny epsilon)
        deg = degrees[:new]
        w = (deg + 1e-9) ** strength
        w_sum = w.sum()
        if not np.isfinite(w_sum) or w_sum <= 0:
            p = np.full(new, 1.0 / new, dtype=np.float64)
        else:
            p = w / w_sum

        targets = rng.choice(new, size=min(m, new), replace=False, p=p)

        for old in targets:
            # Default orientation: old -> new (hubs become regulators)
            src, dst = int(old), int(new)
            if rng.random() < flip_prob:
                src, dst = dst, src

            rows.append(dst)
            cols.append(src)
            data.append(float(_unif_pm(rng, 1.0, 3.0, size=()).item()))

            degrees[old] += 1
            degrees[new] += 1

    A = sparse.csr_matrix((np.asarray(data, dtype=np.float32),
                       (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32))),
                      shape=(G, G))
    return A


def _shift_causal_matrix(A: sparse.csr_matrix, target_max_real_eig: float = -0.5) -> Tuple[sparse.csr_matrix, float]:
    """
    Shift the causal matrix by a constant diagonal so that max real-part eigenvalue <= target_max_real_eig.

    We try ARPACK eigs(which='LR') on the non-symmetric A. If it fails, we fall back to using
    the symmetric part (A + A.T)/2 which provides an upper bound on spectral abscissa.
    """
    G = A.shape[0]
    s_est = None

    try:
        # largest real part eigenvalue estimate
        vals = spla.eigs(A, k=1, which="LR", return_eigenvectors=False, tol=1e-2, maxiter=2000)
        s_est = float(np.max(np.real(vals)))
    except Exception:
        # fallback: use symmetric part upper bound
        As = (A + A.T).multiply(0.5)
        vals = spla.eigsh(As, k=1, which="LA", return_eigenvectors=False, tol=1e-2, maxiter=2000)
        s_est = float(vals[0])

    A_stable = A - (s_est - target_max_real_eig) * sparse.eye(G, format="csr", dtype=np.float32)
    return A_stable


@dataclass
class EMSampler:
    """
    Parallel Euler–Maruyama sampler for:
        dx = (A x + bc) dt + sigma dW

    We maintain 'chains' parallel chains of dimension G and stream out samples.
    """
    A: sparse.csr_matrix                   # (G,G) sparse
    bc: np.ndarray                     # (G,) float32, bc = b + c_q
    rng: np.random.Generator
    dt: float = 1e-3
    sigma: float = math.sqrt(2.0)
    burn_in_steps: int = 200
    thinning_steps: int = 20
    chains: int = 64
    dtype: type = np.float32

    def __post_init__(self):
        G = self.A.shape[0]
        self.bc = np.asarray(self.bc, dtype=self.dtype)
        self.x = self.rng.normal(0.0, 1.0, size=(self.chains, G)).astype(self.dtype, copy=False)

        # Burn-in once per condition to reduce dependence on initialization
        self._step(self.burn_in_steps)

    def _drift(self, x: np.ndarray) -> np.ndarray:
        # drift = (A @ x^T)^T + bc
        lin = (self.A @ x.T).T  # (chains,G)
        lin += self.bc[None, :]
        return lin

    def _step(self, n_steps: int):
        if n_steps <= 0:
            return
        dt = float(self.dt)
        noise_scale = float(self.sigma) * math.sqrt(dt)

        for _ in range(n_steps):
            drift = self._drift(self.x)
            noise = self.rng.normal(0.0, 1.0, size=self.x.shape).astype(self.dtype, copy=False)
            self.x = self.x + dt * drift + noise_scale * noise

    def draw(self, n_samples: int) -> np.ndarray:
        """
        Draw n_samples latent states. Returns array of shape (n_samples, G).
        Samples will be correlated if thinning is small; increase thinning_steps if needed.
        """
        out = []
        remaining = int(n_samples)

        while remaining > 0:
            take = min(remaining, self.chains)
            out.append(self.x[:take].copy())
            remaining -= take
            if remaining > 0:
                self._step(self.thinning_steps)

        return np.vstack(out)


def _sample_zip_counts(
    x: np.ndarray,                 # (B,G) latent
    pi: np.ndarray,                # (G,) dropout probs
    eta: np.ndarray,               # (G,) scale
    rng: np.random.Generator,
) -> sparse.csr_matrix:
    """
    Zero-inflated Poisson (ZIP) observation model (per gene):
        with prob pi_g: y_g = 0
        else: y_g ~ Poisson( mu_g(x_g) )
    and mu_g(x_g) = eta_g * softplus(x_g)
    """
    x = np.asarray(x, dtype=np.float32)
    pi = np.asarray(pi, dtype=np.float32)
    eta = np.asarray(eta, dtype=np.float32)

    mu = eta[None, :] * _softplus(x).astype(np.float32)
    y = rng.poisson(mu).astype(np.int32, copy=False)

    # dropout mask (True => force zero)
    drop = rng.random(size=y.shape, dtype=np.float32) < pi[None, :]
    y[drop] = 0
    return sparse.csr_matrix(y)


def synthetic_causalDGP(
    G=10_000,   # number of genes
    N0=3_000,   # number of control cells
    Nk=150,     # number of perturbed cells per perturbation
    P=50,       # number of perturbations
    pi: np.ndarray = None, # dropout (zero-inflation) probabilities for each gene, shape (G,)
    eta: np.ndarray = None, # gene-specific rate scaling used inside function mu_g(), shape (G,)
    mask_method: str = "Erdos-Renyi", # Erdos-Renyi or Power-law
    trial_id_for_rng=None, # Optional for seeding RNG per trial
    output_dir=None, # Directory to persist temporary chunked h5ad files
    max_cells_per_chunk=2048,
    normalize=True, # Whether to normalize before log1p for the persisted layer
    normalized_layer_key: str = "normalized_log1p", # Layer name for normalized/log1p values
) -> Tuple[list[str], list[np.ndarray]]:
    """
    End-to-end synthetic scRNA-seq generator:

    Latent dynamics (Euler–Maruyama only):
        dx = (A x + b + c_q) dt + sqrt(2) dW
    Observation model (ZIP):
        y_g = 0 w.p. pi_g
        else y_g ~ Poisson( eta_g * softplus(x_g) )

    Outputs chunked .h5ad files:
      - .X: raw counts (CSR, int32)
      - .layers[normalized_layer_key]: normalized/log1p (CSR, float32) if normalize=True

    Returns:
      - chunk_paths: list[str]
      - all_affected_masks: list[np.ndarray] length P, each bool mask of shape (G,)
    """
    # ---- RNG ----
    if trial_id_for_rng is None:
        rng = np.random.default_rng(42)
    else:
        rng = np.random.default_rng(trial_id_for_rng)

    # TODO: add parameter preparation

    if output_dir is None:
        raise ValueError("output_dir must be provided.")
    os.makedirs(output_dir, exist_ok=True)

    # defaults pi / eta
    if pi is None:
        # Moderate dropout; adjust as desired
        pi = rng.beta(2.0, 5.0, size=G).astype(np.float32)
        pi = np.clip(pi, 0.02, 0.98)
    else:
        pi = np.asarray(pi, dtype=np.float32)
        if pi.shape != (G,):
            raise ValueError(f"pi must have shape (G,), got {pi.shape}")

    if eta is None:
        # Positive scale >= 1; adjust as desired
        eta = rng.lognormal(mean=0.0, sigma=0.4, size=G).astype(np.float32)
        eta = np.maximum(eta, 1.0)
    else:
        eta = np.asarray(eta, dtype=np.float32)
        if eta.shape != (G,):
            raise ValueError(f"eta must have shape (G,), got {eta.shape}")
        eta = np.maximum(eta, 1.0)

    # build shift function f_q(x) = Ax + b + c_q
    mask_method = mask_method.lower()
    if mask_method == "power-law":
        A = _build_A_power_law_BA(G=G, rng=rng, m=10, strength=2.0, flip_prob=0.1)
    elif mask_method == "erdos-renyi":
        A = _build_A_erdos_renyi(G=G, rng=rng, expected_edges_per_gene=10)
    else:
        raise ValueError(f"Unknown mask_method: {mask_method}")

    A = _shift_causal_matrix(A, target_max_real_eig=-0.5)

    b = rng.uniform(-3.0, 3.0, size=G).astype(np.float32)

    # sample perturbations c_q
    # Control is q=-1 with c=0. Perturbations are q=0..P-1.
    targets = rng.choice(G, size=P, replace=False).astype(np.int64)
    shifts = _unif_pm(rng, 5.0, 15.0, size=P).astype(np.float32)

    all_affected_masks: list[np.ndarray] = []
    for p in range(P):
        m = np.zeros(G, dtype=bool)
        m[int(targets[p])] = True
        all_affected_masks.append(m)

    # For speed, store bc vectors as (b + c_q) without materializing full c_q repeatedly.
    # bc_control = b
    bc_list = []
    bc_list.append(b.copy())  # index 0 corresponds to control bc (q=-1)
    for p in range(P):
        bc = b.copy()
        bc[int(targets[p])] += float(shifts[p])
        bc_list.append(bc)

    # ---- var (genes) ----
    var = pd.DataFrame(index=pd.Index([f"gene_{i}" for i in range(G)], name="gene"))

    # Buffer small batches
    buffer_X: list[sparse.csr_matrix] = []
    buffer_labels = []
    buffer_rows = 0
    chunk_paths: list[str] = []
    chunk_idx = 0

    def flush_buffer():
        nonlocal buffer_X, buffer_labels, buffer_rows, chunk_idx
        if buffer_rows == 0:
            return
        # Persist one sparse chunk to disk and clear in-memory buffers.
        chunk_counts = sparse.vstack(buffer_X, format="csr", dtype=np.int32)
        chunk_labels = np.concatenate(buffer_labels).astype(np.int32, copy=False)
        obs = pd.DataFrame(
            {"perturbation": chunk_labels},
            index=[f"cell_{chunk_idx}_{i}" for i in range(chunk_counts.shape[0])],
        )
        chunk_adata = ad.AnnData(X=chunk_counts, obs=obs, var=var)
        chunk_adata.layers[normalized_layer_key] = _normalize_and_log1p(chunk_counts, normalize=normalize)
        chunk_path = os.path.join(output_dir, f"synthetic_chunk_{chunk_idx:06d}.h5ad")
        chunk_adata.write_h5ad(chunk_path, compression="gzip")
        chunk_paths.append(chunk_path)

        buffer_X = []
        buffer_labels = []
        buffer_rows = 0
        chunk_idx += 1

    def append_counts(counts_block, perturbation_id):
        nonlocal buffer_rows
        if counts_block.size == 0:
            return
        # perturbation_id == -1 is reserved for control cells.
        buffer_X.append(counts_block.astype(np.int32, copy=False))
        buffer_labels.append(np.full(counts_block.shape[0], perturbation_id, dtype=np.int32))
        buffer_rows += counts_block.shape[0]
        if buffer_rows >= max_cells_per_chunk:
            flush_buffer()

    # ---- sampling hyperparameters (tune as needed) ----
    # These are not specified by the paper for the linear simulation; EM is your requested choice.
    dt = 1e-4
    burn_in_steps = 200
    thinning_steps = 20
    chains_default = 64  # number of parallel chains per condition

    # ---- Generate conditions sequentially ----
    # condition order: control (-1), then perturbations 0..P-1
    conditions = [(-1, N0, 0)] + [(p, Nk, p + 1) for p in range(P)]  # (label, n_cells, bc_index)

    for label, n_cells, bc_index in conditions:
        bc = bc_list[bc_index]

        chains = min(chains_default, n_cells, max_cells_per_chunk)
        sampler = EMSampler(
            A=A, bc=bc, rng=rng,
            dt=dt, sigma=math.sqrt(2.0),
            burn_in_steps=burn_in_steps,
            thinning_steps=thinning_steps,
            chains=chains,
            dtype=np.float32,
        )

        remaining = n_cells
        while remaining > 0:
            # how many cells can we add before flushing?
            space = max_cells_per_chunk - buffer_rows
            if space <= 0:
                flush_buffer()
                continue

            current_batch_size = min(remaining, space)
            x_batch = sampler.draw(current_batch_size)                  # (batch_n, G) latent
            y_batch = _sample_zip_counts(x_batch, pi=pi, eta=eta, rng=rng)  # (batch_n, G)

            append_counts(y_batch, perturbation_id=label)
            remaining -= current_batch_size
        print(f"Finished condition {label} with {n_cells} cells.")
    # flush tail
    flush_buffer()
    return chunk_paths, all_affected_masks

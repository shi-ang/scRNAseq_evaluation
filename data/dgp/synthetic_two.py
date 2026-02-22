from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from scipy import sparse
import scipy.sparse.linalg as spla
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from .util import ChunkedAnnDataWriter, sample_nb_counts

matplotlib.use("Agg")


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

    for i in range(G):
        if expected_edges_per_gene == 0:
            continue
        # choose regulators j != i
        # (for large G, this is fast enough and avoids dense masks)
        regs = rng.choice(G - 1, size=expected_edges_per_gene, replace=False)
        regs = regs + (regs >= i)  # shift up to skip i
        w = _unif_pm(rng, 1.0, 3.0, size=expected_edges_per_gene)
        rows.extend([i] * expected_edges_per_gene)
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
    with approximately m links per gene on average (not a hard fixed count per new node).
    Edge direction is flipped with probability flip_prob to create feedback loops.

    This is O(G^2) in the naive implementation due to computing probabilities each step.
    For G=10_000 it can still be OK on HPC, but Erdős–Rényi is much faster.
    """
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    # Start with a small seed set. We do not require m0 >= m because m is treated
    # as an average number of links, and per-node links are sampled stochastically.
    m0 = 12 if G >= 12 else max(G, 2)
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

        # "10 links per gene" is interpreted as an average rather than fixed links.
        # Sample links per new node from a Poisson distribution centered at m.
        k_links = int(rng.poisson(lam=max(float(m), 0.0)))
        k_links = max(1, min(k_links, new))
        targets = rng.choice(new, size=k_links, replace=False, p=p)

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


def _build_A_alter(
    A: sparse.csr_matrix,
    rng: np.random.Generator,
    swap_fraction: float = 0.2,
) -> sparse.csr_matrix:
    """
    Create a row-wise perturbed copy of A by swapping a fraction of nonzero entries
    with zero locations in the same row.

    For each row:
      - choose floor(nnz_row * swap_fraction) existing edges
      - move those edge weights to currently-zero columns (without replacement)
      - keep row nnz unchanged and preserve weight distribution
    """
    if not sparse.isspmatrix_csr(A):
        A = A.tocsr()
    A = A.astype(np.float32, copy=True)
    G = A.shape[1]

    new_indices_rows: list[np.ndarray] = []
    new_data_rows: list[np.ndarray] = []
    new_indptr = np.zeros(A.shape[0] + 1, dtype=np.int64)

    for i in range(A.shape[0]):
        start = A.indptr[i]
        end = A.indptr[i + 1]
        row_cols = A.indices[start:end].astype(np.int64, copy=True)
        row_vals = A.data[start:end].astype(np.float32, copy=True)
        row_nnz = row_cols.size

        if row_nnz == 0:
            new_indices_rows.append(np.empty(0, dtype=np.int32))
            new_data_rows.append(np.empty(0, dtype=np.float32))
            new_indptr[i + 1] = new_indptr[i]
            continue

        n_swap = int(math.floor(row_nnz * float(swap_fraction)))
        if n_swap <= 0:
            row_cols_out = row_cols.astype(np.int32, copy=False)
            row_vals_out = row_vals
        else:
            swap_pos = rng.choice(row_nnz, size=n_swap, replace=False)
            keep_mask = np.ones(row_nnz, dtype=bool)
            keep_mask[swap_pos] = False

            keep_cols = row_cols[keep_mask]
            keep_vals = row_vals[keep_mask]
            moved_vals = row_vals[swap_pos]

            occupied = np.zeros(G, dtype=bool)
            occupied[row_cols] = True
            # keep no-self-edge behavior when possible
            if i < G:
                occupied[i] = True

            candidate_zero_cols = np.flatnonzero(~occupied)
            if candidate_zero_cols.size < n_swap:
                # fallback: allow self-edge if needed
                if i < G:
                    occupied[i] = row_cols.__contains__(i)
                candidate_zero_cols = np.flatnonzero(~occupied)

            if candidate_zero_cols.size < n_swap:
                n_swap = int(candidate_zero_cols.size)
                if n_swap == 0:
                    row_cols_out = row_cols.astype(np.int32, copy=False)
                    row_vals_out = row_vals
                    new_indices_rows.append(row_cols_out)
                    new_data_rows.append(row_vals_out)
                    new_indptr[i + 1] = new_indptr[i] + row_cols_out.size
                    continue
                moved_vals = moved_vals[:n_swap]

            new_cols = rng.choice(candidate_zero_cols, size=n_swap, replace=False)
            row_cols_out = np.concatenate([keep_cols, new_cols]).astype(np.int32, copy=False)
            row_vals_out = np.concatenate([keep_vals, moved_vals]).astype(np.float32, copy=False)

            order = np.argsort(row_cols_out, kind="mergesort")
            row_cols_out = row_cols_out[order]
            row_vals_out = row_vals_out[order]

        new_indices_rows.append(row_cols_out)
        new_data_rows.append(row_vals_out)
        new_indptr[i + 1] = new_indptr[i] + row_cols_out.size

    if new_indptr[-1] == 0:
        return sparse.csr_matrix(A.shape, dtype=np.float32)

    new_indices = np.concatenate(new_indices_rows).astype(np.int32, copy=False)
    new_data = np.concatenate(new_data_rows).astype(np.float32, copy=False)
    return sparse.csr_matrix((new_data, new_indices, new_indptr), shape=A.shape, dtype=np.float32)


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


def synthetic_causalDGP(
    G=10_000,   # number of genes
    N0=3_000,   # number of control cells
    Nk=150,     # number of perturbed cells per perturbation
    P=50,       # number of perturbations
    mu_l=1.0,   # mean of log library size
    all_theta=None, # Theta parameter for all cells , size of total number of genes in the real dataset (>= G)
    mask_method: str = "Erdos-Renyi", # Erdos-Renyi or Power-law
    trial_id_for_rng=None, # Optional for seeding RNG per trial
    output_dir=None, # Directory to persist temporary chunked h5ad files
    max_cells_per_chunk=2048,
    normalize=True, # Whether to normalize before log1p for the persisted layer
    normalized_layer_key: str = "normalized_log1p", # Layer name for normalized/log1p values
    visualize=False, # Whether to visualize the A matrix and its spectrum for debugging
) -> list[str]:
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
      - .obs["cell_type"]: Bernoulli cell type labels (0/1)

    Returns:
      - chunk_paths: list[str]
    """
    if trial_id_for_rng is None:
        rng = np.random.default_rng(42)
    else:
        rng = np.random.default_rng(trial_id_for_rng)

    if output_dir is None:
        raise ValueError("output_dir must be provided.")

    # Sample G elements from all_theta
    indices = rng.choice(len(all_theta), size=G, replace=False)
    local_all_theta = all_theta[indices]

    # build shift function f_q(x) = Ax + b + c_q
    mask_method = mask_method.lower()
    if mask_method == "power-law":
        A = _build_A_power_law_BA(G=G, rng=rng, m=10, strength=2.0, flip_prob=0.1)
        # A_alter = _build_A_power_law_BA(G=G, rng=rng, m=10, strength=2.0, flip_prob=0.1)
    elif mask_method == "erdos-renyi":
        A = _build_A_erdos_renyi(G=G, rng=rng, expected_edges_per_gene=10)
        # A_alter = _build_A_erdos_renyi(G=G, rng=rng, expected_edges_per_gene=10)
    else:
        raise ValueError(f"Unknown mask_method: {mask_method}")
    
    # Create a perturbed version of A by swapping 20% of each row's nonzero edges
    # into zero positions. This preserves row sparsity and weight distribution.
    A_alter = _build_A_alter(A=A, rng=rng, swap_fraction=0.2)

    A = _shift_causal_matrix(A, target_max_real_eig=-0.5)
    A_alter = _shift_causal_matrix(A_alter, target_max_real_eig=-0.5)
    A_list = [A, A_alter]

    # TODO: change this to incorperate cell types
    b_base = rng.uniform(-3.0, 3.0, size=G).astype(np.float32)
    # b_base_alt = rng.uniform(-30.0, 30.0, size=G).astype(np.float32)
    # b_list = [b_base, b_base_alt]
    direction = rng.standard_normal(G).astype(np.float32)
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm == 0.0:
        # Numerical guard: extremely unlikely, but ensures we have a valid direction.
        direction[0] = 1.0
        direction_norm = 1.0
    direction /= direction_norm

    separation = 10.0  # controls how distinct the two cell types are
    half_sep = np.float32(separation / 2.0)
    b_list = [
        (b_base - half_sep * direction).astype(np.float32, copy=False),  # type A (0)
        (b_base + half_sep * direction).astype(np.float32, copy=False),  # type B (1)
    ]

    # sample perturbations c_q
    # Control is q=-1 with c=0. Perturbations are q=0..P-1.
    targets = rng.choice(G, size=P, replace=False).astype(np.int64)
    targets.sort()  # sort target indices for easier interpretation and debugging
    shifts = _unif_pm(rng, 5.0, 15.0, size=P).astype(np.float32)

    if visualize:
        # Build stacked matrices [A; b] and [A_alter; b_alter], each with shape (G+1, G).
        Ab = np.vstack([A.toarray().astype(np.float32, copy=False), b_list[0][None, :]])
        Ab_alter = np.vstack([A_alter.toarray().astype(np.float32, copy=False), b_list[1][None, :]])

        # Black around 0, red for negative, blue for positive.
        # With vmin/vmax set to [-3, 3], 0 map to 0.5 on the color axis.
        cmap = LinearSegmentedColormap.from_list(
            "red_black_blue",
            [
                (0.00, "#fdabab"),  # strong negative -> red
                (0.50, "#000000"),  # 0 -> black
                (1.00, "#9baffe"),  # strong positive -> blue
            ],
            N=256,
        )
        vmin, vmax = -3.0, 3.0

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150, constrained_layout=True)
        im0 = axes[0].imshow(Ab, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
        im1 = axes[1].imshow(Ab_alter, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")

        axes[0].set_title("[A; b] (cell type 0)")
        axes[1].set_title("[A_alter; b_alter] (cell type 1)")
        for ax in axes:
            ax.set_xlabel("Gene index (column)")
            ax.set_ylabel("Gene index + b row")
            ax.set_yticks([0, G])
            ax.set_yticklabels(["0", "b row"])
            ax.axhline(G - 0.5, color="white", linewidth=0.5, alpha=0.8)

        cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.85, pad=0.02)
        cbar.set_label("Value (clipped to [-3, 3])")

        out_path = os.path.join(output_dir, "causal_effect_visualization.png")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved causal effect heatmap visualization to: {out_path}")


    # Store bc vectors for each cell type as (b_type + c_q).
    # Index 0 corresponds to control bc (q=-1), then perturbations q=0..P-1.
    bc_list: list[list[np.ndarray]] = [[], []]
    for cell_type, b in enumerate(b_list):
        bc_list[cell_type].append(b.copy())
        for p in range(P):
            bc = b.copy()
            bc[targets[p]] += shifts[p]
            bc_list[cell_type].append(bc)

    var = pd.DataFrame(index=pd.Index([f"gene_{i}" for i in range(G)], name="gene"))

    writer = ChunkedAnnDataWriter(
        output_dir=output_dir,
        var=var,
        max_cells_per_chunk=max_cells_per_chunk,
        normalize=normalize,
        normalized_layer_key=normalized_layer_key,
    )

    # EM sampling parameters
    dt = 1e-4
    burn_in_steps = 500
    thinning_steps = 10
    chains_default = 64  # number of parallel chains per condition

    # condition order: control (-1), then perturbations 0..P-1
    conditions = [(-1, N0, 0)] + [(p, Nk, p + 1) for p in range(P)]  # (perturbation_id, n_cells, bc_index)

    for perturbation_id, n_cells, bc_index in conditions:
        chains = min(chains_default, n_cells, max_cells_per_chunk)
        samplers = [
            EMSampler(
                A=A_list[cell_type], 
                bc=bc_list[cell_type][bc_index], 
                rng=rng,
                dt=dt, 
                sigma=math.sqrt(2.0),
                burn_in_steps=burn_in_steps,
                thinning_steps=thinning_steps,
                chains=chains,
                dtype=np.float32,
            )
            for cell_type in (0, 1)
        ]

        remaining = n_cells
        while remaining > 0:
            # how many cells can we add before flushing?
            space = max_cells_per_chunk - writer.buffer_rows
            if space <= 0:
                writer.flush()
                continue

            current_batch_size = min(remaining, space)
            cell_type_batch = rng.binomial(1, 0.5, size=current_batch_size).astype(np.int32, copy=False)
            x_batch = np.empty((current_batch_size, G), dtype=np.float32)
            idx_type0 = np.flatnonzero(cell_type_batch == 0)
            idx_type1 = np.flatnonzero(cell_type_batch == 1)
            if idx_type0.size > 0:
                x_batch[idx_type0] = samplers[0].draw(int(idx_type0.size))
            if idx_type1.size > 0:
                x_batch[idx_type1] = samplers[1].draw(int(idx_type1.size))

            #softplus and normalize mu_batch to have unit library size of mu_l across genes, per cell
            mu_batch = _softplus(x_batch)
            # mu_batch = np.exp(x_batch) - 1.0 + 1e-8  # softplus with small epsilon to avoid log(0)
            # mu_batch_sum = mu_batch.sum(axis=0, keepdims=True)
            # print(f"mu_batch_shape: {mu_batch.shape}, mu_batch_sum_shape: {mu_batch_sum.shape}")
            # mu_batch = mu_batch / (mu_batch_sum + 1e-8)

            lib_size_pert = rng.lognormal(
                mean=mu_l, sigma=0.1714, size=current_batch_size
            )  # 0.1714 from all cells of the Norman19 dataset
            
            counts = sample_nb_counts(
                mean=mu_batch, l_c=lib_size_pert, theta=local_all_theta, rng=rng
            )

            writer.append_counts(counts, perturbation_id=perturbation_id, cell_type=cell_type_batch)
            remaining -= current_batch_size

    # flush tail
    writer.flush()
    return writer.chunk_paths

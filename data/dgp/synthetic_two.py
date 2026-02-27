from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Callable, Tuple

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import sparse

from .util import ChunkedAnnDataWriter, sample_nb_counts, sample_unif_pm

def _softplus(x: np.ndarray) -> np.ndarray:
    """
    Stable softplus: log(1+exp(x)). Works for float32/float64 arrays. 
    softplus(x) = log1p(exp(-|x|)) + max(x, 0)
    """
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def _build_matrix_erdos_renyi(
    G: int,
    rng: np.random.Generator,
    expected_edges_per_gene: int = 10,
) -> sparse.csr_matrix:
    """
    Build sparse with ~expected_edges_per_gene nonzeros per row (target gene),
    i.e., ~10 regulators per gene.

    A_{i,j} != 0 means gene j directly influences gene i in the linear drift.
    """
    edges_per_gene = min(expected_edges_per_gene, G - 1)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for i in range(G):
        # choose regulators j != i
        # (for large G, this is fast enough and avoids dense masks)
        regs = rng.choice(G - 1, size=edges_per_gene, replace=False)
        regs = regs + (regs >= i)  # shift up to skip i
        w = sample_unif_pm(rng, 1.0, 3.0, size=edges_per_gene)
        rows.extend([i] * edges_per_gene)
        cols.extend(regs.tolist())
        data.extend(w.tolist())

    A = sparse.csr_matrix((np.asarray(data, dtype=np.float32),
                       (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32))),
                      shape=(G, G))
    return A


def _build_matrix_power_law(
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
    m0 = min(12, G)
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
        data.append(float(sample_unif_pm(rng, 1.0, 3.0, size=()).item()))

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
            data.append(float(sample_unif_pm(rng, 1.0, 3.0, size=()).item()))

            degrees[old] += 1
            degrees[new] += 1

    A = sparse.csr_matrix((np.asarray(data, dtype=np.float32),
                       (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32))),
                      shape=(G, G))
    return A


def _shift_causal_matrix(
    A: sparse.csr_matrix, 
    target_max_real_eig: float = -0.5
) -> sparse.csr_matrix:
    """
    Shift the causal matrix by a constant diagonal so that max real-part eigenvalue <= target_max_real_eig.

    We try ARPACK eigs(which='LR') on the non-symmetric A. If it fails, we fall back to using
    the symmetric part (A + A.T)/2 which provides an upper bound on spectral abscissa.
    """
    G = A.shape[0]
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


def _build_base_matrix(
    G: int, rng: np.random.Generator, mask_method: str
) -> sparse.csr_matrix:
    method = mask_method.strip().lower().replace("_", "-")
    builders: dict[str, Callable[..., sparse.csr_matrix]] = {
        "power-law": _build_matrix_power_law,
        "erdos-renyi": _build_matrix_erdos_renyi,
    }
    if method not in builders:
        raise ValueError(f"Unknown mask_method: {mask_method}")
    return builders[method](G=G, rng=rng)


def _build_bc_vectors(bias: np.ndarray, c_q: np.ndarray) -> list[np.ndarray]:
    out = [bias.copy()]
    out.extend(bias + c_q[p] for p in range(c_q.shape[0]))
    return out


def _swap_matrix(
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
    swap_fraction: float = 0.2,  # row-wise fraction of A edges rewired for A_alter
    seed=None, # Optional for seeding RNG per trial
    output_dir=None, # Directory to persist temporary chunked h5ad files
    max_cells_per_chunk=2048,
    normalize=True, # Whether to normalize before log1p for the persisted layer
    normalized_layer_key: str = "normalized_log1p", # Layer name for normalized/log1p values
    visualize=False, # Whether to visualize the A matrix and its spectrum for debugging
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
      - .obs["cell_type"]: Bernoulli cell type labels (0/1)

    Returns:
      - chunk_paths: list[str], h5ad files containing sparse count chunks
      - all_affected_masks: list[np.ndarray], one mask per perturbation indicating which genes are affected
    """
    rng = np.random.default_rng(42 if seed is None else seed)

    if output_dir is None:
        raise ValueError("output_dir must be provided.")
    if all_theta is None:
        raise ValueError("all_theta must be provided.")
    if G <= 0:
        raise ValueError(f"G must be positive, got {G}")
    if N0 < 0 or Nk < 0 or P < 0:
        raise ValueError(f"N0, Nk, and P must be non-negative, got N0={N0}, Nk={Nk}, P={P}")
    if max_cells_per_chunk <= 0:
        raise ValueError(
            f"max_cells_per_chunk must be positive, got {max_cells_per_chunk}"
        )
    if not (0.0 <= swap_fraction <= 1.0):
        raise ValueError(
            f"swap_fraction must be in [0, 1], got {swap_fraction}"
        )

    # Sample G elements from all_theta
    all_theta = np.asarray(all_theta, dtype=np.float32).reshape(-1)
    if all_theta.size < G:
        raise ValueError(f"all_theta must have length >= G ({G}), got {all_theta.size}")
    indices = rng.choice(all_theta.size, size=G, replace=False)
    local_all_theta = np.maximum(all_theta[indices], 1e-6).astype(np.float32, copy=False)

    # Build shift function f_q(x) = Ax + b + c_q.
    A = _build_base_matrix(G=G, rng=rng, mask_method=mask_method)
    
    # Create a perturbed version of A by swapping X% (swap_fraction) of each row's nonzero edges
    # into zero positions. This preserves row sparsity and weight distribution.
    A_alter = _swap_matrix(A=A, rng=rng, swap_fraction=swap_fraction)

    A = _shift_causal_matrix(A, target_max_real_eig=-0.5)
    A_alter = _shift_causal_matrix(A_alter, target_max_real_eig=-0.5)

    b_base = rng.uniform(-5.0, 5.0, size=G).astype(np.float32)
    b_base_alt = rng.uniform(-5.0, 5.0, size=G).astype(np.float32)

    # Randomly reorder genes in A and b 
    perm = rng.permutation(G)
    A = A[:, perm][perm, :]
    A_alter = A_alter[:, perm][perm, :]
    b_base = b_base[perm]
    b_base_alt = b_base_alt[perm]

    A_list = [A, A_alter]
    b_list = [b_base, b_base_alt]

    # sample perturbations c_q
    # Control is q=-1 with c=0. Perturbations are q=0..P-1.
    assert P <= G, f"The possible number of perturbation targets exceeds total genes. Reduce P. Got P={P}, G={G}"
    targets = rng.choice(G, size=P, replace=False).astype(np.int64)
    targets.sort()  # for better visualization in the color-map
    shifts = sample_unif_pm(rng, 5.0, 15.0, size=(P)).astype(np.float32)
    c_q = np.zeros((P, G), dtype=np.float32)
    np.add.at(c_q, (np.arange(P), targets), shifts)

    # Get affected genes masks 
    # Find the non-zero entries in A^-1 * c_q[p] to determine which genes are affected by each perturbation.
    # We can use the sparse linear solver to compute A^-1 @ c_q[p]
    all_affected_masks = []
    for p in range(P):
        effect = spla.spsolve(A, c_q[p])
        all_affected_masks.append(np.abs(effect) > 1e-6)

    if visualize:
        # Build stacked matrices [A^T; b] and [A_alter^T; b_alter],
        # each with shape (G + 1, G).
        Ab = np.vstack([A.toarray().T.astype(np.float32, copy=False), b_list[0][None, :]])
        Ab_alter = np.vstack([A_alter.toarray().T.astype(np.float32, copy=False), b_list[1][None, :]])

        # Black around 0, red for negative, blue for positive.
        # With vmin/vmax set to [-3, 3], 0 map to 0.5 on the color axis.
        cmap_1 = LinearSegmentedColormap.from_list(
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
        im0 = axes[0].imshow(Ab, cmap=cmap_1, vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
        im1 = axes[1].imshow(Ab_alter, cmap=cmap_1, vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")

        axes[0].set_title(r"$[A^\top; b]$ (cell type 0)")
        axes[1].set_title(r"$[A_{\text{alter}}^\top; b_{\text{alter}}]$ (cell type 1)")
        for ax in axes:
            ax.set_xlabel("Genes (downstream)")
            ax.set_ylabel("Bias + Genes (upstream)")
            ax.set_yticks([0, G])
            ax.set_yticklabels(["0", "b"])
            ax.tick_params(axis="y", length=0)
            ax.axhline(G - 0.5, color="white", linewidth=0.8, alpha=0.8)
            for spine in ax.spines.values():
                spine.set_visible(False)

        cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.85, pad=0.02)
        cbar.set_label("Value")

        out_path = os.path.join(output_dir, "causal_effect_visualization.png")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved causal effect heatmap visualization to: {out_path}")

        cmap_2 = LinearSegmentedColormap.from_list(
            "orange_white_green",
            [
                (0.00, "#faf9d1"),
                (0.50, "#000000"),
                (1.00, "#e1e2fd"),
            ],
            N=256,
        )
        cq_fig, cq_ax = plt.subplots(figsize=(14, 0.25 * P), dpi=150, constrained_layout=True)
        cq_im = cq_ax.imshow(
            c_q,
            cmap=cmap_2,
            vmin=-15.0,
            vmax=15.0,
            aspect="auto",
            interpolation="nearest",
        )
        cq_ax.set_title(r"Shifts $\left[c_{q_1}, \dots, c_{q_k}\right]^\top$")
        cq_ax.set_xlabel("Gene index")
        cq_ax.set_ylabel("Perturbations")
        for spine in cq_ax.spines.values():
            spine.set_visible(False)
        cq_cbar = cq_fig.colorbar(cq_im, ax=cq_ax, shrink=0.85, pad=0.02)
        cq_cbar.set_label("Shift value")

        cq_out_path = os.path.join(output_dir, "perturbation_shift.png")
        cq_fig.savefig(cq_out_path, bbox_inches="tight")
        plt.close(cq_fig)
        print(f"Saved perturbation shift heatmap visualization to: {cq_out_path}")


    # Store bc vectors for each cell type as (b_type + c_q).
    # Index 0 corresponds to control bc (q=-1), then perturbations q=0..P-1.
    bc_list: list[list[np.ndarray]] = [
        _build_bc_vectors(bias=b, c_q=c_q) for b in b_list
    ]

    var = pd.DataFrame(index=pd.Index([f"gene_{i}" for i in range(G)], name="gene"))

    writer = ChunkedAnnDataWriter(
        output_dir=output_dir,
        var=var,
        max_cells_per_chunk=max_cells_per_chunk,
        normalize=normalize,
        normalized_layer_key=normalized_layer_key,
    )

    # EM sampling parameters
    dt = 1e-3
    burn_in_steps = 8000
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

            # Softplus and normalize mu per cell (rows) so each cell has unit total mean.
            # NOTE: axis=1 is required here; axis=0 can collapse structure across cells.
            mu_batch = _softplus(x_batch).astype(np.float32, copy=False)
            mu_batch = np.maximum(mu_batch, 1e-8, out=mu_batch)
            mu_batch_sum = mu_batch.sum(axis=1, keepdims=True, dtype=np.float32)
            mu_batch_sum[mu_batch_sum <= 0.0] = 1.0
            mu_batch = mu_batch / mu_batch_sum

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
    return writer.chunk_paths, all_affected_masks

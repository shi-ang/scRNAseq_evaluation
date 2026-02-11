from sklearn.metrics import pairwise_distances
import numpy as np

def compute_pds(
    true_effects: np.ndarray,
    pred_effects: np.ndarray,
    metric: str = "l1",
) -> float:
    """
    Compute discrimination score for each perturbation.

    metric:
      - "l1", "l2": sklearn pairwise distances
      - "correlation"/"pearson": 1 - Pearson correlation
      - "cosine": sklearn cosine distance
      - "sign": 1 - (proportion of sign matches), computed over genes with nonzero true effect
    """
    n_perts = true_effects.shape[0]
    scores = np.empty(n_perts)

    for i in range(n_perts):
        R = true_effects[:, ]   # (n_perts, n_genes_sel)
        p = pred_effects[i, ]   # (n_genes_sel,)

        m = metric.lower()
        if m in {"correlation", "pearson"}:
            M = np.vstack([R, p])
            C = np.corrcoef(M)
            corr = C[-1, :-1]
            corr = np.where(np.isnan(corr), -1.0, corr)
            distances = 1.0 - corr

        elif m == "sign":
            # sign-based distance: 1 - proportion of sign matches over nonzero true genes
            R_sign = np.sign(R)                     # (n_perts, n_genes_sel)
            p_sign = np.sign(p)[None, :]            # (1, n_genes_sel) broadcast
            mask_nz = (R != 0)                      # only count where true != 0
            agree = (R_sign == p_sign) & mask_nz    # correct sign & valid
            correct_counts = agree.sum(axis=1)
            denom = mask_nz.sum(axis=1)

            with np.errstate(divide="ignore", invalid="ignore"):
                prop_match = np.where(denom > 0, correct_counts / denom, 0.0)
            distances = 1.0 - prop_match            # lower is better

        else:
            distances = pairwise_distances(R, p.reshape(1, -1), metric=m).flatten()

        sorted_idx = np.argsort(distances)
        rank = int(np.flatnonzero(sorted_idx == i)[0])
        norm_rank = rank / n_perts
        scores[i] = 1.0 - norm_rank

    return np.mean(scores)

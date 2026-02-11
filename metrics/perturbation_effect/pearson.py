import numpy as np
from scipy.stats import pearsonr

def pearson_pert(
        X_obs: np.ndarray, 
        X_pred: np.ndarray, 
        reference: np.ndarray, 
        DEGs: np.ndarray = None
    ) -> float:
    """
    Compute Pearson using a specific reference, for one perturbation.

    Arguments:
    * X_obs: observed post-perturbation profile. Shape: (n_genes,)
    * X_pred: predicted post-perturbation profile. Shape: (n_genes,)
    * reference: reference. Shape: (n_genes,)
    * DEGs: indicators of differentially expressed genes. Shape: (n_genes,)

    Returns a dictionary with 2 metrics: corr_all_allpert (PearsonΔ) and corr_20de_allpert (PearsonΔ20)
    """
    delta_obs = X_obs - reference
    delta_pred = X_pred - reference

    if DEGs is not None:
        if DEGs.sum() >= 2:
            delta_obs = delta_obs[DEGs]
            delta_pred = delta_pred[DEGs]
        else:
            # if there is only 1 DEG, we cannot compute a meaningful Pearson correlation
            return np.nan

    if np.std(delta_obs) > 1e-6 and np.std(delta_pred) > 1e-6:
        return pearsonr(delta_obs, delta_pred)[0]
        # return np.corrcoef(delta_obs, delta_pred)[0, 1]
    else:
        return np.nan
import numpy as np
from anndata import AnnData
from anndata.experimental import AnnCollection
from scipy.stats import nbinom, poisson
from sklearn.metrics.pairwise import pairwise_kernels

from util.anndata_util import extract_rows, fit_control_incremental_pca, obs_has_key

_EPS = 1e-12


def distribution_distance(
    obs: AnnData | AnnCollection,
    pred: AnnData | AnnCollection,
    layer_obs: str,
    layer_pred: str,
    control_label: str | int | float = -1,
    method: str = "parametric",
    distribution_form: str = "NB",
    dist_type: str = "JS-divergence",
    kernel: str = "rbf",
    use_pca: bool = False,
    n_pca_components: int = 50,
    ipca_batch_size: int = 1024,
    **kernel_params,
):
    """
    Calculate a distance between the distributions of observed and predicted values, either by:
      1) fitting parametric distributions to both observed and predicted data (per gene),
         then computing a distance between the two fitted distributions (per gene),
         and returning the mean distance across genes.
      2) directly computing a kernel-based distance (MMD) between the two sets of samples.

    Went through all perturbation types, calculate the distance for each perturbation, and return the median distance across perturbations.
    Arguments:
        obs: observed post-perturbation profile, an object of annData or annCollection. 
        pred: predicted post-perturbation profile, an object of annData or annCollection
        layer_obs: layer key in obs to use for distance calculation.
        layer_pred: layer key in pred to use for distance calculation.
        control_label: label in obs.obs['perturbation'] and pred.obs['perturbation'] that indicates control samples to exclude.
         Default is -1, but this should be set according to the actual control label in the data.
        method: "parametric" or "mmd"
        distribution_form: if method=="parametric", one of "NB" (Negative Binomial), "Poisson", "ZINB" (Zero-Inflated NB).
            NB/ZINB are fit in the common (mu, theta) parameterization:
                Var = mu + mu^2 / theta
        dist_type: if method=="parametric", "JS-divergence" or "Wasserstein" (1-Wasserstein on counts).
        kernel: if method=="mmd", kernel type to use in MMD calculation. Default is 'rbf'.
            choices are 'rbf', 'linear', 'poly', 'sigmoid'.
        use_pca: if True, fit IncrementalPCA on obs control cells only
            (obs.obs["perturbation"] == control_label), then transform all
            non-control cells from both obs and pred before distance computation.
            Currently supported only when method=="mmd".
        n_pca_components: number of components for IncrementalPCA when
            use_pca=True.
        ipca_batch_size: batch size used for IncrementalPCA partial_fit when
            use_pca=True.
        **kernel_params: optional kernel parameters passed to sklearn.metrics.pairwise.pairwise_kernels
            e.g., for rbf/sigmoid: gamma=...
                  for poly: degree=..., gamma=..., coef0=...
                  for sigmoid: gamma=..., coef0=...
    Returns:
        A float representing the distribution distance between obs and pred.
    """

    method_key = method.strip().lower()
    if method_key not in {"parametric", "mmd"}:
        raise ValueError(
            f"Unknown method: {method}. Expected 'parametric' or 'mmd'."
        )
    if use_pca and method_key != "mmd":
        raise ValueError(
            "use_pca=True is currently supported only when method='mmd'."
        )

    for name, data_obj in (("obs", obs), ("pred", pred)):
        obs_df = getattr(data_obj, "obs", None)
        if not obs_has_key(obs_df, "perturbation"):
            raise KeyError(f"{name}.obs must contain a 'perturbation' column.")

    if obs.n_vars != pred.n_vars:
        raise ValueError(
            f"obs and pred must have the same number of genes. "
            f"Got {obs.n_vars} vs {pred.n_vars}."
        )

    obs_labels = np.asarray(obs.obs["perturbation"])
    pred_labels = np.asarray(pred.obs["perturbation"])

    # remove control labels, based on the control_label value
    obs_perts = set(np.unique(obs_labels)) - {control_label}
    pred_perts = set(np.unique(pred_labels)) - {control_label}

    if obs_perts != pred_perts:
        missing_in_pred = obs_perts - pred_perts
        missing_in_obs = pred_perts - obs_perts
        raise ValueError(
            "obs and pred have mismatched perturbation labels (excluding control). "
            f"Missing in pred: {missing_in_pred}; missing in obs: {missing_in_obs}."
        )

    if not obs_perts or not pred_perts:
        return float("nan")

    try:
        ordered_perts = sorted(obs_perts)
    except TypeError:
        ordered_perts = sorted(obs_perts, key=lambda x: str(x))

    obs_idx_by_pert = {pert: np.flatnonzero(obs_labels == pert) for pert in ordered_perts}
    pred_idx_by_pert = {pert: np.flatnonzero(pred_labels == pert) for pert in ordered_perts}

    pca_model = None
    if use_pca:
        pca_model = fit_control_incremental_pca(
            data_obj=obs,
            layer_key=layer_obs,
            control_label=control_label,
            n_pca_components=n_pca_components,
            batch_size=ipca_batch_size,
            obs_key="perturbation",
            data_name="obs",
        )

    per_pert_dists: list[float] = []
    for pert in ordered_perts:
        x_obs = extract_rows(obs, obs_idx_by_pert[pert], layer_obs)
        x_pred = extract_rows(pred, pred_idx_by_pert[pert], layer_pred)
        if x_obs.size == 0 or x_pred.size == 0:
            d = np.nan
        else:
            if pca_model is not None:
                x_obs = pca_model.transform(x_obs).astype(np.float64, copy=False)
                x_pred = pca_model.transform(x_pred).astype(np.float64, copy=False)

            if method_key == "parametric":
                d = param_dist_pert(
                    x_obs=x_obs,
                    x_pred=x_pred,
                    parametric_form=distribution_form,
                    dist_type=dist_type,
                )
            elif method_key == "mmd":
                d = mmd_pert(
                    x_obs=x_obs,
                    x_pred=x_pred,
                    kernel=kernel,
                    **kernel_params,
                )

        per_pert_dists.append(d)

    return np.nanmedian(per_pert_dists) if per_pert_dists else np.nan


def mmd_pert(
        x_obs,
        x_pred,
        kernel='rbf',
        **kernel_params,
):
    """
    Calculate Maximum Mean Discrepancy (MMD^2) between the distributions of observed and predicted values,
    for one perturbation.

    Arguments:
        x_obs: observed post-perturbation profile. Shape: (n_cells, n_dims)
        x_pred: predicted post-perturbation profile. Shape: (n_cells, n_dims)
        kernel: Kernel type to use in MMD calculation. Default is 'rbf'.
            choices are 'rbf', 'linear', 'poly', 'sigmoid'.
        **kernel_params: optional kernel parameters passed to sklearn.metrics.pairwise.pairwise_kernels
            e.g., for rbf/sigmoid: gamma=...
                  for poly: degree=..., gamma=..., coef0=...
                  for sigmoid: gamma=..., coef0=...

    Returns:
        MMD^2 as a float (unbiased estimator when possible; falls back to biased if sample size < 2).
    """
    if x_obs.ndim == 1:
        x_obs = x_obs.reshape(1, -1)
    if x_pred.ndim == 1:
        x_pred = x_pred.reshape(1, -1)

    if x_obs.shape[1] != x_pred.shape[1]:
        raise ValueError(f"x_obs and x_pred must have the same number of dimensions (features). "
                         f"Got {x_obs.shape[1]} and {x_pred.shape[1]}.")

    m, n = x_obs.shape[0], x_pred.shape[0]

    # Compute Gram matrices
    Kxx = pairwise_kernels(x_obs, x_obs, metric=kernel, filter_params=True, **kernel_params)
    Kyy = pairwise_kernels(x_pred, x_pred, metric=kernel, filter_params=True, **kernel_params)
    Kxy = pairwise_kernels(x_obs, x_pred, metric=kernel, filter_params=True, **kernel_params)
    # Unbiased MMD^2 estimator excludes diagonals of Kxx and Kyy
    if m >= 2 and n >= 2:
        xx = (Kxx.sum() - np.trace(Kxx)) / (m * (m - 1))
        yy = (Kyy.sum() - np.trace(Kyy)) / (n * (n - 1))
        xy = Kxy.mean()
        mmd2 = xx + yy - 2.0 * xy
    else:
        # Fallback to biased estimator if too few samples
        mmd2 = Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean()

    return float(mmd2)


def param_dist_pert(
        x_obs: np.ndarray,
        x_pred: np.ndarray,
        parametric_form: str = "NB",
        dist_type: str = "JS-divergence",
) -> float:
    """
    Calculate distribution distance between observed and predicted values for each perturbation,
      1) fit parametric distributions to both observed and predicted data (per gene),
      2) compute a distance between the two fitted distributions (per gene),
      3) return the mean distance across genes.

    Arguments:
        x_obs: observed post-perturbation profile. Shape: (n_cells, n_genes)
        x_pred: predicted post-perturbation profile. Shape: (n_cells, n_genes)
        parametric_form: "NB" (Negative Binomial), "Poisson", "ZINB" (Zero-Inflated NB).
            NB/ZINB are fit in the common (mu, theta) parameterization:
                Var = mu + mu^2 / theta
        dist_type: "JS-divergence" or "Wasserstein" (1-Wasserstein on counts).
    """
    q_tail = 1.0 - 1e-8          # truncation quantile
    kmax_cap = 10_000            # safety cap to avoid huge loops

    def _fit_poisson(mu: float):
        lam = max(float(mu), 0.0)
        return {"lam": lam}

    def _fit_nb(mu: float, var: float):
        mu = max(float(mu), 0.0)
        var = max(float(var), 0.0)
        # Method-of-moments for theta: var = mu + mu^2/theta  => theta = mu^2/(var-mu)
        denom = max(var - mu, 0.0)
        if mu <= _EPS:
            theta = 1e8
        elif denom <= 1e-8 * max(mu, 1.0):  # near-Poisson
            theta = 1e8
        else:
            theta = (mu * mu) / max(denom, _EPS)
        theta = float(np.clip(theta, 1e-8, 1e12))
        return {"mu": mu, "theta": theta}

    def _nbinom_from_mu_theta(mu: float, theta: float):
        # SciPy nbinom(n, p): mean = n*(1-p)/p
        n = float(theta)
        p = n / (n + float(mu) + _EPS)
        p = float(np.clip(p, _EPS, 1.0 - _EPS))
        n = float(max(n, _EPS))
        return n, p

    def _fit_zinb(mu: float, var: float, p0: float):
        # crude but stable: fit NB by moments, then set pi from "excess zeros"
        nb = _fit_nb(mu, var)
        n, p = _nbinom_from_mu_theta(nb["mu"], nb["theta"])
        nb_p0 = float(nbinom(n, p).pmf(0))  # = p**n
        p0 = float(np.clip(p0, 0.0, 1.0))
        pi = (p0 - nb_p0) / max(1.0 - nb_p0, _EPS)
        pi = float(np.clip(pi, 0.0, 1.0 - 1e-12))
        return {"mu": nb["mu"], "theta": nb["theta"], "pi": pi}

    def _pmf(params: dict, ks: np.ndarray, form: str) -> np.ndarray:
        if form == "POISSON":
            pmf = poisson(params["lam"]).pmf(ks)
        elif form == "NB":
            n, p = _nbinom_from_mu_theta(params["mu"], params["theta"])
            pmf = nbinom(n, p).pmf(ks)
        elif form == "ZINB":
            n, p = _nbinom_from_mu_theta(params["mu"], params["theta"])
            pi = params["pi"]
            pmf = (1.0 - pi) * nbinom(n, p).pmf(ks)
            pmf[0] += pi
        else:
            raise ValueError(f"Unknown parametric_form: {form}")
        pmf = np.asarray(pmf, dtype=float)
        s = pmf.sum()
        if not np.isfinite(s) or s <= 0:
            # fallback to a degenerate-at-zero distribution
            pmf = np.zeros_like(pmf)
            pmf[0] = 1.0
            return pmf
        return pmf / s

    def _ppf(params: dict, q: float, form: str) -> float:
        q = float(np.clip(q, _EPS, 1.0 - _EPS))
        if form == "POISSON":
            return float(poisson(params["lam"]).ppf(q))
        elif form == "NB":
            n, p = _nbinom_from_mu_theta(params["mu"], params["theta"])
            return float(nbinom(n, p).ppf(q))
        elif form == "ZINB":
            n, p = _nbinom_from_mu_theta(params["mu"], params["theta"])
            pi = params["pi"]
            nb_dist = nbinom(n, p)

            p0 = pi + (1.0 - pi) * float(nb_dist.cdf(0))
            if q <= p0:
                return 0.0

            q_nb = (q - pi) / max(1.0 - pi, _EPS)
            q_nb = float(np.clip(q_nb, _EPS, 1.0 - _EPS))
            return float(nb_dist.ppf(q_nb))
        else:
            raise ValueError(f"Unknown parametric_form: {form}")

    def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
        m = 0.5 * (p + q)
        # KL(p||m) and KL(q||m)
        kl_pm = np.sum(p * (np.log(p + _EPS) - np.log(m + _EPS)))
        kl_qm = np.sum(q * (np.log(q + _EPS) - np.log(m + _EPS)))
        return float(0.5 * (kl_pm + kl_qm))

    def _wasserstein_1(p: np.ndarray, q: np.ndarray) -> float:
        # Exact 1-Wasserstein on integers with unit spacing: sum_k |CDF_p(k)-CDF_q(k)|
        cdf_p = np.cumsum(p)
        cdf_q = np.cumsum(q)
        return float(np.sum(np.abs(cdf_p - cdf_q)))

    x_obs = np.asarray(x_obs, dtype=float)
    x_pred = np.asarray(x_pred, dtype=float)
    if x_obs.ndim == 1:
        x_obs = x_obs.reshape(1, -1)
    if x_pred.ndim == 1:
        x_pred = x_pred.reshape(1, -1)

    if x_obs.ndim != 2 or x_pred.ndim != 2:
        raise ValueError(f"x_obs and x_pred must be 1D or 2D arrays, got {x_obs.ndim}D and {x_pred.ndim}D.")

    if x_obs.shape[1] != x_pred.shape[1]:
        raise ValueError(
            f"x_obs and x_pred must have the same number of genes, "
            f"got {x_obs.shape[1]} vs {x_pred.shape[1]}."
        )
    if x_obs.shape[0] == 0 or x_pred.shape[0] == 0:
        return float("nan")

    # Count models: clip negatives
    x_obs = np.clip(x_obs, 0.0, None)
    x_pred = np.clip(x_pred, 0.0, None)

    form = parametric_form.strip().upper()
    dist = dist_type.strip().upper()

    n_cells_obs, n_genes = x_obs.shape
    n_cells_pred = x_pred.shape[0]
    mu_obs = x_obs.mean(axis=0)
    mu_pred = x_pred.mean(axis=0)
    var_obs = x_obs.var(axis=0, ddof=1) if n_cells_obs > 1 else np.zeros(n_genes)
    var_pred = x_pred.var(axis=0, ddof=1) if n_cells_pred > 1 else np.zeros(n_genes)
    p0_obs = (x_obs == 0).mean(axis=0)
    p0_pred = (x_pred == 0).mean(axis=0)

    dists = []

    for g in range(n_genes):
        if form == "POISSON":
            p_params = _fit_poisson(mu_obs[g])
            q_params = _fit_poisson(mu_pred[g])
        elif form == "NB":
            p_params = _fit_nb(mu_obs[g], var_obs[g])
            q_params = _fit_nb(mu_pred[g], var_pred[g])
        elif form == "ZINB":
            p_params = _fit_zinb(mu_obs[g], var_obs[g], p0_obs[g])
            q_params = _fit_zinb(mu_pred[g], var_pred[g], p0_pred[g])
        else:
            raise ValueError('parametric_form must be one of {"NB","Poisson","ZINB"}')

        # choose truncation support
        data_max = int(max(np.max(x_obs[:, g]), np.max(x_pred[:, g])))
        k1 = _ppf(p_params, q_tail, form)
        k2 = _ppf(q_params, q_tail, form)
        kmax = int(min(max(data_max, k1, k2), kmax_cap))
        ks = np.arange(kmax + 1, dtype=int)

        p_pmf = _pmf(p_params, ks, form)
        q_pmf = _pmf(q_params, ks, form)

        if dist in {"JS-DIVERGENCE", "JS"}:
            d = _js_divergence(p_pmf, q_pmf)
        elif dist in {"WASSERSTEIN", "WASSERSTEIN-1", "W1"}:
            d = _wasserstein_1(p_pmf, q_pmf)
        else:
            raise ValueError('dist_type must be "JS-divergence" or "Wasserstein"')

        if np.isfinite(d):
            dists.append(d)

    return float(np.mean(dists)) if len(dists) > 0 else float("nan")

import argparse
import numpy as np
import pandas as pd
import os
import time
import tempfile
from datetime import datetime
import multiprocessing
from tqdm import tqdm
from scipy import sparse

import anndata as ad
from anndata.experimental import AnnCollection

from .util import est_cost, systematic_variation, intra_correlation, sum_and_sumsq, get_pseudobulks_and_degs
from analyses.evaluator import evaluation, get_eval_perturbation_ids
from analyses.context_splitter import ContextSplitter
from metrics.reconstruction.vendi_score import vendi_score
from data.dgp import synthetic_DGP, synthetic_causalDGP
from models.scvi_pert import ScviPerturbation
from util.anndata_util import AnnDataProxy

# Set OpenBLAS threads early if it was found to be helpful, otherwise optional
# os.environ["OPENBLAS_NUM_THREADS"] = "1"

_GLOBAL = {}
_PARAM_RANGES = {
    'G': {'type': 'log2_seq', 'min': 1024, 'max': 8192}, # 1000, 8192, g
    'N0': {'type': 'log2_seq', 'min': 128, 'max': 1024}, # 10, 1024, n_0
    'Nk': {'type': 'log2_seq', 'min': 128, 'max': 512}, # 128, 256, n_p
    'P': {'type': 'int', 'min': 10, 'max': 100}, # 10, 100, k
    'p_effect': {'type': 'float', 'min': 0.001, 'max': 0.1}, # 0.001, 0.1, delta
    'effect_factor': {'type': 'float', 'min': 1.2, 'max': 5.0}, # 1.2, 5.0, epsilon
    'B': {'type': 'float', 'min': 0.0, 'max': 2.0}, # 0.0, 2.0, beta
    'mu_l': {'type': 'float', 'min': 0.2, 'max': 5.0} # 0.2, 5.0, mu_l
}
_MODELS = ["Control", "Average", "scVI"]
# _MODELS = ["scVI"]
_NORM_LAYER_KEY = "normalized_log1p"


def simulate_one_run(
    dataset_name: str,
    G: int=10_000,   # number of genes
    N0: int=3_000,   # number of control cells
    Nk: int=150,     # number of perturbed cells per perturbation
    P: int=50,       # number of perturbations
    p_effect: float=0.01,  # a threshold for fraction of genes affected per perturbation
    effect_factor: float=2.0,  # effect factor for affected genes, epsilon in the paper
    B: float=0.0,      # global perturbation bias factor, beta in the paper
    mu_l: float=1.0,   # mean of log library size
    all_theta: np.ndarray | None = None, # Theta parameter for all cells , size of total number of genes in the real dataset (>= G)
    control_mu: np.ndarray | None = None, # Control mu parameters, size of total number of genes in the real dataset (>= G)
    pert_mu: np.ndarray | None = None, # Perturbed mu parameters, size of total number of genes in the real dataset (>= G)
    trial_id_for_rng: int | None = None, # Optional for seeding RNG per trial,
    model: str="Average", # The prediction model to use
    normalize: bool=True, # Whether to normalize the data
    max_cells_per_chunk: int=2048, # Count-generation chunk size
    ann_batch_size: int=1024, # AnnCollection batch size for scanpy processing
    split_strategy: str="in-context", # Data splitting strategy for evaluation
):
    """
    Simulate one experiment using chunked AnnData/AnnCollection processing.
    This avoids materializing the full (cells x genes) matrix in memory.
    """
    if dataset_name == "synthetic_one" and split_strategy == "cross-context":
        print("Warning: 'cross-context' split strategy is not applicable to 'synthetic_one' dataset." \
        " Defaulting to 'in-context' split strategy for this dataset.")
        split_strategy = "in-context"

    # Setup temporary directory for chunked data
    # The directory and its contents will be deleted after use
    with tempfile.TemporaryDirectory(prefix=f"synthetic_trial_{trial_id_for_rng}_", dir="/tmp") as tmp_dir:
        if dataset_name == "synthetic_one":
            chunk_paths, affected_genes = synthetic_DGP(
                G=G,
                N0=N0,
                Nk=Nk,
                P=P,
                p_effect=p_effect,
                effect_factor=effect_factor,
                B=B,
                mu_l=mu_l,
                all_theta=all_theta,
                control_mu=control_mu,
                pert_mu=pert_mu,
                seed=trial_id_for_rng,
                output_dir=tmp_dir,
                max_cells_per_chunk=max_cells_per_chunk,
                normalize=normalize,
                normalized_layer_key=_NORM_LAYER_KEY,
            )
        elif dataset_name == "synthetic_two":
            chunk_paths, _ = synthetic_causalDGP(
                G=G,
                N0=N0,
                Nk=Nk,
                P=P,
                mu_l=mu_l,
                all_theta=all_theta,
                mask_method='Power-law',
                seed=trial_id_for_rng,
                output_dir=tmp_dir,
                max_cells_per_chunk=max_cells_per_chunk,
                normalize=normalize,
                normalized_layer_key=_NORM_LAYER_KEY,
            )
            # TODO: for this synthetic data, affected_genes has two parts (each for one of the cell types)
            affected_genes = None
        else:
            raise ValueError(f"Unsupported dataset_name: {dataset_name}")
        # Memmaps keep large (P x G) accumulators off RAM while still supporting ndarray ops.
        # pert_sum is the sum of expressions for each gene in each perturbation, used to calculate means.
        # pert_sumsq is the sum of squared expressions for each gene in each perturbation, used to calculate variances.
        pert_sum = np.memmap(
            os.path.join(tmp_dir, "pert_sum.dat"),
            mode="w+",
            dtype=np.float32,
            shape=(P, G),
        )
        pert_sumsq = np.memmap(
            os.path.join(tmp_dir, "pert_sumsq.dat"),
            mode="w+",
            dtype=np.float32,
            shape=(P, G),
        )
        pert_sum[:] = 0.0
        pert_sumsq[:] = 0.0
        pert_counts = np.zeros(P, dtype=np.int64)

        control_sum = np.zeros(G, dtype=np.float64)
        control_sumsq = np.zeros(G, dtype=np.float64)
        control_count = 0

        pool_sum = np.zeros(G, dtype=np.float64)

        total_cells = N0 + Nk * P
        total_entries = total_cells * G
        total_nonzero = 0
        library_sizes = np.empty(total_cells, dtype=np.float64)
        current_idx = 0

        # Read chunk files lazily and iterate in observation batches.
        backed_chunks = [ad.read_h5ad(path, backed="r") for path in chunk_paths]
        try:
            collection = AnnCollection(
                backed_chunks,
                join_vars="inner",
                join_obsm='inner',
                label="chunk_id",
                keys=[str(i) for i in range(len(backed_chunks))],
                index_unique="-",
            )

            # Calculate global metrics in batches to avoid loading all cells at once.
            for batch_view, _ in collection.iterate_axis(batch_size=ann_batch_size, axis=0, shuffle=False):
                batch_counts = batch_view.X

                if sparse.issparse(batch_counts):
                    total_nonzero += int(batch_counts.nnz)
                    batch_library_sizes = np.asarray(batch_counts.sum(axis=1)).ravel().astype(np.float64, copy=False)
                else:
                    batch_counts_dense = np.asarray(batch_counts)
                    total_nonzero += int(np.count_nonzero(batch_counts_dense))
                    batch_library_sizes = batch_counts_dense.sum(axis=1, dtype=np.float64)
                batch_size_actual = int(batch_library_sizes.shape[0])
                next_idx = current_idx + batch_size_actual
                library_sizes[current_idx:next_idx] = batch_library_sizes
                current_idx = next_idx

                batch_log = batch_view.layers[_NORM_LAYER_KEY]

                perturbation_ids = batch_view.obs["perturbation"].to_numpy(dtype=np.int32, copy=False)
                for perturbation_id in np.unique(perturbation_ids):
                    # Aggregate sufficient statistics per perturbation ID.
                    row_mask = perturbation_ids == perturbation_id
                    batch_subset = batch_log[row_mask]
                    subset_count = int(batch_subset.shape[0])
                    subset_sum, subset_sumsq = sum_and_sumsq(batch_subset)

                    if perturbation_id == -1:
                        control_sum += subset_sum
                        control_sumsq += subset_sumsq
                        control_count += subset_count
                        continue

                    if perturbation_id < 0 or perturbation_id >= P:
                        raise ValueError(f"Invalid perturbation id encountered: {perturbation_id}")

                    pert_sum[perturbation_id, :] += subset_sum.astype(np.float32, copy=False)
                    pert_sumsq[perturbation_id, :] += subset_sumsq.astype(np.float32, copy=False)
                    pert_counts[perturbation_id] += subset_count
                    pool_sum += subset_sum
        finally:
            # even if we error out, make sure to close the open file handles for backed AnnData objects
            for backed_adata in backed_chunks:
                if getattr(backed_adata, "file", None) is not None:
                    backed_adata.file.close()

        # Basic sanity checks before proceeding with metric calculations
        assert control_count == N0, f"Control cell count mismatch: expected {N0}, got {control_count}"
        assert pert_counts.sum() == Nk * P, f"Perturbed cell count mismatch: expected {Nk * P}, got {pert_counts.sum()}"

        # Compute control mean and perturbed mean
        mu_control = (control_sum / N0).astype(np.float32, copy=False)
        mu_pool = (pool_sum / (Nk * P)).astype(np.float32, copy=False)

        # Compute observed means per perturbation
        mu_obs = np.empty((P, G), dtype=np.float32)
        for p_idx in range(P):
            mu_pert_float64 = pert_sum[p_idx, :].astype(np.float64, copy=False) / pert_counts[p_idx]
            mu_obs[p_idx, :] = mu_pert_float64.astype(np.float32, copy=False)

        sparsity = 1.0 - (total_nonzero / total_entries)
        median_library_size = np.nanmedian(library_sizes[:current_idx]) if current_idx > 0 else np.nan

        sys_var = systematic_variation(
            ptb_shifts=mu_obs - mu_control,
            avg_ptb_shift=mu_pool - mu_control,
        )
        intra_corr = intra_correlation(mu_obs - mu_control)
        vs = vendi_score(
            ac=collection,
            ac_batch_size=ann_batch_size,
            n_pca_components=30,
            sample_size=2_000,
            random_state=0 if trial_id_for_rng is None else int(trial_id_for_rng),
            layer_key=_NORM_LAYER_KEY,
        )

        data_stats = {
            'sparsity': sparsity,
            'median_library_size': median_library_size,
            'systematic_variation': sys_var,
            'intra_corr': intra_corr,
            'vendi_score': vs,
        }

        # Split indices 
        proxy = AnnDataProxy(collection.obs)
        splitter = ContextSplitter(
            adata=proxy,
            train_frac=0.7,
            val_frac=0.1,
            test_frac=0.2,
            context_mode="cell" if split_strategy == "cross-context" else None,
            perturbation_key="perturbation",
            cell_line_key="cell_line",
            donor_key="donor",
            holdout_context_values=[1] if split_strategy == "cross-context" else None,
            control_label=-1,
        )
        train_idx, val_idx, test_idx = splitter.split(seed=trial_id_for_rng)

        ac_train_val = collection[np.concatenate([train_idx, val_idx])]
        ac_test = collection[test_idx]

        mu_control_train, mu_pool_train, _ = get_pseudobulks_and_degs(
            ac_view=ac_train_val,
            ac_batch_size=ann_batch_size,
            return_degs=False,
            layer_key=_NORM_LAYER_KEY,
        )

        mu_control_test, mu_pool_test, degs_test = get_pseudobulks_and_degs(
            ac_view=ac_test,
            ac_batch_size=ann_batch_size,
            return_degs=True,
            alpha=p_effect,
            method="t-test",
            layer_key=_NORM_LAYER_KEY,
        )
        test_perturbation_ids = get_eval_perturbation_ids(obs=ac_test, control_label=-1, strict_match=False)
        test_perturbation_ids = np.asarray(test_perturbation_ids, dtype=np.int32)
        mu_obs_test = mu_obs[test_perturbation_ids, :]

        all_results = []
        for model in _MODELS:
            start_time = time.time()
            mu_pred = None
            ad_test_pred = None
            if model == "Control":
                mu_pred = np.tile(mu_control_train, (test_perturbation_ids.size, 1))
            elif model == "Average":
                mu_pred = np.tile(mu_pool_train, (test_perturbation_ids.size, 1))
            elif model == "scVI":
                scvi_model = ScviPerturbation(
                    data=collection,
                    counts_layer="counts",
                    perturbation_key="perturbation", # NOTE: check if this is useful
                    train_idx=train_idx,
                    val_idx=val_idx,
                    test_idx=test_idx,
                    seed=trial_id_for_rng if trial_id_for_rng is not None else 42,
                )
                ad_test_pred = scvi_model.run(
                    n_latent=10,
                    n_hidden=128,
                    n_layers=3,
                    gene_likelihood="zinb",
                    dispersion="gene",
                    max_epochs=800,
                    batch_size=512,
                    early_stopping=True,
                    dataloader_num_workers=0,
                )
            else:
                raise NotImplementedError(f"Model '{model}' is not implemented.")

            model_results = evaluation(
                pred=ad_test_pred,
                obs=ac_test,
                mu_pred=mu_pred,
                mu_obs=mu_obs_test,
                mu_control_obs=mu_control_test,
                mu_pool_obs=mu_pool_test,
                true_DEGs=affected_genes,
                DEGs_stats=degs_test,
                perturbation_ids=test_perturbation_ids,
                model=model,
                layer_obs=_NORM_LAYER_KEY,
                control_label=-1,
            )

            print(model_results)

            model_results.update({
                'model': model,
                'execution_time': time.time() - start_time,
                **data_stats
            })
            all_results.append(model_results)

        return all_results


def sample_parameters(param_ranges): # Unchanged from original
    params = {}
    for param, range_info in param_ranges.items():
        if range_info['type'] == 'int':
            params[param] = np.random.randint(range_info['min'], range_info['max'] + 1)
        elif range_info['type'] == 'float':
            params[param] = np.random.uniform(range_info['min'], range_info['max'])
        elif range_info['type'] == 'log2_seq':
            log_min = int(np.log2(range_info['min']))
            log_max = int(np.log2(range_info['max']))
            log_value = np.random.uniform(log_min, log_max)
            params[param] = int(2 ** log_value)
        elif range_info['type'] == 'fixed':
            params[param] = range_info['value']
    return params

def init_worker(control_mu, all_theta, pert_mu):
    _GLOBAL["control_mu"] = control_mu
    _GLOBAL["all_theta"] = all_theta
    _GLOBAL["pert_mu"] = pert_mu

# Revised _pool_worker to include timing (matches spirit of original)
def _pool_worker_timed(task_info_dict):
    dataset_name = task_info_dict['dataset_name']
    trial_id = task_info_dict['trial_id']
    params_dict = task_info_dict['params_dict']
    split_strategy = task_info_dict['split_strategy']
    control_mu_from_main = _GLOBAL["control_mu"]
    all_theta_from_main  = _GLOBAL["all_theta"]
    pert_mu_from_main    = _GLOBAL["pert_mu"]

    # Add trial_id for RNG seeding within simulate_one_run_numpy
    params_for_sim = params_dict.copy() # Avoid modifying original params_dict
    params_for_sim['dataset_name'] = dataset_name
    params_for_sim['trial_id_for_rng'] = trial_id
    params_for_sim['split_strategy'] = split_strategy
    params_for_sim['control_mu'] = control_mu_from_main
    params_for_sim['all_theta'] = all_theta_from_main
    params_for_sim['pert_mu'] = pert_mu_from_main
    
    try:
        # Ensure all required keys by simulate_one_run_numpy are in params_for_sim
        # G, N0, Nk, P, p_effect, effect_factor are expected from sample_parameters
        results_per_sim = simulate_one_run(**params_for_sim)
        
        # Prepare results: original sampled params + metrics + supporting info
        # `params_dict` is the original sampled params.
        final_results_per_sim = []
        for results_per_sim_model in results_per_sim:
            final_results_per_sim.append({
                **params_dict, 
                **results_per_sim_model, 
                'trial_id': trial_id, 
                'status': 'success'}
            )
        return final_results_per_sim
        
    except Exception as e:
        # Define metrics_error_keys locally for safety
        metrics_error_keys_local = { 
            'pearson', 'pearson_true_degs', 'pearson_degs',
            'mae', 'mae_true_degs', 'mae_degs',
            'mse', 'mse_true_degs', 'mse_degs',
            'r2', 'r2_true_degs', 'r2_degs',
            'parametric_distance', 'mmd_distance', 
            'vendi_score_pred', 'vendi_score_obs',
            'pds_l1', 'pds_l2', 'pds_cosine',
            'model', 'execution_time',
            'sparsity', 'median_library_size', 'systematic_variation', 'intra_corr', 'vendi_score',
        }
        metrics_error = {key: np.nan for key in metrics_error_keys_local}

        final_result_error = {
            **params_dict, # original sampled params
            **metrics_error,
            'trial_id': trial_id,
            'status': 'failed',
            'error': str(e)
        }
        return [final_result_error] * len(_MODELS)


def run_random_sweep(
    dataset_name,
    n_trials,
    output_dir,
    control_mu=None,
    all_theta=None,
    pert_mu=None,
    num_workers=None,
    use_multiprocessing=True,
    max_cells_per_chunk=2048,
    ann_batch_size=1024,
    split_strategy="in-context",
):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = os.path.join(output_dir, f"random_sweep_results_{timestamp}.csv")
    error_log_file = os.path.join(output_dir, f"error_log_{timestamp}.txt")

    if use_multiprocessing:
        if num_workers is None:
            num_workers = os.cpu_count()
        print(f"Starting AnnData-backed random parameter sweep with {n_trials} trials using {num_workers} worker processes (spawn context).")
    else:
        print(f"Starting AnnData-backed random parameter sweep with {n_trials} trials using sequential execution.")

    tasks_for_pool = []
    for i in range(n_trials):
        params = sample_parameters(_PARAM_RANGES)
        params['max_cells_per_chunk'] = int(max_cells_per_chunk)
        params['ann_batch_size'] = int(ann_batch_size)
        tasks_for_pool.append({
            'trial_id': i, 
            'dataset_name': dataset_name,
            'params_dict': params,
            'split_strategy': split_strategy,
        })
    # sort tasks by estimated cost in descending order to optimize workload distribution
    tasks_for_pool.sort(key=lambda t: est_cost(t["params_dict"]), reverse=True)

    all_results_data = []
    
    if use_multiprocessing:
        print("Multiprocessing can be memory intensive, so if running into swap, reduce the number of workers.")
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(
            processes=num_workers, 
            initializer=init_worker, 
            initargs=(control_mu, all_theta, pert_mu),
            maxtasksperchild=100,
        ) as pool:
            print("\nProcessing trials (AnnData-backed version with worker timing):")
            with tqdm(total=n_trials, desc="Running Trials (AnnData-backed)") as pbar:
                for result_from_worker in pool.imap_unordered(_pool_worker_timed, tasks_for_pool):
                    all_results_data += result_from_worker
                    pbar.update(1)
    else:
        init_worker(control_mu, all_theta, pert_mu)
        print("\nProcessing trials (AnnData-backed version with worker timing):")
        with tqdm(total=n_trials, desc="Running Trials (AnnData-backed)") as pbar:
            for task in tasks_for_pool:
                result_from_worker = _pool_worker_timed(task)
                all_results_data += result_from_worker
                pbar.update(1)

    results_df = pd.DataFrame(all_results_data)
    
    success_count = results_df[(results_df['status'] == 'success') & (results_df["model"] == _MODELS[0])].shape[0] if 'status' in results_df else 0
    failure_count = n_trials - success_count

    if failure_count > 0 and 'status' in results_df: # Ensure 'status' column exists
        print("")
        failed_trials = results_df[results_df['status'] == 'failed']
        # Define metrics_error keys for excluding them from params logging
        metrics_error_keys = { 
            'pearson', 'pearson_true_degs', 'pearson_degs',
            'mae', 'mae_true_degs', 'mae_degs',
            'mse', 'mse_true_degs', 'mse_degs',
            'r2', 'r2_true_degs', 'r2_degs',
            'parametric_distance', 'mmd_distance', 
            'vendi_score_pred', 'vendi_score_obs',
            'pds_l1', 'pds_l2', 'pds_cosine',
            'sparsity', 'vendi_score'
        }
        with open(error_log_file, 'a') as f:
            for _, row in failed_trials.iterrows():
                # Ensure 'trial_id' and 'error' exist in row, provide defaults if not
                trial_id_val = int(row.get('trial_id', -1))
                error_val = row.get('error', 'Unknown error')
                
                error_params = {k: v for k, v in row.items() if k not in metrics_error_keys and k not in ['status', 'error', 'trial_id', 'execution_time']}
                f.write(f"Trial {trial_id_val + 1} failed\n")
                f.write(f"Parameters: {str(error_params)}\n")
                f.write(f"Error: {error_val}\n")
                f.write("-" * 80 + "\n")

    if not results_df.empty:
        results_df.to_csv(csv_file, index=False)
        print(f"\nSweep complete. Results saved to '{csv_file}'")
    else:
        print("\nSweep complete. No results to save.")

    print(f"Success: {success_count}/{n_trials} trials")
    print(f"Failed: {failure_count}/{n_trials} trials")
    if failure_count > 0:
        print(f"See error log for details: {error_log_file}")
    return csv_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run random sweep simulations.")
    parser.add_argument("--output_dir", type=str, default="results/synthetic_simulations/random_sweep_results", help="Directory to save sweep results")
    parser.add_argument("--n_trials", type=int, default=4, help="Number of trials to run")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of worker processes for multiprocessing")
    parser.add_argument("--multiprocessing", action="store_true", help="Enable multiprocessing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--max_cells_per_chunk", type=int, default=2048, help="Maximum cells per generated h5ad chunk")
    parser.add_argument("--ann_batch_size", type=int, default=1024, help="Batch size when iterating AnnCollection")
    parser.add_argument("--dataset", type=str, default="synthetic_one", choices=["synthetic_one", "synthetic_two"], help="Dataset to use for the simulation")
    parser.add_argument("--split_strategy", type=str, default="in-context", choices=["in-context", "cross-context"], help="Data splitting strategy for evaluation, cross-context is only available for synthetic_two")
    args = parser.parse_args()

    # Set seed
    np.random.seed(args.seed)
    
    # Load fitted parameters from parameter estimation files
    control_params_df = pd.read_csv("results/synthetic_simulations/parameter_estimation/control_fitted_params.csv", index_col=0)
    perturbed_params_df = pd.read_csv("results/synthetic_simulations/parameter_estimation/perturbed_fitted_params.csv", index_col=0)
    all_params_df = pd.read_csv("results/synthetic_simulations/parameter_estimation/all_fitted_params.csv", index_col=0)
    
    print("Using theta estimates from all cells combined")
    
    # Extract parameters for simulation
    control_mu = control_params_df['mu'].values
    pert_mu = perturbed_params_df['mu'].values
    
    # Use theta (n) from all cells estimation
    all_theta = all_params_df['n'].values
    
    print(f"Using {len(control_mu)} genes for simulation.")
    
    # Call the final version of run_random_sweep
    print("Running the sweep...")
    csv_file = run_random_sweep(
        args.dataset,
        args.n_trials, 
        args.output_dir, 
        control_mu=control_mu, 
        all_theta=all_theta,
        pert_mu=pert_mu,
        num_workers=args.num_workers, # num_worker should be around 0.6 * RAM / MAX_SPACE_PER_WORK
        use_multiprocessing=args.multiprocessing,
        max_cells_per_chunk=args.max_cells_per_chunk,
        ann_batch_size=args.ann_batch_size,
        split_strategy=args.split_strategy,
    ) 
    print("\nDone doing the sweep. Plotting results...")

    # Run uv run python simulations/simulation_plots.py
    os.system(f"uv run python analyses/synthetic_simulations/paper_plots.py --results {csv_file}")

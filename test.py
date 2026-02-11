import json  # Read JSON split/config file used by the benchmark
from pathlib import Path  # Build filesystem paths safely

import anndata as ad  # Read .h5ad single-cell files
import numpy as np  # Numeric arrays and linear algebra
import pandas as pd  # Table operations for metadata/embeddings
from scipy import sparse  # Sparse matrix checks/conversions
from sklearn.decomposition import PCA


def _to_dense(x):
    """Convert sparse matrix to dense ndarray; leave dense arrays as ndarray."""
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


def _r_match(x, table):
    """
    R-like match(x, table):
    - returns first index in `table` for each value in `x`
    - returns -1 when not found (R would return NA)
    """
    first = {}  # Map value -> first index where it appears in table
    for i, val in enumerate(table):  # Scan table once
        first.setdefault(val, i)  # Keep only first occurrence
    return np.array([first.get(val, -1) for val in x], dtype=int)  # Vectorized lookup output


def fit_linear(Y, G, P_transpose, G_ridge=0.01, P_ridge=0.01):
    """
    Solve the objective for the linear method:
        argsmin_W ||Y - ( G @ W @ P_transpose + b) ||_F^2 + G_ridge * ||G @ W||_F^2 + P_ridge * ||W @ P_transpose||_F^2 + G_ridge * P_ridge * ||W||_F^2
    Compare with the paper description, the code add penalties for numerical stability. 
    This implements the closed-form solution used in the R code for the branch where both G and P are provided. 
    So we are trying to this behaviorally equivalent.
    https://github.com/const-ae/linear_perturbation_prediction-Paper/blob/main/benchmark/src/run_linear_pretrained_model.R
    """
    # Per-gene mean across perturbations (row means), serve as the residual term b
    b = Y.mean(axis=1, keepdims=True)  
    Y_center = Y - b 

    aa = G.T @ G + G_ridge * np.eye(G.shape[1], dtype=float)  # (G'G + lambda I)
    bb = P_transpose @ P_transpose.T + P_ridge * np.eye(P_transpose.shape[0], dtype=float)  # (PP' + lambda I)

    # Closed-form ridge solution used in the R implementation
    W = np.linalg.solve(aa, G.T @ Y_center @ P_transpose.T) @ np.linalg.solve(bb, np.eye(bb.shape[0]))

    # In case of numerical issues, ensure no NaNs in output
    W[np.isnan(W)] = 0.0
    return {"W": W, "center": b.ravel()}


def pseudobulk_sum(adata, group_cols):
    """
    Aggregate single-cell data into pseudobulk samples by summing expression across cells in each group.
    
    Arguments:
        adata: AnnData object containing single-cell data (adata.X) and metadata (adata.obs)
        group_cols: List of column names in adata.obs to group by 
    Returns:
        - X_pb: np.ndarray of shape (n_genes, n_groups), summed expression per gene per group
        - pb_obs: pd.DataFrame of shape (n_groups, len(group_cols)), group-level metadata corresponding to columns of X_pb
    """
    # TODO: change the AnnData structure to more memory efficient AnnCollection
    grouped = adata.obs.groupby(group_cols, sort=False, dropna=False).indices  # Group cell indices by metadata keys
    X = adata.X  # Cell x gene matrix

    cols = []  # Will collect one aggregated expression vector per group
    rows = []  # Will collect one metadata row (group key tuple) per group

    for key, idx in grouped.items():  # Iterate over each group
        key = key if isinstance(key, tuple) else (key,)  # Normalize key to tuple
        if sparse.issparse(X):  # Sparse-safe aggregation branch
            col = np.asarray(X[idx].sum(axis=0)).ravel()  # Sum selected rows (cells) into one gene vector
        else:  # Dense aggregation branch
            col = np.asarray(X[idx]).sum(axis=0).ravel()  # Same aggregation for dense arrays
        cols.append(col)  # Store aggregated gene vector
        rows.append(key)  # Store group labels

    X_pb = np.column_stack(cols)  # genes x groups
    pb_obs = pd.DataFrame(rows, columns=group_cols)  # Group-level metadata
    return X_pb, pb_obs


# ================================
# Load benchmark-ready perturbation data
# ================================

# Assumes `pa` already exists (from argparse or equivalent), with:
# pa.dataset_name, pa.test_train_config_id, pa.pca_dim, pa.ridge_penalty,
# pa.gene_embedding, pa.pert_embedding, pa.working_dir, pa.seed

rng = np.random.default_rng(pa.seed)  # Deterministic random generator for "random" embeddings

folder = Path("data/gears_pert_data")  # Base benchmark data directory
sce = ad.read_h5ad(folder / pa.dataset_name / "perturb_processed.h5ad")  # Read AnnData file

# Read split definition JSON (train/test/holdout condition lists)
with open(Path(pa.working_dir) / "results" / pa.test_train_config_id, "r") as f:
    set2condition = json.load(f)

# Ensure "ctrl" is part of training split, mirroring the R logic
if "ctrl" not in set2condition["train"]:
    set2condition["train"] = list(set2condition["train"]) + ["ctrl"]

# Keep only cells whose condition appears in any split list
valid_conditions = {c for conds in set2condition.values() for c in conds}
sce = sce[sce.obs["condition"].isin(valid_conditions)].copy()

# Clean/derive condition strings
sce.obs["condition"] = sce.obs["condition"].astype(str)  # Ensure string dtype
sce.obs["clean_condition"] = sce.obs["condition"].str.replace(r"\+ctrl", "", regex=True)  # Remove trailing "+ctrl"

# Build mapping: condition -> split label (train/test/holdout)
cond_to_training = {}
for training_name, conds in set2condition.items():
    for cond in conds:
        cond_to_training.setdefault(cond, training_name)  # First assignment wins, like R join first-match behavior
sce.obs["training"] = sce.obs["condition"].map(cond_to_training)  # Add split label per cell

# Use gene_name as row identifier (same as rownames(sce) <- gene_names in R)
gene_names = sce.var["gene_name"].astype(str).to_numpy()
sce.var_names = gene_names

# Compute baseline = mean expression across all control cells
X_cells_genes = sce.X  # Shape: cells x genes
ctrl_mask = (sce.obs["condition"].to_numpy() == "ctrl")  # Control-cell selector
if sparse.issparse(X_cells_genes):
    baseline = np.asarray(X_cells_genes[ctrl_mask].mean(axis=0)).ravel()  # Sparse mean across control cells
else:
    baseline = np.asarray(X_cells_genes[ctrl_mask]).mean(axis=0)  # Dense mean across control cells

# TODO: important things starts here, baseline is the average prediction.


# Pseudobulk by (condition, clean_condition, training)
psce_X, psce_obs = pseudobulk_sum(sce, ["condition", "clean_condition", "training"])  # genes x pseudobulk-groups
change_X = psce_X - baseline[:, None]  # Per-group change from baseline

# Restrict to training pseudobulk groups
train_mask = (psce_obs["training"].to_numpy() == "train")
train_X = psce_X[:, train_mask]  # Training expression matrix (genes x train-groups)
train_change = change_X[:, train_mask]  # Training delta matrix (genes x train-groups)
train_clean_condition = psce_obs.loc[train_mask, "clean_condition"].astype(str).tolist()  # Train condition names
train_gene_names = sce.var_names.astype(str).tolist()  # Gene identifiers in matrix row order

# ================================
# Get embeddings
# ================================

# Build gene embedding matrix (rows keyed by gene names)
if pa.gene_embedding == "training_data":
    pca = PCA(n_components=pa.pca_dim, random_state=pa.seed)  # PCA model
    x = pca.fit_transform(_to_dense(train_X))  # Fit on genes x train-groups -> genes x pca_dim scores
    gene_emb = pd.DataFrame(x, index=train_gene_names)  # Keep gene names as row index
elif pa.gene_embedding == "identity":
    gene_emb = pd.DataFrame(np.eye(psce_X.shape[0]), index=train_gene_names)  # One-hot gene embedding
elif pa.gene_embedding == "zero":
    gene_emb = pd.DataFrame(np.zeros((psce_X.shape[0], psce_X.shape[0])), index=train_gene_names)  # All-zero gene embedding
elif pa.gene_embedding == "random":
    gene_emb = pd.DataFrame(
        rng.standard_normal((psce_X.shape[0], pa.pca_dim)),  # Random gene features
        index=train_gene_names,
    )
else:
    g = pd.read_csv(pa.gene_embedding, sep="\t")  # Load external TSV
    gene_emb = pd.DataFrame(g.to_numpy().T, index=g.columns.astype(str))  # Match R: transpose file matrix

# Build perturbation embedding matrix (columns keyed by perturbation labels)
pca = PCA(n_components=pa.pca_dim, random_state=pa.seed)  # PCA model
x = pca.fit_transform(_to_dense(train_X))  # Same source matrix as R branch
pert_emb = pd.DataFrame(x.T, columns=train_gene_names)  # Match R: t(pca$x), so columns are rownames(pca$x)

# Ensure a ctrl column exists in perturbation embedding
if "ctrl" not in pert_emb.columns:
    pert_emb["ctrl"] = 0.0  # Zero embedding for control perturbation

# Match embedding identifiers to training matrix identifiers
pert_matches = _r_match(pert_emb.columns.astype(str).tolist(), train_clean_condition)  # pert_emb cols -> train conditions
gene_matches = _r_match(gene_emb.index.astype(str).tolist(), train_gene_names)  # gene_emb rows -> train genes

# Same safety checks as R
if np.sum(pert_matches >= 0) <= 1:
    raise ValueError("Too few matches between clean_conditions and pert_embedding")
if np.sum(gene_matches >= 0) <= 1:
    raise ValueError("Too few matches between gene names and gene_embedding")

# Keep only matched entities
gene_keep = (gene_matches >= 0)  # Boolean selector for rows in gene_emb that matched train genes
pert_keep = (pert_matches >= 0)  # Boolean selector for cols in pert_emb that matched train conditions

gene_emb_sub = gene_emb.iloc[gene_keep, :].to_numpy()  # Matched gene embedding matrix A
pert_emb_training = pert_emb.iloc[:, pert_keep].to_numpy()  # Matched perturbation embedding matrix B

# Build Y using matched gene row order and matched train-condition column order
Y = train_change[np.asarray(gene_matches[gene_keep], dtype=int), :][:, np.asarray(pert_matches[pert_keep], dtype=int)]

# Fit model coefficients
coefs = fit_linear(
    Y=Y,
    G=gene_emb_sub,
    P_transpose=pert_emb_training,
    G_ridge=pa.ridge_penalty,
    B_ridge=pa.ridge_penalty,
)

# ================================
# Predict held-out perturbations
# ================================

# Match all pseudobulk clean_condition values to pert_emb columns
pert_matches_all = _r_match(psce_obs["clean_condition"].astype(str).tolist(), pert_emb.columns.astype(str).tolist())

pert_emb_np = pert_emb.to_numpy()  # Dense numeric perturbation embedding
pert_emb_all = np.full((pert_emb_np.shape[0], len(pert_matches_all)), np.nan, dtype=float)  # Placeholder for all groups
ok = pert_matches_all >= 0  # Which pseudobulk groups found an embedding column
pert_emb_all[:, ok] = pert_emb_np[:, pert_matches_all[ok]]  # Fill matched columns

baseline_sub = baseline[np.asarray(gene_matches[gene_keep], dtype=int)]  # Baseline aligned to matched gene order

# Prediction formula from R:
# pred = A @ K @ B_all + b + baseline
pred = gene_emb_sub @ coefs["W"] @ pert_emb_all + coefs["center"][:, None] + baseline_sub[:, None]

pred_gene_names = np.asarray(train_gene_names)[np.asarray(gene_matches[gene_keep], dtype=int)]  # Predicted gene names
pred_df = pd.DataFrame(pred, index=pred_gene_names, columns=psce_obs["clean_condition"].astype(str).tolist())  # Final predictions table

# Optional explicit held-out slice (if held-out groups are labeled "test")
heldout_pred_df = pred_df.loc[:, psce_obs["training"].astype(str).eq("test").to_numpy()]
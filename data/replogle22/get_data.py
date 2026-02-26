import scanpy as sc
import os
import subprocess as sp
from scipy.sparse import csr_matrix
import pandas as pd
from tqdm import tqdm
import numpy as np
from anndata.experimental import AnnCollection
from ..preprocess import compute_degs


np.random.seed(42)

data_k562_url = 'https://zenodo.org/records/13350497/files/ReplogleWeissman2022_K562_essential.h5ad?download=1'
data_rpe1_url = 'https://zenodo.org/records/13350497/files/ReplogleWeissman2022_rpe1.h5ad?download=1'
data_cache_dir = './data/replogle22'

if not os.path.exists(data_cache_dir):
    os.makedirs(data_cache_dir)

data_k562_path = f'{data_cache_dir}/replogle22_K562_essential.h5ad'
data_rpe1_path = f'{data_cache_dir}/replogle22_RPE1.h5ad'

if not os.path.exists(data_k562_path):
    sp.call(f'wget -q {data_k562_url} -O {data_k562_path}', shell=True)

if not os.path.exists(data_rpe1_path):
    sp.call(f'wget -q {data_rpe1_url} -O {data_rpe1_path}', shell=True)

adata_k562 = sc.read_h5ad(data_k562_path)
adata_rpe1 = sc.read_h5ad(data_rpe1_path)

# concat the two adata objects using AnnCollection
ac = AnnCollection(
    [adata_k562, adata_rpe1],
    join_vars="inner",
    keys=["k562", "rpe1"],
    index_unique="-",
)

# Filter perturbations with too few cells
MIN_CELLS_PER_PERT_PER_CELLLINE = 256
ct = (
    ac.obs
    .pivot_table(index="perturbation", columns="cell_line", aggfunc="size", fill_value=0)
    .sort_index()
)
valid_1st_cell = ct.iloc[:, 0] > MIN_CELLS_PER_PERT_PER_CELLLINE
valid_2nd_cell = ct.iloc[:, 1] > MIN_CELLS_PER_PERT_PER_CELLLINE
print(f"Number of perturbations with > {MIN_CELLS_PER_PERT_PER_CELLLINE} cells in K562: {valid_1st_cell.sum()}")
print(f"Number of perturbations with > {MIN_CELLS_PER_PERT_PER_CELLLINE} cells in RPE1: {valid_2nd_cell.sum()}")

valid_perts = ct[(ct.iloc[:, 0] > MIN_CELLS_PER_PERT_PER_CELLLINE) &
                 (ct.iloc[:, 1] > MIN_CELLS_PER_PERT_PER_CELLLINE)].index

obs_mask = ac.obs["perturbation"].isin(valid_perts).to_numpy()
obs_idx = np.flatnonzero(obs_mask)

# now materialize only kept cells
adata = ac[obs_idx, :].to_adata()

# Rename columns only when the destination column does not already exist.
# Otherwise pandas creates duplicate column names, which breaks AnnData slicing.
adata.obs.rename(columns = {
    'mitopercent': 'percent_mito',
}, inplace=True)
adata.obs['perturbation'] = adata.obs['perturbation'].str.replace('_', '+')
adata.obs['perturbation'] = adata.obs['perturbation'].astype('category')
adata.obs['condition'] = adata.obs['perturbation'].copy()
adata.X = csr_matrix(adata.X)

# Filter cells
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

# Stash raw counts
adata.layers['counts'] = adata.X.copy()

# For every perturbation, for every gene, calculate the mean and variance of the counts
mean_df = pd.DataFrame(index=adata.var_names, columns=adata.obs['condition'].unique())
disp_df = pd.DataFrame(index=adata.var_names, columns=adata.obs['condition'].unique())
for pert in tqdm(adata.obs['condition'].unique()):
    pert_mask = (adata.obs['condition'].to_numpy() == pert)
    pert_counts = adata.X[pert_mask].toarray()
    mean_df.loc[:, pert] = np.mean(pert_counts, axis=0)
    disp_df.loc[:, pert] = np.var(pert_counts, axis=0)

# Save to the uns dictionary
mean_df_dict = mean_df.to_dict(orient='list')
disp_df_dict = disp_df.to_dict(orient='list')
adata.uns['mean_dict'] = mean_df_dict
adata.uns['disp_dict'] = disp_df_dict
adata.uns['mean_disp_dict_genes'] = disp_df.index.tolist()


# Do library size norm and log1p
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Get 8192 HVGs -- subset the adata object to only include the HVGs
sc.pp.highly_variable_genes(adata, n_top_genes=8192, subset=True)

# Do PCA
sc.pp.pca(adata)

# Do UMAP
sc.pp.neighbors(adata)
sc.tl.umap(adata)

# Calculate DEGs between each perturbation and all other perturbations
_ = compute_degs(adata, mode='vsrest')

# Calculate DEGs with respect to the control perturbation
_ = compute_degs(adata, mode='vscontrol')

# Convert to format that can be saved
SCORE_TYPE = 'scores' # or 'logfoldchanges'
names_df_vsrest = pd.DataFrame(adata.uns["rank_genes_groups_vsrest"]["names"])
scores_df_vsrest = pd.DataFrame(adata.uns["rank_genes_groups_vsrest"][SCORE_TYPE])
names_df_vsctrl = pd.DataFrame(adata.uns["rank_genes_groups_vscontrol"]["names"])
scores_df_vsctrl = pd.DataFrame(adata.uns["rank_genes_groups_vscontrol"][SCORE_TYPE])

# Save dataframes to csv
names_df_vsrest.to_pickle(f'{data_cache_dir}/replogle22_names_df_vsrest.pkl')
scores_df_vsrest.to_pickle(f'{data_cache_dir}/replogle22_scores_df_vsrest.pkl')
names_df_vsctrl.to_pickle(f'{data_cache_dir}/replogle22_names_df_vsctrl.pkl')
scores_df_vsctrl.to_pickle(f'{data_cache_dir}/replogle22_scores_df_vsctrl.pkl')

# Remove these from the adata object
adata.uns.pop('rank_genes_groups_vsrest', None)
adata.uns.pop('rank_genes_groups_vscontrol', None)
adata.uns.pop('rank_genes_groups', None)

# Save the data
output_data_path = f'{data_cache_dir}/replogle22_processed.h5ad'
adata.write_h5ad(output_data_path)
# Save adata.var to a CSV
genes_path = f'{data_cache_dir}/replogle22_genes.csv.gz'
adata.var.to_csv(genes_path)

import scanpy as sc
import os
import subprocess as sp
from scipy.sparse import csr_matrix
import pandas as pd
from tqdm import tqdm
import numpy as np
from ..preprocess import compute_degs

np.random.seed(42)

data_url = 'https://zenodo.org/records/7041849/files/NormanWeissman2019_filtered.h5ad?download=1'
data_cache_dir = './data/norman19'

if not os.path.exists(data_cache_dir):
    os.makedirs(data_cache_dir)

tmp_data_dir = f'{data_cache_dir}/norman19_downloaded.h5ad'

if not os.path.exists(tmp_data_dir):
    sp.call(f'wget -q {data_url} -O {tmp_data_dir}', shell=True)

adata = sc.read_h5ad(tmp_data_dir)

MIN_CELLS_PER_PERT_PER_CELLLINE = 256
ct = (
    adata.obs
    .pivot_table(index="perturbation", aggfunc="size", fill_value=0)
    .sort_index()
)
valid = ct.iloc[:] > MIN_CELLS_PER_PERT_PER_CELLLINE
print(f"Number of perturbations with > {MIN_CELLS_PER_PERT_PER_CELLLINE} cells: {valid.sum()}")
print(f"Number of perturbations in original data: {adata.obs['perturbation'].nunique()}")


# Rename columns
adata.obs.rename(columns = {
    'nCount_RNA': 'ncounts',
    'nFeature_RNA': 'ngenes',
    'percent.mt': 'percent_mito',
    'cell_line': 'cell_type',
}, inplace=True)
adata.obs['perturbation'] = adata.obs['perturbation'].str.replace('_', '+')
adata.obs['perturbation'] = adata.obs['perturbation'].astype('category')
adata.obs['condition'] = adata.obs.perturbation.copy()
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
    pert_cells = adata.obs[adata.obs['condition'] == pert].index.tolist()
    pert_counts = adata[pert_cells].X.toarray()
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
# logfc_df_vsctrl = pd.DataFrame(adata.uns["rank_genes_groups_vscontrol"]["logfoldchanges"])

# Save dataframes to csv
names_df_vsrest.to_pickle(f'{data_cache_dir}/norman19_names_df_vsrest.pkl')
scores_df_vsrest.to_pickle(f'{data_cache_dir}/norman19_scores_df_vsrest.pkl')
names_df_vsctrl.to_pickle(f'{data_cache_dir}/norman19_names_df_vsctrl.pkl')
scores_df_vsctrl.to_pickle(f'{data_cache_dir}/norman19_scores_df_vsctrl.pkl')
# logfc_df_vsctrl.to_pickle(f'{data_cache_dir}/norman19_logfc_df_vsctrl.pkl')

# Remove these from the adata object
adata.uns.pop('rank_genes_groups_vsrest', None)
adata.uns.pop('rank_genes_groups_vscontrol', None)
adata.uns.pop('rank_genes_groups', None)


# Save the data
output_data_path = f'{data_cache_dir}/norman19_processed.h5ad'
adata.write_h5ad(output_data_path)
# Save adata.var to a CSV
genes_path = f'{data_cache_dir}/norman19_genes.csv.gz'
adata.var.to_csv(genes_path)

# Norman19 `AnnData` Structure

This describes the processed dataset produced by [`get_data.py`](/home/shiang/projects/aip-rgreiner/shiang/diversity_by_design/data/norman19/get_data.py) and saved at:

- `data/norman19/norman19_processed.h5ad`

When loaded with `anndata`/`scanpy`, the object has:

- `adata.shape = (111445, 8192)` (cells x genes)
- `adata.obs_names` unique cell barcodes
- `adata.var_names` unique gene symbols

## `adata.X` and `adata.layers`

- `adata.X`
  - Type: `scipy.sparse.csr_matrix`
  - Dtype: `float32`
  - Shape: `(111445, 8192)`
  - Contents: library-size normalized (`target_sum=1e4`) and `log1p`-transformed expression
- `adata.layers["counts"]`
  - Type: `scipy.sparse.csr_matrix`
  - Dtype: `float32`
  - Shape: `(111445, 8192)`
  - Contents: raw counts after cell/gene filtering, before normalization/log transform

## `adata.obs` (cell metadata)

`adata.obs` has 22 columns:

- `guide_id` (`category`)
- `read_count` (`int64`)
- `UMI_count` (`int64`)
- `coverage` (`float64`)
- `gemgroup` (`int64`)
- `good_coverage` (`bool`)
- `number_of_cells` (`int64`)
- `tissue_type` (`category`)
- `cell_line` (`category`)
- `cancer` (`bool`)
- `disease` (`category`)
- `perturbation_type` (`category`)
- `celltype` (`category`)
- `organism` (`category`)
- `perturbation` (`category`, 237 levels including `control`)
- `nperts` (`int64`, values in `{0,1,2}`)
- `ngenes` (`int64`)
- `ncounts` (`float64`)
- `percent_mito` (`float64`)
- `percent_ribo` (`float64`)
- `condition` (`category`, 237 levels; same values as `perturbation`)
- `n_genes` (`int64`)

## `adata.var` (gene metadata)

`adata.var` has 8 columns:

- `ensemble_id` (`object`)
- `ncounts` (`float64`)
- `ncells` (`int64`)
- `n_cells` (`int64`)
- `highly_variable` (`bool`)
- `means` (`float64`)
- `dispersions` (`float64`)
- `dispersions_norm` (`float32`)

## Embeddings and graph slots

- `adata.obsm`
  - `X_pca`: `(111445, 50)`, `float32`
  - `X_umap`: `(111445, 2)`, `float32`
- `adata.varm`
  - `PCs`: `(8192, 50)`, `float64`
- `adata.obsp`
  - `connectivities`: sparse `(111445, 111445)`, `float32`
  - `distances`: sparse `(111445, 111445)`, `float32`

## `adata.uns`

Keys present:

- `deg_dict_vscontrol`: dict, 236 perturbations (non-control), each with `{"up": [...], "down": [...]}`
- `deg_dict_vsrest`: dict, 236 perturbations (non-control), each with `{"up": [...], "down": [...]}`
- `mean_dict`: dict over 237 conditions (including control); each entry is a per-gene mean list (length 22608)
- `disp_dict`: dict over 237 conditions (including control); each entry is a per-gene variance list (length 22608)
- `mean_disp_dict_genes`: gene list (length 22608) indexing `mean_dict`/`disp_dict` lists
- `hvg`: parameters from HVG selection
- `log1p`: log-transform metadata
- `neighbors`: neighbor graph metadata
- `pca`: PCA metadata
- `umap`: UMAP metadata

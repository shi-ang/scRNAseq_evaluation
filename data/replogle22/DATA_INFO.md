# Replogle22 `AnnData` Structure

This describes the processed dataset produced by [`get_data.py`](/home/shiang/projects/aip-rgreiner/shiang/diversity_by_design/data/replogle22/get_data.py) and saved at:

- `data/replogle22/replogle22_processed.h5ad`

When loaded with `anndata`/`scanpy`, the object has:

- `adata.shape = (39846, 7226)` (cells x genes)
- `adata.obs_names` unique cell barcodes (index name: `cell_barcode`)
- `adata.var_names` unique gene symbols (index name: `gene_name`)

## `adata.X` and `adata.layers`

- `adata.X`
  - Type: `scipy.sparse.csr_matrix`
  - Dtype: `float32`
  - Shape: `(39846, 7226)`
  - Contents: library-size normalized (`target_sum=1e4`) and `log1p`-transformed expression
- `adata.layers["counts"]`
  - Type: `scipy.sparse.csr_matrix`
  - Dtype: `float32`
  - Shape: `(39846, 7226)`
  - Contents: raw counts after cell/gene filtering, before normalization/log transform

## `adata.obs` (cell metadata)

`adata.obs` has 26 columns:

- `batch` (`int64`)
- `gene` (`category`)
- `gene_id` (`category`)
- `transcript` (`category`)
- `gene_transcript` (`category`)
- `guide_id` (`category`)
- `percent_mito` (`float64`)
- `UMI_count` (`float64`)
- `z_gemgroup_UMI` (`float64`)
- `core_scale_factor` (`float64`)
- `core_adjusted_UMI_count` (`float64`)
- `disease` (`category`)
- `cancer` (`bool`)
- `cell_line` (`category`, 2 levels: `K562`, `RPE1`)
- `sex` (`category`)
- `age` (`int64`)
- `perturbation` (`category`, 22 levels including `control`)
- `organism` (`category`)
- `perturbation_type` (`category`)
- `tissue_type` (`category`)
- `ncounts` (`float64`)
- `ngenes` (`int64`)
- `nperts` (`int64`, values in `{0,1}`)
- `percent_ribo` (`float64`)
- `condition` (`category`, 22 levels; same values as `perturbation`)
- `n_genes` (`int64`)

## `adata.var` (gene metadata)

`adata.var` has 5 columns:

- `n_cells` (`int64`)
- `highly_variable` (`bool`)
- `means` (`float64`)
- `dispersions` (`float64`)
- `dispersions_norm` (`float32`)

## Embeddings and graph slots

- `adata.obsm`
  - `X_pca`: `(39846, 50)`, `float32`
  - `X_umap`: `(39846, 2)`, `float32`
- `adata.varm`
  - `PCs`: `(7226, 50)`, `float64`
- `adata.obsp`
  - `connectivities`: sparse `(39846, 39846)`, `float32`
  - `distances`: sparse `(39846, 39846)`, `float32`

## `adata.uns`

Keys present:

- `deg_dict_vscontrol`: dict, 21 perturbations (non-control), each with `{"up": [...], "down": [...]}`
- `deg_dict_vsrest`: dict, 21 perturbations (non-control), each with `{"up": [...], "down": [...]}`
- `mean_dict`: dict over 22 conditions (including control); each entry is a per-gene mean list (length 7226)
- `disp_dict`: dict over 22 conditions (including control); each entry is a per-gene variance list (length 7226)
- `mean_disp_dict_genes`: gene list (length 7226) indexing `mean_dict`/`disp_dict` lists
- `hvg`: parameters from HVG selection
- `log1p`: log-transform metadata
- `neighbors`: neighbor graph metadata
- `pca`: PCA metadata
- `umap`: UMAP metadata

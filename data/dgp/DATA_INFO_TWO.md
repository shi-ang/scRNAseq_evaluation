# Synthetic Dataset Two (`synthetic_two.py`)

This describes the structure produced by [`synthetic_two.py`](/home/shiang/projects/aip-rgreiner/shiang/diversity_by_design/data/dgp/synthetic_two.py) (`synthetic_causalDGP`) when data are written as chunked `.h5ad` files and loaded with backed `AnnCollection`.

## Output format

`synthetic_causalDGP(...)` returns:

- `chunk_paths: list[str]`
- `affected_masks_pair: list[list[np.ndarray]]`, length 2:
  - `affected_masks_pair[0]`: affected-gene masks for `A`
  - `affected_masks_pair[1]`: affected-gene masks for `A_alter`
  - each inner list has length `P`, each mask is boolean with shape `(G,)`

Written chunk files follow:

- `.../synthetic_chunk_000000.h5ad`
- `.../synthetic_chunk_000001.h5ad`
- etc.

## Dataset size (parameterized)

Given generator inputs:

- genes: `G`
- control cells: `N0`
- perturbed cells per perturbation: `Nk`
- perturbations: `P`
- chunk size: `max_cells_per_chunk`

Resulting size:

- total cells: `N0 + P * Nk`
- total genes: `G`
- number of chunk files: `ceil((N0 + P * Nk) / max_cells_per_chunk)`

## Loading with backed `AnnCollection`

```python
import anndata as ad
from anndata.experimental import AnnCollection

chunk_paths, affected_masks_pair = synthetic_causalDGP(...)
backed_chunks = [ad.read_h5ad(path, backed="r") for path in chunk_paths]
ac = AnnCollection(
    backed_chunks,
    join_vars="inner",
    label="chunk_id",
    keys=[str(i) for i in range(len(backed_chunks))],
    index_unique="-",
)
```

Collection-level schema:

- `ac.shape = (N0 + P*Nk, G)`
- `ac.obs` columns: `["perturbation", "cell_line", "chunk_id"]`
- `ac.var_names`: `gene_0 ... gene_{G-1}`
- `ac.attrs_keys`:
  - `obs`: `["perturbation", "cell_line", "chunk_id"]`
  - `layers`: `["counts", "normalized_log1p"]` (layer key configurable)
  - `obsm`: `[]`

Notes:

- Control cells are labeled `perturbation = -1`.
- Perturbation groups are `0 ... P-1`.
- `cell_line` is synthetic binary context (`0` or `1`) sampled per cell.
- The returned affected-gene masks are computed from nonzero entries in `A^{-1} c_q` and `A_alter^{-1} c_q`.

## Per-chunk `AnnData` schema (`ad.read_h5ad(..., backed="r")`)

- `adata.X`
  - type (backed): `_CSRDataset`
  - dtype: `int32`
  - contents: raw counts
- `adata.layers["counts"]`
  - type: `scipy.sparse.csr_matrix`
  - dtype: `int32`
  - contents: raw counts copy
- `adata.layers["normalized_log1p"]` (default key)
  - type: `scipy.sparse.csr_matrix`
  - dtype: `float64`
  - contents: normalized (if `normalize=True`) and `log1p` transformed values

`adata.obs`:

- `perturbation` (`int32`)
- `cell_line` (`int32`)

`adata.var`:

- no columns
- index name: `gene`
- index values: `gene_0 ... gene_{G-1}`

Other slots:

- `adata.uns`: empty
- `adata.obsm`: empty
- `adata.obsp`: empty
- `adata.varm`: empty
- `adata.raw is None`

## Materializing from `AnnCollection`

For `sample = ac[idx].to_adata()`:

- `sample.X`: in-memory `csr_matrix` (`int32`)
- `sample.layers`: `["counts", "normalized_log1p"]`
- `sample.obs`: includes `perturbation`, `cell_line`, and `chunk_id`
- `sample.var`: shape `(G, 0)` (only index preserved)
- `sample.uns`: empty

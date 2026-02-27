from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
from anndata.experimental import AnnCollection


def load_manifest(processed_dir: Path) -> dict:
    manifest_path = processed_dir / "processed_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing manifest: {manifest_path}. Run process_data.py first."
        )
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_chunk_paths(manifest: dict) -> list[Path]:
    chunks = manifest.get("chunks", [])
    if not chunks:
        raise ValueError("Manifest has no chunk entries.")
    return [Path(entry["path"]) for entry in chunks]


def load_backed_collection(processed_dir: Path) -> tuple[AnnCollection, list[ad.AnnData], dict]:
    manifest = load_manifest(processed_dir)
    chunk_paths = get_chunk_paths(manifest)
    backed_chunks = [ad.read_h5ad(path, backed="r") for path in chunk_paths]
    collection = AnnCollection(
        backed_chunks,
        join_vars="inner",
        label="chunk_id",
        keys=[path.stem for path in chunk_paths],
        index_unique="-",
    )
    return collection, backed_chunks, manifest


def materialize_first_n_cells(collection: AnnCollection, n_cells: int) -> ad.AnnData:
    n_obs = int(collection.shape[0])
    n_take = min(n_cells, n_obs)
    if n_take <= 0:
        raise ValueError("No cells available to materialize.")
    return collection[np.arange(n_take, dtype=np.int64)].to_adata()


def main():
    parser = argparse.ArgumentParser(description="Load processed CD4+ chunks.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "processed",
        help="Directory containing processed_manifest.json and chunks/.",
    )
    parser.add_argument(
        "--sample-cells",
        type=int,
        default=10_000,
        help="Number of cells to materialize for demonstration.",
    )
    args = parser.parse_args()

    processed_dir = args.processed_dir.resolve()
    collection, backed_chunks, manifest = load_backed_collection(processed_dir)
    chunk_paths = get_chunk_paths(manifest)

    try:
        print(f"Processed dir: {processed_dir}")
        print(f"Chunks: {len(backed_chunks)}")
        print(f"Collection shape (cells x genes): {collection.shape}")
        print(f"Total cells (manifest): {manifest.get('n_obs_total')}")
        print(f"Final genes (manifest): {manifest.get('n_vars_final')}")
        print(f"Gene metadata CSV: {manifest.get('genes_csv')}")

        first_chunk = ad.read_h5ad(chunk_paths[0], backed="r")
        try:
            print(f"First chunk path: {chunk_paths[0]}")
            print(f"First chunk shape: {first_chunk.shape}")
            print("First chunk var columns:", list(first_chunk.var.columns))
        finally:
            if getattr(first_chunk, "file", None) is not None:
                first_chunk.file.close()

        if args.sample_cells > 0:
            adata_sample = materialize_first_n_cells(collection, args.sample_cells)
            print(f"Materialized sample shape: {adata_sample.shape}")
            print("Sample obs columns:", list(adata_sample.obs.columns))
            print("Sample var columns:", list(adata_sample.var.columns))
            if "condition" in adata_sample.obs.columns:
                condition_counts = adata_sample.obs["condition"].astype(str).value_counts().head(10)
                print("Top sample conditions:")
                print(condition_counts)
    finally:
        for chunk in backed_chunks:
            if getattr(chunk, "file", None) is not None:
                chunk.file.close()


if __name__ == "__main__":
    main()

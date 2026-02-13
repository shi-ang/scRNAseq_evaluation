"""
Model for perturbation analysis using scVI.

This module provides tools for performing perturbation analysis on single-cell
RNA-seq data using the scVI framework.
"""

from typing import Literal

import anndata
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import scvi
import torch
from scipy.sparse import csr_matrix
from scipy.stats import pearsonr  # type: ignore

torch.set_float32_matmul_precision("medium")
scvi.settings.seed = 42


class ScviPerturbationModel:
    """
    A model for perturbation analysis using scVI.

    Attributes:
    ----------
    adata : anndata.AnnData
        Annotated data matrix containing single-cell RNA-seq data.
    model : scvi.model.SCVI
        The scVI model instance.
    """

    def __init__(self, anndata: anndata.AnnData):
        """
        Initialize the scVI perturbation model with annotated data.

        Parameters
        ----------
        anndata : anndata.AnnData
            Annotated data matrix containing single-cell RNA-seq data.
        """
        self.adata = anndata
        self.model = None

    def run(
        self,
        n_latent: int = 10,
        n_hidden: int = 128,
        n_layers: int = 1,
        gene_likelihood: Literal["zinb", "nb", "poisson", "normal"] = "zinb",
        max_epochs: int = 100,
        batch_size: int = 128,
        n_samples: int = 10,
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        early_stopping: bool = False,
    ) -> anndata.AnnData:
        """
        Setup, train the scVI model, and store results in the AnnData object.

        Parameters
        ----------
        n_latent : int, optional
            Dimensionality of the latent space, by default 10.
        n_hidden : int, optional
            Number of nodes per hidden layer, by default 128.
        n_layers : int, optional
            Number of hidden layers, by default 1.
        gene_likelihood : Literal["zinb", "nb", "poisson", "normal"], optional
            Distribution of the likelihood, by default "zinb".
        max_epochs : int, optional
            Maximum number of training epochs, by default 100.
        batch_size : int, optional
            Batch size for training, by default 128.
        n_samples : int, optional
            Number of posterior samples, by default 10.
        dispersion : Literal['gene', 'gene-batch', 'gene-label', 'gene-cell'], optional
            Dispersion parameterization, by default "gene".
        early_stopping : bool, optional
            Whether to use early stopping during training, by default False. Note this will split the data into training and validation sets, which may not be desirable for small datasets.
        """
        # Setup anndata with perturbation as categorical covariate
        scvi.model.SCVI.setup_anndata(  # type: ignore
            self.adata,
            layer="counts",
            categorical_covariate_keys=["perturbation"],
        )

        # Initialize model
        self.model = scvi.model.SCVI(
            self.adata,
            n_latent=n_latent,
            n_hidden=n_hidden,
            n_layers=n_layers,
            gene_likelihood=gene_likelihood,
            dispersion=dispersion,
        )

        # Train
        print("Training scVI model...")
        self.model.train(  # type: ignore
            max_epochs=max_epochs, batch_size=batch_size, early_stopping=early_stopping
        )

        print("Training complete. Storing results...")
        # Store results
        self.adata.obsm["Z_scvi"] = self.model.get_latent_representation()  # type: ignore

        self.adata.layers["scvi_normalized"] = self.model.get_normalized_expression()  # type: ignore

        posterior_samples = self.model.posterior_predictive_sample(n_samples=n_samples)  # type: ignore
        print(f"Posterior samples shape: {posterior_samples}")
        median_posterior_samples = np.rint(np.median(posterior_samples.todense(), axis=2))  # type: ignore
        print(f"Median posterior samples shape: {median_posterior_samples.shape}")
        self.adata.layers["scvi_posterior_samples"] = median_posterior_samples

        return self.adata


if __name__ == "__main__":
    # create synthetic data
    is_synthetic = False
    adata = anndata.AnnData()
    if is_synthetic:
        np.random.seed(42)  # noqa
        n_cells = 500
        n_genes = 200
        n_perturbations = 5

        # Simulate count data
        counts = np.random.negative_binomial(5, 0.3, size=(n_cells, n_genes))  # noqa
        adata = anndata.AnnData(
            X=csr_matrix(counts.astype(np.float32)),
            obs={
                "perturbation": np.random.choice(  # noqa
                    [f"pert_{i}" for i in range(n_perturbations)] + ["control"],
                    size=n_cells,
                )
            },
        )
        adata.layers["counts"] = adata.X.copy()  # type: ignore
    else:
        # Load Norman19 data
        adata = sc.read_h5ad(
            "data/norman19/norman19_processed.h5ad"
        )
        # print("Original Norman adata shape:", adata.shape)
        # indices = np.random.choice(adata.n_obs, size=1000, replace=False)  # noqa
        # adata = adata[indices].copy()

    print("adata shape:", adata.shape)
    print("adata obs", adata.obs.keys())
    print("adata layers", adata.layers.keys())
    print("adata obsm", adata.obsm.keys())
    print("adata var", adata.var.keys())
    print("Cell Types:", adata.obs["cell_type"].unique())
    print(f"Perturbations: {adata.obs['perturbation'].nunique()}")
    print(f"Perturbation counts:\n{adata.obs['perturbation'].value_counts().head(10)}")

    # --- Train model ---
    model = ScviPerturbationModel(adata)
    model.run(
        n_latent=10,
        n_hidden=128,
        n_layers=1,
        gene_likelihood="zinb",
        max_epochs=5,
        batch_size=256,
        early_stopping=False,
    )
    print(model.model.summary_string)  # type: ignore
    print(model.model.history.keys())  # type: ignore
    print(model.model.history["train_loss"].tail(10))  # type: ignore

    # Check scales match
    print("Counts range:", adata.layers["counts"].min(), adata.layers["counts"].max())
    print(
        "Posterior range:",
        adata.layers["scvi_posterior_samples"].min(),
        adata.layers["scvi_posterior_samples"].max(),
    )
    print("adata.X range:", adata.X.min(), adata.X.max())  # type: ignore # if much smaller, it's log-normalized
    print(
        "Normalized expression range:",
        adata.layers["scvi_normalized"].min(),
        adata.layers["scvi_normalized"].max(),
    )
    # Mean expression per gene
    original = np.array(adata.layers["counts"].todense())  # type: ignore
    reconstructed = adata.layers["scvi_posterior_samples"]
    orig_mean = original.mean(axis=0)
    recon_mean = reconstructed.mean(axis=0)

    r, _ = pearsonr(
        orig_mean.A1 if hasattr(orig_mean, "A1") else orig_mean,
        recon_mean.A1 if hasattr(recon_mean, "A1") else recon_mean,
    )
    print(f"Pearson correlation between original and reconstructed mean expression: {r:.3f}")

    plt.scatter(orig_mean, recon_mean, alpha=0.3, s=5)  # type: ignore
    plt.xlabel("Original mean expression")  # type: ignore
    plt.ylabel("Reconstructed mean expression")  # type: ignore
    plt.title(f"Per-gene mean expression (Pearson r={r:.3f})")  # type: ignore
    plt.plot([0, orig_mean.max()], [0, orig_mean.max()], "r--")  # type: ignore
    # Diagonal line matching axis limits)
    model_dir = "models/"
    plt.savefig(model_dir + "SCVI_gene_correlation.png")  #  type: ignore
    # plt.show()

    print(f"Latent shape: {adata.obsm['Z_scvi'].shape}")
    print(f"Normalized expression shape: {adata.layers['scvi_normalized'].shape}")
    print(f"Posterior samples shape: {adata.layers['scvi_posterior_samples'].shape}")
    print("Done!")

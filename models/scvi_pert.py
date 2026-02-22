from typing import Optional, Literal

import anndata
import numpy as np
import scvi
import torch
from scvi.dataloaders import CollectionAdapter

torch.set_float32_matmul_precision("medium")


class ScviPerturbation:
    """Train scVI on train data and infer on test data."""

    def __init__(
        self,
        data: anndata.AnnData | anndata.experimental.AnnCollection,
        counts_layer: Optional[str] = None,
        perturbation_key: str = "perturbation",
        cell_type_key: str = "cell_type",
        batch_key: Optional[str] = None,
        train_idx: Optional[np.ndarray] = None,
        val_idx: Optional[np.ndarray] = None,
        test_idx: Optional[np.ndarray] = None,
        seed: int = 42,
    ) -> None:
        self.data = self._adapt_data(data)
        self.counts_layer = counts_layer
        self.perturbation_key = perturbation_key
        self.cell_type_key = cell_type_key
        self.batch_key = batch_key
        self.seed = seed

        self.model: Optional[scvi.model.SCVI] = None
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx

    @staticmethod
    def _adapt_data(
        data: anndata.AnnData | anndata.experimental.AnnCollection | CollectionAdapter,
    ) -> anndata.AnnData | CollectionAdapter:
        if isinstance(data, (anndata.AnnData, CollectionAdapter)):
            return data
        else:
            return CollectionAdapter(data)

    def run(
        self,
        n_latent: int = 10,
        n_hidden: int = 128,
        n_layers: int = 1,
        gene_likelihood: str = "zinb",
        dispersion: str = "gene",
        max_epochs: int = 50,
        batch_size: int = 256,
        early_stopping: bool = False,
        dataloader_num_workers: int = 0,
        dataloader_persistent_workers: Optional[bool] = None,
    ) -> anndata.AnnData:
        setup_kwargs: dict[str, object] = {
            "categorical_covariate_keys": [self.perturbation_key],
        }
        if self.counts_layer is not None:
            setup_kwargs["layer"] = self.counts_layer
        if self.batch_key is not None:
            setup_kwargs["batch_key"] = self.batch_key

        scvi.model.SCVI.setup_anndata(self.data, **setup_kwargs)
        self.model = scvi.model.SCVI(
            self.data,
            n_latent=n_latent,
            n_hidden=n_hidden,
            n_layers=n_layers,
            gene_likelihood=gene_likelihood,
            dispersion=dispersion,
        )
        if dataloader_num_workers < 0:
            raise ValueError("dataloader_num_workers must be >= 0.")
        if dataloader_persistent_workers is None:
            dataloader_persistent_workers = dataloader_num_workers > 0

        datasplitter_kwargs: dict[str, object] = {
            "external_indexing": [self.train_idx, self.val_idx, self.test_idx],
            # Keep this at 0 by default to avoid nested multiprocessing semaphore
            # issues when the outer sweep already runs in many worker processes.
            "num_workers": int(dataloader_num_workers),
        }
        if dataloader_num_workers > 0:
            datasplitter_kwargs["persistent_workers"] = bool(dataloader_persistent_workers)

        self.model.train(
            max_epochs=max_epochs,
            batch_size=batch_size,
            early_stopping=early_stopping,
            datasplitter_kwargs=datasplitter_kwargs,
        )
        
        # store results in an annData object
        z = self.model.get_latent_representation(indices=self.test_idx)
        scvi_normalized = self.model.get_normalized_expression(indices=self.test_idx)
        if isinstance(self.data, CollectionAdapter):
            # IMPORTANT: AnnCollectionView.to_adata() gives X/layers; AnnCollection.to_adata() does not. :contentReference[oaicite:1]{index=1}
            test_view = self.data.collection[self.test_idx, :]   # AnnCollectionView
            adata_out = test_view.to_adata()
        else:
            # regular AnnData case
            adata_out = self.data[self.test_idx].copy()
        
        adata_out.obsm["X_scvi"] = z
        adata_out.layers["scvi_normalized"] = scvi_normalized.astype("float32", copy=False)
        return adata_out


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

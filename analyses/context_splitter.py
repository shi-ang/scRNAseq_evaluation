from __future__ import annotations

from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd


class ContextSplitter:
    """Seeded train/val/test splitter for AnnData/AnnCollection-like objects."""

    def __init__(
        self,
        adata: Any,
        train_frac: float,
        val_frac: float,
        test_frac: float,
        context_mode: Literal["cell", "donor"] | None = "cell",
        perturbation_key: str = "perturbation",
        cell_line_key: str = "cell_line",
        donor_key: str = "donor",
        holdout_context_values: Sequence[Any] | None = None,
        control_label: Any | Sequence[Any] | None = None,
    ) -> None:
        self.adata = adata
        self.perturbation_key = str(perturbation_key)
        self.cell_line_key = str(cell_line_key)
        self.donor_key = str(donor_key)
        self.context_mode = context_mode

        self.obs = self._extract_obs_df(adata)
        self.n_obs = self._extract_n_obs(adata, self.obs)

        self._fractions = self._normalize_fractions(train_frac, val_frac, test_frac)
        self._validate_perturbation_column()
        self._available_context_key = self._infer_available_context_key()

        # Convert holdout_context_values to a tuple of unique values for consistent handling
        self.holdout_context_values = tuple(
            dict.fromkeys(list(holdout_context_values or []))
        )
        self._validate_context_mode_and_holdouts()

        self.control_label = control_label

    def split(self, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(int(seed))
        if self.context_mode is None:
            return self._split_in_context(rng)
        return self._split_cross_context(rng)

    @staticmethod
    def _extract_obs_df(adata: Any) -> pd.DataFrame:
        obs = getattr(adata, "obs", None)
        if obs is None:
            raise AttributeError("Input object must expose `.obs`.")
        if not isinstance(obs, pd.DataFrame):
            raise TypeError(f"`obj.obs` must be a pandas DataFrame, got {type(obs)!r}.")
        return obs

    @staticmethod
    def _extract_n_obs(adata: Any, obs: pd.DataFrame) -> int:
        n_obs_from_obs = int(len(obs))
        if hasattr(adata, "n_obs"):
            n_obs = int(getattr(adata, "n_obs"))
        elif hasattr(adata, "shape") and len(getattr(adata, "shape")) >= 1:
            n_obs = int(getattr(adata, "shape")[0])
        else:
            n_obs = n_obs_from_obs

        if n_obs != n_obs_from_obs:
            raise ValueError(
                f"Inconsistent observation count: obj.n_obs={n_obs}, len(obj.obs)={n_obs_from_obs}."
            )
        if n_obs < 0:
            raise ValueError("Number of observations must be non-negative.")
        return n_obs

    @staticmethod
    def _normalize_fractions(
        train_frac: float,
        val_frac: float,
        test_frac: float,
    ) -> np.ndarray:
        fractions = np.asarray([train_frac, val_frac, test_frac], dtype=np.float64)
        if np.any(~np.isfinite(fractions)):
            raise ValueError("train_frac, val_frac, and test_frac must be finite numbers.")
        if np.any(fractions < 0):
            raise ValueError("train_frac, val_frac, and test_frac must be non-negative.")

        total = float(fractions.sum())
        if total <= 0:
            raise ValueError("At least one split fraction must be > 0.")

        return fractions / total

    def _validate_perturbation_column(self) -> None:
        if self.perturbation_key not in self.obs.columns:
            raise KeyError(
                f"Missing perturbation column '{self.perturbation_key}' in obj.obs. "
                f"Available columns: {list(self.obs.columns)}"
            )

    def _infer_available_context_key(self) -> str | None:
        has_cell = self.cell_line_key in self.obs.columns
        has_donor = self.donor_key in self.obs.columns
        if has_cell and has_donor:
            raise ValueError(
                "Dataset must contain either a cell-line context column or a donor context column, not both."
            )
        if has_cell:
            return self.cell_line_key
        if has_donor:
            return self.donor_key
        return None

    def _validate_context_mode_and_holdouts(self) -> None:
        if self.context_mode not in ("cell", "donor", None):
            raise ValueError(
                f"context_mode must be one of 'cell', 'donor', or None. Got {self.context_mode!r}."
            )

        if self.context_mode is None:
            return

        context_key = self._context_key_for_mode()
        if context_key not in self.obs.columns:
            raise KeyError(
                f"Cross-context split requires context column '{context_key}', but it was not found in obj.obs."
            )

        if len(self.holdout_context_values) == 0:
            raise ValueError(
                "Cross-context split requires non-empty holdout_context_values."
            )

        context_values = set(pd.unique(self.obs[context_key]))
        missing = [value for value in self.holdout_context_values if value not in context_values]
        if missing:
            raise ValueError(
                f"holdout_context_values contains unknown value(s) for '{context_key}': {missing}. "
                f"Available values include: {sorted(context_values, key=lambda x: str(x))}"
            )

    @staticmethod
    def _rounded_partition_counts(n_total: int, fractions: np.ndarray) -> tuple[int, int, int]:
        if n_total < 0:
            raise ValueError("n_total must be non-negative.")
        if n_total == 0:
            return 0, 0, 0

        targets = fractions * float(n_total)
        counts = np.rint(targets).astype(np.int64)
        delta = int(n_total - counts.sum())

        if delta > 0:
            priority = np.argsort(-(targets - counts), kind="stable")
            for i in range(delta):
                counts[priority[i % 3]] += 1
        elif delta < 0:
            priority = np.argsort(-(counts - targets), kind="stable")
            to_remove = -delta
            ptr = 0
            while to_remove > 0:
                idx = int(priority[ptr % 3])
                if counts[idx] > 0:
                    counts[idx] -= 1
                    to_remove -= 1
                ptr += 1

        if int(counts.sum()) != n_total or np.any(counts < 0):
            raise RuntimeError(
                f"Failed to partition n_total={n_total} with fractions={fractions.tolist()}."
            )

        return int(counts[0]), int(counts[1]), int(counts[2])

    def _context_key_for_mode(self) -> str:
        if self.context_mode == "cell":
            return self.cell_line_key
        if self.context_mode == "donor":
            return self.donor_key
        raise RuntimeError("Context key requested for in-context mode.")

    def _split_in_context(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        group_cols = [self.perturbation_key]
        if self._available_context_key is not None:
            group_cols.append(self._available_context_key)

        if len(group_cols) == 1:
            grouped = self.obs.groupby(group_cols[0], sort=True, observed=False).indices
        else:
            grouped = self.obs.groupby(group_cols, sort=True, observed=False).indices

        train_idx: list[int] = []
        val_idx: list[int] = []
        test_idx: list[int] = []

        for group_key in sorted(grouped.keys(), key=lambda x: repr(x)):
            idx = np.asarray(grouped[group_key], dtype=np.int64)
            if idx.size == 0:
                continue

            shuffled = rng.permutation(idx)
            n_train, n_val, n_test = self._rounded_partition_counts(
                int(shuffled.size), self._fractions
            )
            cutoff_1 = n_train
            cutoff_2 = n_train + n_val

            train_idx.extend(shuffled[:cutoff_1].tolist())
            val_idx.extend(shuffled[cutoff_1:cutoff_2].tolist())
            test_idx.extend(shuffled[cutoff_2 : cutoff_2 + n_test].tolist())

        train_arr = np.asarray(train_idx, dtype=np.int64)
        val_arr = np.asarray(val_idx, dtype=np.int64)
        test_arr = np.asarray(test_idx, dtype=np.int64)

        self._assert_complete_partition(train_arr, val_arr, test_arr)
        return train_arr, val_arr, test_arr

    def _split_cross_context(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        context_key = self._context_key_for_mode()
        perturbation_values = self.obs[self.perturbation_key].to_numpy(copy=False)
        perturbations = np.sort(pd.unique(perturbation_values))

        control_values = self._fetch_controls(perturbations)
        perturbation_is_control = (
            np.isin(perturbation_values, control_values)
            if control_values.size > 0
            else np.zeros(self.n_obs, dtype=bool)
        )
        non_control_perturbations = (
            perturbations[~np.isin(perturbations, control_values)]
            if control_values.size > 0
            else perturbations
        )

        n_train, n_val, _ = self._rounded_partition_counts(
            int(non_control_perturbations.size), self._fractions
        )
        permutation = rng.permutation(non_control_perturbations)

        train_perts = permutation[:n_train]
        val_perts = permutation[n_train : n_train + n_val]
        test_perts = permutation[n_train + n_val :]

        context_values = self.obs[context_key].to_numpy(copy=False)

        holdout_mask = np.isin(context_values, self.holdout_context_values)
        train_mask = (
            perturbation_is_control
            | (~holdout_mask)
            | (holdout_mask & np.isin(perturbation_values, train_perts))
        )
        val_mask = (
            (~perturbation_is_control)
            & holdout_mask
            & np.isin(perturbation_values, val_perts)
        )
        test_mask = (
            (~perturbation_is_control)
            & holdout_mask
            & np.isin(perturbation_values, test_perts)
        )

        train_idx = np.flatnonzero(train_mask).astype(np.int64, copy=False)
        val_idx = np.flatnonzero(val_mask).astype(np.int64, copy=False)
        test_idx = np.flatnonzero(test_mask).astype(np.int64, copy=False)

        self._assert_complete_partition(train_idx, val_idx, test_idx)
        return train_idx, val_idx, test_idx

    def _fetch_controls(self, perturbations: np.ndarray) -> np.ndarray:
        if self.control_label is not None:
            if isinstance(self.control_label, (str, bytes)):
                requested = np.asarray([self.control_label], dtype=object)
            else:
                try:
                    requested = np.asarray(list(self.control_label), dtype=object)
                except TypeError:
                    requested = np.asarray([self.control_label], dtype=object)

            matched = perturbations[np.isin(perturbations, requested)]
            if matched.size == 0:
                raise ValueError(
                    f"Requested control_label={self.control_label!r} was not found in "
                    f"obs['{self.perturbation_key}']."
                )
            return matched

        inferred_controls: list[Any] = []
        for value in perturbations:
            if isinstance(value, (np.integer, int, np.floating, float)) and float(value) == -1.0:
                inferred_controls.append(value)
                continue
            if isinstance(value, str) and value.strip().lower() in {"control", "ctrl"}:
                inferred_controls.append(value)

        if not inferred_controls:
            return np.asarray([], dtype=object)
        return np.asarray(list(dict.fromkeys(inferred_controls)), dtype=object)

    def _assert_complete_partition(
        self, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray
    ) -> None:
        assignment = np.zeros(self.n_obs, dtype=np.int8)
        for split_name, split_idx in (
            ("train", train_idx),
            ("val", val_idx),
            ("test", test_idx),
        ):
            if split_idx.size == 0:
                continue
            idx = np.asarray(split_idx, dtype=np.int64)
            if np.any((idx < 0) | (idx >= self.n_obs)):
                raise RuntimeError(f"{split_name} split contains out-of-range indices.")
            np.add.at(assignment, idx, 1)

        if np.any(assignment != 1):
            n_unassigned = int(np.sum(assignment == 0))
            n_overlapping = int(np.sum(assignment > 1))
            raise RuntimeError(
                "Split indices must be disjoint and cover all rows. "
                f"Unassigned rows: {n_unassigned}; overlapping rows: {n_overlapping}."
            )

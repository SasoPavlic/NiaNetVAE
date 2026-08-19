"""Shared training, fine-tuning, checkpointing, and scoring for recurrent models."""

from __future__ import annotations

import copy
import os
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from ..config import TrainingConfig
from ..contracts import ArchitectureSpec
from ..dataloaders.sequences import SegmentedSequenceDataset
from ..models.recurrent import (
    RecurrentAutoencoder,
    build_recurrent_model,
    recurrent_anomaly_score,
    recurrent_training_loss,
)


@dataclass(frozen=True)
class FitResult:
    completed_epochs: int
    best_epoch: int | None
    best_validation_loss: float | None
    restored_best_weights: bool
    training_windows: int
    validation_windows: int
    history: tuple[dict[str, float], ...]


def set_deterministic_seed(seed: int, deterministic: bool = True) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False


def _resolve_device(configured: str) -> torch.device:
    value = str(configured).strip().lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("training.device='cuda' but CUDA is unavailable.")
    return torch.device(value)


def _loader(
    dataset: Dataset,
    config: TrainingConfig,
    *,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    persistent = bool(config.persistent_workers and config.num_workers > 0)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=persistent,
        drop_last=config.drop_last if shuffle else False,
        generator=generator,
    )


def _even_indices(total: int, count: int) -> list[int]:
    if count <= 0 or total <= 0:
        return []
    if count >= total:
        return list(range(total))
    # Midpoint sampling avoids systematically over-representing the first and
    # final baseline windows while remaining deterministic.
    return np.floor((np.arange(count, dtype=np.float64) + 0.5) * total / count).astype(int).tolist()


class RecurrentRuntime:
    def __init__(self, architecture: ArchitectureSpec, config: TrainingConfig) -> None:
        self.architecture = architecture.validate()
        self.config = config
        set_deterministic_seed(config.seed, config.deterministic)
        self.model: RecurrentAutoencoder = build_recurrent_model(architecture)
        self.device = _resolve_device(config.device)
        self.model.to(self.device)
        self.last_fit_result: FitResult | None = None

    def _loss(self, batch: torch.Tensor) -> torch.Tensor:
        return recurrent_training_loss(
            self.model,
            batch,
            kl_beta=self.config.kl_beta,
            sparsity_beta=self.config.sae_sparsity_beta,
            sparsity_rho=self.config.sae_sparsity_rho,
        )

    def fit(
        self,
        train_segments: Sequence[pd.DataFrame],
        validation_segments: Sequence[pd.DataFrame] = (),
        *,
        learning_rate: float | None = None,
        min_epochs: int | None = None,
        max_epochs: int | None = None,
        early_stopping: bool = True,
        train_dataset_override: Dataset | None = None,
    ) -> FitResult:
        set_deterministic_seed(self.config.seed, self.config.deterministic)
        train_dataset = train_dataset_override or SegmentedSequenceDataset(
            train_segments,
            sequence_length=self.architecture.sequence_length,
            stride=1,
        )
        validation_dataset = SegmentedSequenceDataset(
            validation_segments,
            sequence_length=self.architecture.sequence_length,
            stride=1,
        )
        if len(train_dataset) < 1:
            raise ValueError("Recurrent training produced zero sequence windows.")
        train_loader = _loader(
            train_dataset,
            self.config,
            shuffle=self.config.shuffle,
            seed=self.config.seed,
        )
        validation_loader = (
            _loader(validation_dataset, self.config, shuffle=False, seed=self.config.seed)
            if len(validation_dataset)
            else None
        )
        minimum = int(min_epochs if min_epochs is not None else self.config.min_epochs)
        maximum = int(max_epochs if max_epochs is not None else self.config.max_epochs)
        if minimum < 1 or maximum < minimum:
            raise ValueError("Invalid recurrent training epoch bounds.")
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(learning_rate if learning_rate is not None else self.config.learning_rate),
            weight_decay=float(self.config.weight_decay),
        )
        best_state: dict | None = None
        best_loss: float | None = None
        best_epoch: int | None = None
        epochs_without_improvement = 0
        history: list[dict[str, float]] = []
        completed = 0
        for epoch in range(1, maximum + 1):
            self.model.train()
            train_losses: list[float] = []
            for batch, _segment, _anchor in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = self._loss(batch)
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.detach().cpu()))
            train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
            validation_loss = float("nan")
            if validation_loader is not None:
                self.model.eval()
                losses: list[float] = []
                with torch.no_grad():
                    for batch, _segment, _anchor in validation_loader:
                        losses.append(float(self._loss(batch.to(self.device)).detach().cpu()))
                validation_loss = float(np.mean(losses)) if losses else float("nan")
            history.append(
                {
                    "epoch": float(epoch),
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                }
            )
            completed = epoch
            if validation_loader is not None and np.isfinite(validation_loss):
                improved = best_loss is None or validation_loss < best_loss - self.config.min_delta
                if improved:
                    best_loss = validation_loss
                    best_epoch = epoch
                    best_state = copy.deepcopy(self.model.state_dict())
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                if (
                    early_stopping
                    and epoch >= minimum
                    and epochs_without_improvement >= self.config.patience
                ):
                    break
        restored = False
        if best_state is not None and self.config.restore_best_weights:
            self.model.load_state_dict(best_state)
            restored = True
        result = FitResult(
            completed_epochs=completed,
            best_epoch=best_epoch,
            best_validation_loss=best_loss,
            restored_best_weights=restored,
            training_windows=len(train_dataset),
            validation_windows=len(validation_dataset),
            history=tuple(history),
        )
        self.last_fit_result = result
        return result

    def fine_tune(
        self,
        *,
        baseline_segments: Sequence[pd.DataFrame],
        local_train_segments: Sequence[pd.DataFrame],
        local_validation_segments: Sequence[pd.DataFrame],
        baseline_replay_fraction: float,
        learning_rate_scale: float,
        min_epochs: int,
        max_epochs: int,
        early_stopping: bool,
    ) -> FitResult:
        baseline_dataset = SegmentedSequenceDataset(
            baseline_segments,
            sequence_length=self.architecture.sequence_length,
            stride=1,
        )
        local_dataset = SegmentedSequenceDataset(
            local_train_segments,
            sequence_length=self.architecture.sequence_length,
            stride=1,
        )
        if len(local_dataset) < 1:
            raise ValueError("Fine-tuning requires at least one local sequence.")
        requested_baseline = (
            int(
                round(
                    len(local_dataset) * baseline_replay_fraction / (1.0 - baseline_replay_fraction)
                )
            )
            if baseline_replay_fraction > 0.0
            else 0
        )
        baseline_indices = _even_indices(
            len(baseline_dataset), min(len(baseline_dataset), requested_baseline)
        )
        parts: list[Dataset] = []
        if baseline_indices:
            parts.append(Subset(baseline_dataset, baseline_indices))
        parts.append(local_dataset)
        training_dataset: Dataset = parts[0] if len(parts) == 1 else ConcatDataset(parts)
        return self.fit(
            train_segments=(),
            validation_segments=local_validation_segments,
            learning_rate=self.config.learning_rate * float(learning_rate_scale),
            min_epochs=min_epochs,
            max_epochs=max_epochs,
            early_stopping=early_stopping,
            train_dataset_override=training_dataset,
        )

    def score_segments(self, segments: Sequence[pd.DataFrame]) -> pd.Series:
        dataset = SegmentedSequenceDataset(
            segments,
            sequence_length=self.architecture.sequence_length,
            stride=1,
        )
        if len(dataset) < 1:
            return pd.Series(dtype=float, name="anomaly_score")
        loader = _loader(dataset, self.config, shuffle=False, seed=self.config.seed)
        timestamps: list[pd.Timestamp] = []
        scores: list[float] = []
        self.model.eval()
        with torch.no_grad():
            for batch, segment_indices, anchor_offsets in loader:
                batch_scores = (
                    recurrent_anomaly_score(self.model, batch.to(self.device))
                    .detach()
                    .cpu()
                    .numpy()
                )
                for segment_index, anchor_offset, score in zip(
                    segment_indices.tolist(),
                    anchor_offsets.tolist(),
                    batch_scores.tolist(),
                    strict=True,
                ):
                    timestamps.append(
                        pd.Timestamp(dataset.anchor_index(segment_index, anchor_offset))
                    )
                    scores.append(float(score))
        return pd.Series(
            scores, index=pd.DatetimeIndex(timestamps), name="anomaly_score"
        ).sort_index()

    def reconstruction_smape(self, segments: Sequence[pd.DataFrame]) -> float:
        """Return the shared, element-wise symmetric MAPE on sequence windows."""
        dataset = SegmentedSequenceDataset(
            segments,
            sequence_length=self.architecture.sequence_length,
            stride=1,
        )
        if len(dataset) < 1:
            raise ValueError("SMAPE evaluation produced zero sequence windows.")
        loader = _loader(dataset, self.config, shuffle=False, seed=self.config.seed)
        total = 0.0
        count = 0
        epsilon = 1e-8
        self.model.eval()
        with torch.no_grad():
            for batch, _segment, _anchor in loader:
                batch = batch.to(self.device)
                reconstructed = self.model.reconstruct_deterministic(batch)
                values = (
                    2.0
                    * torch.abs(reconstructed - batch)
                    / (torch.abs(reconstructed) + torch.abs(batch) + epsilon)
                )
                total += float(values.sum().detach().cpu())
                count += int(values.numel())
        if count < 1:
            raise ValueError("SMAPE evaluation produced no finite elements.")
        result = total / count
        if not np.isfinite(result):
            raise ValueError("SMAPE evaluation returned a non-finite value.")
        return float(result)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "architecture": self.architecture.as_dict(),
                "architecture_hash": self.architecture.architecture_hash,
                "training_config": asdict(self.config),
                "model_state_dict": self.model.state_dict(),
                "fit_result": asdict(self.last_fit_result) if self.last_fit_result else None,
            },
            target,
        )
        return target

    @classmethod
    def load(
        cls,
        path: str | Path,
        architecture: ArchitectureSpec,
        config: TrainingConfig,
    ) -> RecurrentRuntime:
        payload = torch.load(path, map_location="cpu")
        if payload.get("architecture_hash") != architecture.architecture_hash:
            raise ValueError("Checkpoint architecture hash does not match ArchitectureSpec.")
        runtime = cls(architecture, config)
        runtime.model.load_state_dict(payload["model_state_dict"])
        return runtime

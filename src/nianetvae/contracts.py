"""Stable domain contracts shared by search, training, and evaluation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Literal

ModelKind = Literal["iforest", "recurrent_sae", "recurrent_vae"]
StrategyKind = Literal["static", "per_maintenance"]


@dataclass(frozen=True)
class ArchitectureSpec:
    model_kind: ModelKind
    source: Literal["handcrafted", "nsga3", "fixed"]
    input_dim: int
    sequence_length: int
    recurrent_cell: Literal["LSTM", "GRU", "RNN_TANH"] | None = None
    encoder_hidden_dims: tuple[int, ...] = ()
    decoder_hidden_dims: tuple[int, ...] = ()
    latent_dim: int | None = None
    activation: str = "identity"
    genome: tuple[float, ...] | None = None
    mapping_version: str = "architecture_spec_v1"
    label: str = ""

    def validate(self) -> ArchitectureSpec:
        if self.input_dim < 1:
            raise ValueError("Architecture input_dim must be positive.")
        if self.model_kind == "iforest":
            if self.encoder_hidden_dims or self.decoder_hidden_dims or self.latent_dim is not None:
                raise ValueError("IForest ArchitectureSpec cannot contain recurrent dimensions.")
            return self
        if self.sequence_length < 2:
            raise ValueError("Recurrent architecture sequence_length must be >=2.")
        if self.recurrent_cell not in {"LSTM", "GRU", "RNN_TANH"}:
            raise ValueError(f"Unsupported recurrent_cell={self.recurrent_cell!r}.")
        if not self.encoder_hidden_dims or not self.decoder_hidden_dims:
            raise ValueError("Recurrent architecture requires encoder and decoder dimensions.")
        if any(int(value) < 1 for value in (*self.encoder_hidden_dims, *self.decoder_hidden_dims)):
            raise ValueError("Every recurrent hidden dimension must be positive.")
        if self.latent_dim is None or int(self.latent_dim) < 1:
            raise ValueError("Recurrent architecture requires a positive latent_dim.")
        if self.genome is not None and len(self.genome) != 6:
            raise ValueError("NSGA-III architecture genomes must contain exactly six genes.")
        return self

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ArchitectureSpec:
        values = dict(payload)
        for key in ("encoder_hidden_dims", "decoder_hidden_dims", "genome"):
            if values.get(key) is not None:
                values[key] = tuple(values[key])
        return cls(**values).validate()

    @property
    def architecture_hash(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class WorkflowSpec:
    workflow_id: str
    model_kind: ModelKind
    strategy: StrategyKind
    architecture_source: Literal["handcrafted", "nsga3", "fixed"]

    @classmethod
    def from_id(cls, workflow_id: str) -> WorkflowSpec:
        mapping = {
            "iforest_static": cls("iforest_static", "iforest", "static", "fixed"),
            "iforest_per_maintenance": cls(
                "iforest_per_maintenance", "iforest", "per_maintenance", "fixed"
            ),
            "sae_static": cls("sae_static", "recurrent_sae", "static", "handcrafted"),
            "vae_static": cls("vae_static", "recurrent_vae", "static", "handcrafted"),
            "nianetvae_per_maintenance": cls(
                "nianetvae_per_maintenance", "recurrent_vae", "per_maintenance", "nsga3"
            ),
        }
        try:
            return mapping[str(workflow_id)]
        except KeyError as exc:
            raise ValueError(f"Unsupported workflow_id={workflow_id!r}.") from exc


def handcrafted_vae_spec(input_dim: int, sequence_length: int) -> ArchitectureSpec:
    return ArchitectureSpec(
        model_kind="recurrent_vae",
        source="handcrafted",
        input_dim=int(input_dim),
        sequence_length=int(sequence_length),
        recurrent_cell="LSTM",
        encoder_hidden_dims=(64, 64),
        decoder_hidden_dims=(64, 64),
        latent_dim=32,
        activation="identity",
        label="handcrafted_lstm_2x64_latent32",
    ).validate()


def handcrafted_sae_spec(input_dim: int, sequence_length: int) -> ArchitectureSpec:
    return ArchitectureSpec(
        model_kind="recurrent_sae",
        source="handcrafted",
        input_dim=int(input_dim),
        sequence_length=int(sequence_length),
        recurrent_cell="LSTM",
        encoder_hidden_dims=(64, 64),
        decoder_hidden_dims=(64, 64),
        latent_dim=32,
        activation="identity",
        label="handcrafted_sparse_lstm_2x64_latent32",
    ).validate()


def iforest_spec(input_dim: int) -> ArchitectureSpec:
    return ArchitectureSpec(
        model_kind="iforest",
        source="fixed",
        input_dim=int(input_dim),
        sequence_length=1,
        label="iforest_100_auto",
    ).validate()

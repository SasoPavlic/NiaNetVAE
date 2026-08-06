"""Typed, fail-fast configuration for the MetroPT research study."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

BASE_FEATURE_NAMES = (
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Motor_current",
    "Oil_temperature",
    "Caudal_impulses",
    "COMP",
    "DV_eletric",
    "Towers",
    "MPG",
    "LPS",
    "Pressure_switch",
    "Oil_level",
)

BINARY_FEATURE_NAMES = (
    "COMP",
    "DV_eletric",
    "Towers",
    "MPG",
    "LPS",
    "Pressure_switch",
    "Oil_level",
    "Caudal_impulses",
)

ROLLING_AGGREGATIONS = ("mean", "median", "std", "skew", "min", "max")

DEFAULT_MAINTENANCE_WINDOWS = (
    ("2020-04-12 11:50:00", "2020-04-12 23:30:00", "#1", "high"),
    ("2020-04-18 00:00:00", "2020-04-18 23:59:00", "#2", "high"),
    ("2020-04-19 00:00:00", "2020-04-19 01:30:00", "#3", "high"),
    ("2020-04-29 03:20:00", "2020-04-29 04:00:00", "#4", "high"),
    ("2020-04-29 22:00:00", "2020-04-29 22:20:00", "#5", "high"),
    ("2020-05-13 14:00:00", "2020-05-13 23:59:00", "#6", "high"),
    ("2020-05-18 05:00:00", "2020-05-18 05:30:00", "#7", "high"),
    ("2020-05-19 10:10:00", "2020-05-19 11:00:00", "#8", "high"),
    ("2020-05-19 22:10:00", "2020-05-19 23:59:00", "#9", "high"),
    ("2020-05-20 00:00:00", "2020-05-20 20:00:00", "#10", "high"),
    ("2020-05-23 09:50:00", "2020-05-23 10:10:00", "#11", "high"),
    ("2020-05-29 23:30:00", "2020-05-29 23:59:00", "#12", "high"),
    ("2020-05-30 00:00:00", "2020-05-30 06:00:00", "#13", "high"),
    ("2020-06-01 15:00:00", "2020-06-01 15:40:00", "#14", "high"),
    ("2020-06-03 10:00:00", "2020-06-03 11:00:00", "#15", "high"),
    ("2020-06-05 10:00:00", "2020-06-05 23:59:00", "#16", "high"),
    ("2020-06-06 00:00:00", "2020-06-06 23:59:00", "#17", "high"),
    ("2020-06-07 00:00:00", "2020-06-07 14:30:00", "#18", "high"),
    ("2020-07-08 17:30:00", "2020-07-08 19:00:00", "#19", "high"),
    ("2020-07-15 14:30:00", "2020-07-15 19:00:00", "#20", "medium"),
    ("2020-07-17 04:30:00", "2020-07-17 05:30:00", "#21", "high"),
)

DEFAULT_WORKFLOWS = (
    "iforest_static",
    "iforest_per_maintenance",
    "sae_static",
    "vae_static",
    "nianetvae_per_maintenance",
)


@dataclass(frozen=True)
class DataConfig:
    input_path: str = "data/metropt_dataset/MetroPT3.csv"
    timestamp_column: str | None = None
    rolling_window: str = "60s"
    initial_train_minutes: int = 43_200
    post_maintenance_train_minutes: int = 600
    pre_maintenance_minutes: int = 120
    sequence_length: int = 200
    stride: int = 1
    validation_fraction: float = 0.10
    base_feature_names: tuple[str, ...] = BASE_FEATURE_NAMES
    binary_feature_names: tuple[str, ...] = BINARY_FEATURE_NAMES
    train_phases: tuple[int, ...] = (0, 1)
    test_phases: tuple[int, ...] = (0, 1)
    maintenance_windows: tuple[tuple[str, str, str, str], ...] = DEFAULT_MAINTENANCE_WINDOWS


@dataclass(frozen=True)
class PreprocessingConfig:
    policy: str = "binary_passthrough_v1"
    fit_scope: str = "initial_baseline_train"
    frozen_after_fit: bool = True


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    batch_size: int = 64
    learning_rate: float = 0.003
    weight_decay: float = 1e-5
    kl_beta: float = 0.001
    sae_sparsity_beta: float = 0.05
    sae_sparsity_rho: float = 0.05
    min_epochs: int = 3
    max_epochs: int = 30
    patience: int = 2
    min_delta: float = 0.0001
    restore_best_weights: bool = True
    deterministic: bool = True
    shuffle: bool = True
    drop_last: bool = False
    num_workers: int = 2
    pin_memory: bool = False
    persistent_workers: bool = False
    device: str = "auto"


@dataclass(frozen=True)
class AdaptationConfig:
    learning_rate_scale: float = 0.1
    baseline_replay_fraction: float = 0.50
    local_validation_fraction: float = 0.20
    validation_embargo: bool = True
    short_local_fallback: str = "train_all_fixed_min_epochs"
    min_epochs: int = 3
    max_epochs: int = 30


@dataclass(frozen=True)
class SearchConfig:
    enabled: bool = True
    n_partitions: int = 8
    max_generations: int = 300
    max_time: str = "72:00:00"
    checkpoint_interval_generations: int = 1
    candidate_min_epochs: int = 3
    candidate_max_epochs: int = 4
    reconstruction_metric: str = "SMAPE"
    pdm_metric: str = "one_minus_smoothed_auroc"
    alarm_burden_metric: str = "normal_high_risk_rate"
    alarm_burden_risk_threshold: float = 0.95
    winner_weights: tuple[float, float, float] = (0.20, 0.50, 0.30)
    invalid_penalty: float = 9e10
    database_backend: str = "sqlite"
    database_path: str = "search/candidates.sqlite"
    database_table: str = "architecture_candidates_v1"


@dataclass(frozen=True)
class CalibrationConfig:
    method: str = "empirical_cdf_v1"
    reference_scope: str = "fixed_initial_baseline"
    exceedance_quantile: float = 0.95


@dataclass(frozen=True)
class EvaluationConfig:
    risk_window_minutes: int = 120
    theta_grid: str = "0.10:0.90:0.05"
    extra_thresholds: tuple[float, ...] = (0.925, 0.95, 0.975, 0.985, 0.99, 0.995)
    target_recall: float = 0.60
    target_coverage: float = 0.20
    fixed_lead_minutes: int = 120
    sensitivity_leads: tuple[int, ...] = (30, 60, 90)
    lead_step_minutes: int = 30
    selection_scope: str = "retrospective_full_timeline"


@dataclass(frozen=True)
class ArtifactConfig:
    root: str = "artifacts"
    study_id: str = "metropt_controlled_v1"
    save_predictions: bool = True
    save_models: bool = True
    save_plots: bool = True


@dataclass(frozen=True)
class StudyConfig:
    schema_version: str = "1.0"
    study_name: str = "NiaNetVAE MetroPT controlled comparison"
    data: DataConfig = field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    adaptation: AdaptationConfig = field(default_factory=AdaptationConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    artifacts: ArtifactConfig = field(default_factory=ArtifactConfig)
    workflows: tuple[str, ...] = DEFAULT_WORKFLOWS

    def validate(self) -> StudyConfig:
        if self.schema_version != "1.0":
            raise ValueError(f"Unsupported study schema_version={self.schema_version!r}.")
        if self.preprocessing.policy != "binary_passthrough_v1":
            raise ValueError(
                "The controlled v1 study requires preprocessing.policy='binary_passthrough_v1'."
            )
        if not self.preprocessing.frozen_after_fit:
            raise ValueError("The controlled study requires a frozen preprocessor.")
        if self.preprocessing.fit_scope != "initial_baseline_train":
            raise ValueError(
                "Preprocessing must be fitted only on the initial baseline training rows."
            )
        if self.calibration.reference_scope != "fixed_initial_baseline":
            raise ValueError("Calibration must use the fixed initial-baseline reference rows.")
        if self.calibration.method != "empirical_cdf_v1":
            raise ValueError("The controlled study requires empirical_cdf_v1 calibration.")
        if self.evaluation.selection_scope != "retrospective_full_timeline":
            raise ValueError("The v1 operating point must be labeled retrospective_full_timeline.")
        if not 0.0 < self.data.validation_fraction < 0.5:
            raise ValueError("data.validation_fraction must be in (0, 0.5).")
        if self.data.sequence_length < 2 or self.data.stride < 1:
            raise ValueError("sequence_length must be >=2 and stride must be >=1.")
        if self.data.stride != 1:
            raise ValueError("The controlled v1 study requires data.stride=1.")
        if (
            min(
                self.data.initial_train_minutes,
                self.data.post_maintenance_train_minutes,
                self.data.pre_maintenance_minutes,
            )
            < 1
        ):
            raise ValueError("All temporal data horizons must be positive.")
        if set(self.data.train_phases) - {0, 1} or set(self.data.test_phases) - {0, 1}:
            raise ValueError("Only phases 0 and 1 may be eligible for training/testing.")
        if not self.workflows:
            raise ValueError("At least one controlled workflow must be enabled.")
        unknown = sorted(set(self.workflows) - set(DEFAULT_WORKFLOWS))
        if unknown:
            raise ValueError(f"Unsupported workflows: {unknown}.")
        if len(set(self.workflows)) != len(self.workflows):
            raise ValueError("workflows must be unique.")
        if self.training.seed < 0:
            raise ValueError("training.seed must be non-negative.")
        if self.training.batch_size < 1 or self.training.learning_rate <= 0.0:
            raise ValueError("Training batch size and learning rate must be positive.")
        if self.training.weight_decay < 0.0 or self.training.num_workers < 0:
            raise ValueError("Training weight decay and worker count cannot be negative.")
        if self.training.kl_beta < 0.0 or self.training.sae_sparsity_beta < 0.0:
            raise ValueError("Recurrent regularization weights cannot be negative.")
        if not 0.0 < self.training.sae_sparsity_rho < 1.0:
            raise ValueError("training.sae_sparsity_rho must be in (0,1).")
        if self.training.min_epochs < 1 or self.training.max_epochs < self.training.min_epochs:
            raise ValueError("Invalid shared training epoch bounds.")
        if self.training.patience < 1:
            raise ValueError("training.patience must be at least one epoch.")
        if not 0.0 <= self.adaptation.baseline_replay_fraction < 1.0:
            raise ValueError("baseline_replay_fraction must be in [0,1).")
        if not 0.0 < self.adaptation.local_validation_fraction < 1.0:
            raise ValueError("local_validation_fraction must be in (0,1).")
        if self.adaptation.learning_rate_scale <= 0.0:
            raise ValueError("adaptation.learning_rate_scale must be positive.")
        if (
            self.adaptation.min_epochs < 1
            or self.adaptation.max_epochs < self.adaptation.min_epochs
        ):
            raise ValueError("Invalid adaptation epoch bounds.")
        if self.search.n_partitions < 1 or self.search.max_generations < 1:
            raise ValueError("NSGA-III partitions and generations must be positive.")
        if self.search.checkpoint_interval_generations < 1:
            raise ValueError("Search checkpoint interval must be positive.")
        if (
            self.search.candidate_min_epochs < 1
            or self.search.candidate_max_epochs < self.search.candidate_min_epochs
        ):
            raise ValueError("Invalid search-candidate epoch bounds.")
        if len(self.search.winner_weights) != 3 or any(
            value < 0 for value in self.search.winner_weights
        ):
            raise ValueError("search.winner_weights must contain three non-negative values.")
        if sum(self.search.winner_weights) <= 0:
            raise ValueError("search.winner_weights must have a positive sum.")
        if self.search.database_backend != "sqlite":
            raise ValueError("The controlled v1 study requires a SQLite candidate ledger.")
        database_path = Path(self.search.database_path)
        if database_path.is_absolute() or ".." in database_path.parts:
            raise ValueError("search.database_path must remain inside the study artifact root.")
        if self.search.invalid_penalty <= 0.0:
            raise ValueError("search.invalid_penalty must be positive.")
        if not 0.0 <= self.search.alarm_burden_risk_threshold <= 1.0:
            raise ValueError("Search alarm-burden risk threshold must be in [0,1].")
        _validate_duration(self.search.max_time)
        if self.search.reconstruction_metric != "SMAPE":
            raise ValueError("The controlled search requires SMAPE reconstruction error.")
        if self.search.pdm_metric != "one_minus_smoothed_auroc":
            raise ValueError("The controlled search requires one_minus_smoothed_auroc.")
        if self.search.alarm_burden_metric != "normal_high_risk_rate":
            raise ValueError("The controlled search requires normal_high_risk_rate alarm burden.")
        if not self.artifacts.save_predictions or not self.artifacts.save_models:
            raise ValueError("Controlled evidence requires predictions and model checkpoints.")
        if not 0.0 < self.calibration.exceedance_quantile < 1.0:
            raise ValueError("Calibration exceedance_quantile must be in (0,1).")
        if not 0.0 <= self.evaluation.target_recall <= 1.0:
            raise ValueError("Evaluation target_recall must be in [0,1].")
        if not 0.0 <= self.evaluation.target_coverage <= 1.0:
            raise ValueError("Evaluation target_coverage must be in [0,1].")
        if any(not 0.0 <= float(value) <= 1.0 for value in self.evaluation.extra_thresholds):
            raise ValueError("Every extra evaluation threshold must be in [0,1].")
        if self.evaluation.fixed_lead_minutes < 1 or self.evaluation.risk_window_minutes < 1:
            raise ValueError("Evaluation lead and risk windows must be positive.")
        if any(
            int(value) <= 0 or int(value) >= self.evaluation.fixed_lead_minutes
            for value in self.evaluation.sensitivity_leads
        ):
            raise ValueError("Sensitivity leads must be positive and below the fixed horizon.")
        study_id = Path(self.artifacts.study_id)
        if (
            not self.artifacts.study_id.strip()
            or study_id.is_absolute()
            or len(study_id.parts) != 1
            or self.artifacts.study_id in {".", ".."}
        ):
            raise ValueError("artifacts.study_id must be one safe directory name.")
        return self

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        controlled = self.as_dict()
        # Search termination is a replaceable execution budget. It does not
        # alter the candidate population, objectives, trainer, or data contract.
        controlled["search"].pop("max_generations", None)
        controlled["search"].pop("max_time", None)
        payload = json.dumps(controlled, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def resolved_fingerprint(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


_SECTIONS = {
    "data": DataConfig,
    "preprocessing": PreprocessingConfig,
    "training": TrainingConfig,
    "adaptation": AdaptationConfig,
    "search": SearchConfig,
    "calibration": CalibrationConfig,
    "evaluation": EvaluationConfig,
    "artifacts": ArtifactConfig,
}


def _validate_duration(value: str) -> None:
    pieces = [piece.strip() for piece in str(value).split(":")]
    if len(pieces) != 3:
        raise ValueError("search.max_time must use HH:MM:SS syntax.")
    try:
        hours, minutes, seconds = (int(piece) for piece in pieces)
    except ValueError as exc:
        raise ValueError("search.max_time must contain integer fields.") from exc
    if hours < 0 or minutes not in range(60) or seconds not in range(60):
        raise ValueError("search.max_time contains invalid duration fields.")
    if hours * 3600 + minutes * 60 + seconds < 1:
        raise ValueError("search.max_time must be positive.")


def _coerce_tuple_fields(cls: type, values: dict[str, Any]) -> dict[str, Any]:
    out = dict(values)
    tuple_fields = {
        DataConfig: {
            "base_feature_names",
            "binary_feature_names",
            "train_phases",
            "test_phases",
            "maintenance_windows",
        },
        SearchConfig: {"winner_weights"},
        EvaluationConfig: {"extra_thresholds", "sensitivity_leads"},
    }.get(cls, set())
    for key in tuple_fields:
        if key in out:
            if key == "maintenance_windows":
                out[key] = tuple(tuple(item) for item in out[key])
            else:
                out[key] = tuple(out[key])
    return out


def load_study_config(path: str | Path) -> StudyConfig:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Study configuration not found: {source}")
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("Study configuration must be a YAML mapping.")
    allowed = {"schema_version", "study_name", "workflows", *_SECTIONS}
    unexpected = sorted(set(payload) - allowed)
    if unexpected:
        raise ValueError(f"Unknown top-level configuration keys: {unexpected}")
    kwargs: dict[str, Any] = {}
    for name, cls in _SECTIONS.items():
        section = payload.get(name, {}) or {}
        if not isinstance(section, dict):
            raise ValueError(f"Configuration section {name!r} must be a mapping.")
        valid_fields = set(cls.__dataclass_fields__)
        unknown = sorted(set(section) - valid_fields)
        if unknown:
            raise ValueError(f"Unknown keys in {name}: {unknown}")
        kwargs[name] = cls(**_coerce_tuple_fields(cls, section))
    if "schema_version" in payload:
        kwargs["schema_version"] = str(payload["schema_version"])
    if "study_name" in payload:
        kwargs["study_name"] = str(payload["study_name"])
    if "workflows" in payload:
        kwargs["workflows"] = tuple(str(item) for item in payload["workflows"])
    return StudyConfig(**kwargs).validate()

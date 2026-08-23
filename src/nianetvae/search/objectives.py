"""Candidate training and the three frozen NSGA-III objectives."""

from __future__ import annotations

import gc
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from ..config import StudyConfig
from ..dataloaders.metropt import PreparedMetroPTData, cycle_source_and_anchor_masks
from ..dataloaders.sequences import contiguous_frames
from ..evaluation.calibration import EmpiricalCDFCalibrator
from ..evaluation.risk import build_segmented_maintenance_risk
from ..training.trainer import RecurrentRuntime
from .genome import decode_genome
from .storage import CandidateStore


def _calibration_drift(
    risk: pd.Series,
    labels,
    *,
    exceedance_quantile: float,
) -> tuple[float, float | None, float | None]:
    """Measure how far a model's calibration slips across the scoring window.

    Scores are calibrated once against the frozen initial baseline, so a
    well-behaved architecture should keep exceeding the calibration quantile at
    roughly the nominal rate for the whole cycle. In practice reconstruction
    error grows as the machine drifts away from the training period, and the
    observed exceedance rate climbs far above nominal, which is what makes a
    stale detector fire on ordinary operation.

    The objective is the absolute change in exceedance rate between the first
    and second chronological half of the normal windows. It needs no failure
    labels, so unlike a supervised term it does not depend on the single
    failure present in the cycle-0 search population.
    """
    normal = risk.to_numpy(dtype=float)[labels == 0]
    if len(normal) < 2:
        return 1.0, None, None
    midpoint = len(normal) // 2
    threshold = float(exceedance_quantile)
    early = float(np.mean(normal[:midpoint] >= threshold))
    late = float(np.mean(normal[midpoint:] >= threshold))
    return float(abs(late - early)), early, late

class CandidateEvaluator:
    def __init__(
        self,
        config: StudyConfig,
        prepared: PreparedMetroPTData,
        store: CandidateStore,
        search_contract_fingerprint: str,
    ) -> None:
        self.config = config
        self.prepared = prepared
        self.store = store
        self.search_contract_fingerprint = search_contract_fingerprint

    def evaluate(self, genome) -> tuple[float, float, float]:
        architecture = decode_genome(
            genome,
            input_dim=len(self.prepared.feature_names),
            sequence_length=self.config.data.sequence_length,
        )
        cached = self.store.lookup(
            study_id=self.config.artifacts.study_id,
            search_contract_fingerprint=self.search_contract_fingerprint,
            architecture_hash=architecture.architecture_hash,
        )
        if cached is not None:
            return (
                float(cached["obj_error"]),
                float(cached["obj_pdm"]),  # column name is historical
                float(cached["obj_alarm_burden"]),
            )

        penalty = float(self.config.search.invalid_penalty)
        runtime: RecurrentRuntime | None = None
        try:
            training = replace(
                self.config.training,
                min_epochs=self.config.search.candidate_min_epochs,
                max_epochs=self.config.search.candidate_max_epochs,
            )
            runtime = RecurrentRuntime(architecture, training)
            fit = runtime.fit(
                contiguous_frames(self.prepared.scaled_features, self.prepared.baseline_train_mask),
                contiguous_frames(
                    self.prepared.scaled_features,
                    self.prepared.baseline_validation_mask,
                ),
                min_epochs=self.config.search.candidate_min_epochs,
                max_epochs=self.config.search.candidate_max_epochs,
                early_stopping=True,
            )
            source_mask, anchor_mask = self._cycle_zero_masks()
            source_segments = contiguous_frames(self.prepared.scaled_features, source_mask)
            obj_error = runtime.reconstruction_smape(source_segments)

            calibration_scores = runtime.score_segments(
                contiguous_frames(self.prepared.scaled_features, self.prepared.baseline_mask)
            ).reindex(self.prepared.calibration_mask[self.prepared.calibration_mask].index)
            if calibration_scores.isna().any():
                raise ValueError("Candidate missed fixed calibration timestamps.")
            calibrator = EmpiricalCDFCalibrator.fit(calibration_scores)
            scores = runtime.score_segments(source_segments).reindex(anchor_mask[anchor_mask].index)
            if scores.isna().any():
                raise ValueError("Candidate missed cycle-0 search timestamps.")
            risk = calibrator.transform(scores)
            smoothed = build_segmented_maintenance_risk(
                risk,
                [pd.Series(True, index=risk.index, dtype=bool)],
                exceedance_quantile=self.config.calibration.exceedance_quantile,
                risk_window_minutes=self.config.evaluation.risk_window_minutes,
            )
            labels = (
                self.prepared.operation_phase.reindex(smoothed.index).to_numpy(dtype=int) == 1
            ).astype(int)
            positive_count = int(labels.sum())
            negative_count = int((labels == 0).sum())
            if negative_count:
                normal_high_risk_rate = float(
                    np.mean(
                        smoothed.to_numpy(dtype=float)[labels == 0]
                        >= self.config.search.alarm_burden_risk_threshold
                    )
                )
                obj_stability, early_rate, late_rate = _calibration_drift(
                    risk.reindex(smoothed.index),
                    labels,
                    exceedance_quantile=self.config.calibration.exceedance_quantile,
                )
                invalid_reason = None
            else:
                normal_high_risk_rate = 1.0
                obj_stability, early_rate, late_rate = 1.0, None, None
                invalid_reason = "cycle_zero_search_population_has_no_normal_windows"
            # Retained as a diagnostic only. On the single-failure cycle-0 population this
            # is below chance for every architecture, which is why it is no longer an
            # objective; it stays recorded so the effect remains auditable.
            auroc = (
                float(roc_auc_score(labels, smoothed.to_numpy(dtype=float)))
                if positive_count and negative_count
                else None
            )
            objectives = (obj_error, obj_stability, normal_high_risk_rate)
            if not np.isfinite(np.asarray(objectives, dtype=float)).all():
                raise ValueError("Candidate produced non-finite objectives.")
            parameters = int(sum(parameter.numel() for parameter in runtime.model.parameters()))
            diagnostics: dict[str, Any] = {
                "completed_epochs": fit.completed_epochs,
                "best_epoch": fit.best_epoch,
                "best_validation_loss": fit.best_validation_loss,
                "training_windows": fit.training_windows,
                "validation_windows": fit.validation_windows,
                "calibration_windows": len(calibration_scores),
                "search_windows": len(scores),
                "positive_windows": positive_count,
                "negative_windows": negative_count,
                "smoothed_auroc": auroc,
                "smoothed_rank_gap": (2.0 * auroc - 1.0) if auroc is not None else None,
                "normal_high_risk_rate": normal_high_risk_rate,
                "calibration_drift": obj_stability,
                "early_exceedance_rate": early_rate,
                "late_exceedance_rate": late_rate,
                "stability_invalid_reason": invalid_reason,
                "parameter_count": parameters,
            }
            self.store.insert(
                study_id=self.config.artifacts.study_id,
                search_contract_fingerprint=self.search_contract_fingerprint,
                architecture_hash=architecture.architecture_hash,
                genome=architecture.genome or (),
                architecture=architecture.as_dict(),
                status="valid",
                objectives=objectives,
                diagnostics=diagnostics,
            )
            return objectives
        except Exception as error:
            objectives = (penalty, penalty, penalty)
            self.store.insert(
                study_id=self.config.artifacts.study_id,
                search_contract_fingerprint=self.search_contract_fingerprint,
                architecture_hash=architecture.architecture_hash,
                genome=architecture.genome or (),
                architecture=architecture.as_dict(),
                status="invalid",
                objectives=objectives,
                diagnostics={"reason": "candidate_evaluation_failed"},
                error=error,
            )
            return objectives
        finally:
            if runtime is not None:
                del runtime
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _cycle_zero_masks(self) -> tuple[pd.Series, pd.Series]:
        source, anchors = cycle_source_and_anchor_masks(
            self.prepared,
            self.prepared.cycles[0],
            self.config.data.test_phases,
        )
        if not anchors.any():
            raise ValueError("Cycle-0 search population has no shared evaluation anchors.")
        return source, anchors

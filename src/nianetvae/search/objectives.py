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
from ..dataloaders.metropt import PreparedMetroPTData
from ..dataloaders.sequences import contiguous_frames
from ..evaluation.calibration import EmpiricalCDFCalibrator
from ..evaluation.risk import build_segmented_maintenance_risk
from ..training.trainer import RecurrentRuntime
from .genome import decode_genome
from .storage import CandidateStore


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
                float(cached["obj_pdm"]),
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
            if positive_count and negative_count:
                auroc = float(roc_auc_score(labels, smoothed.to_numpy(dtype=float)))
                obj_pdm = float(np.clip(1.0 - auroc, 0.0, 1.0))
                normal_high_risk_rate = float(
                    np.mean(
                        smoothed.to_numpy(dtype=float)[labels == 0]
                        >= self.config.search.alarm_burden_risk_threshold
                    )
                )
                invalid_reason = None
            else:
                auroc = None
                obj_pdm = 1.0
                normal_high_risk_rate = 1.0
                invalid_reason = "cycle_zero_search_population_missing_positive_or_negative_class"
            objectives = (obj_error, obj_pdm, normal_high_risk_rate)
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
                "pdm_invalid_reason": invalid_reason,
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
        index = self.prepared.scaled_features.index
        baseline_times = self.prepared.baseline_mask[self.prepared.baseline_mask].index
        start = pd.Timestamp(baseline_times.max())
        end = self.prepared.cycles[0].score_end
        source = pd.Series((index > start) & (index < end), index=index, dtype=bool)
        source &= self.prepared.operation_phase.isin(self.config.data.test_phases)
        source &= ~self.prepared.baseline_mask
        source &= ~self.prepared.post_maintenance_train_mask
        anchors = self.prepared.evaluation_mask & source
        if not anchors.any():
            raise ValueError("Cycle-0 search population has no shared evaluation anchors.")
        return source, anchors

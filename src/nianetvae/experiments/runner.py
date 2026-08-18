"""Run the five controlled MetroPT workflows through one shared core."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..artifacts import (
    StudyArtifactStore,
    atomic_write_csv,
    atomic_write_json,
    read_json,
    sha256_file,
    utc_now,
)
from ..config import StudyConfig
from ..contracts import (
    ArchitectureSpec,
    WorkflowSpec,
    handcrafted_sae_spec,
    handcrafted_vae_spec,
    iforest_spec,
)
from ..dataloaders.metropt import (
    CyclePlan,
    PreparedMetroPTData,
    cycle_source_and_anchor_masks,
)
from ..dataloaders.sequences import contiguous_frames
from ..evaluation.calibration import EmpiricalCDFCalibrator
from ..evaluation.event import evaluate_maintenance_prediction
from ..evaluation.islands import analyze_alarm_islands
from ..evaluation.risk import (
    build_segmented_maintenance_risk,
    evaluate_risk_thresholds,
    select_operating_point,
)
from ..models.iforest import IsolationForestRuntime
from ..training.trainer import FitResult, RecurrentRuntime
from ..visualization import plot_theta_tradeoff, plot_workflow_comparison, plot_workflow_timeline

Runtime = IsolationForestRuntime | RecurrentRuntime


class WorkflowRunner:
    """Incremental and resumable runner over a prepared immutable study."""

    def __init__(
        self,
        config: StudyConfig,
        prepared: PreparedMetroPTData,
        store: StudyArtifactStore,
    ) -> None:
        self.config = config.validate()
        self.prepared = prepared
        self.store = store
        self.store.assert_initialized(config, prepared)

    def run_workflow(self, workflow_id: str) -> dict[str, Any]:
        self._validate_workflow_enabled(workflow_id)
        manifest_path = self.store.workflow_manifest_path(workflow_id)
        if manifest_path.is_file():
            existing = read_json(manifest_path)
            if existing.get("status") == "completed":
                summary = self.store.workflow_dir(workflow_id) / "workflow_summary.json"
                if summary.is_file():
                    return read_json(summary)
        for cycle in self.prepared.cycles:
            self.run_cycle(workflow_id, cycle.cycle_id)
        return self.finalize_workflow(workflow_id)

    def run_cycle(self, workflow_id: str, cycle_id: int) -> dict[str, Any]:
        lock_name = f"{workflow_id}-cycle-{int(cycle_id):02d}"
        with self.store.exclusive_lock(lock_name, timeout_seconds=1.0):
            return self._run_cycle_locked(workflow_id, cycle_id)

    def _run_cycle_locked(self, workflow_id: str, cycle_id: int) -> dict[str, Any]:
        self._validate_workflow_enabled(workflow_id)
        workflow = WorkflowSpec.from_id(workflow_id)
        architecture = self._architecture(workflow)
        manifest = self._ensure_workflow_manifest(workflow, architecture)
        if manifest.get("status") == "completed":
            raise ValueError(
                f"Workflow {workflow_id} is already completed. Use a new study_id for a rerun."
            )
        cycle = self._cycle(cycle_id)
        cycle_dir = self.store.workflow_dir(workflow_id) / "cycles" / f"cycle_{cycle_id:02d}"
        result_path = cycle_dir / "cycle_result.json"
        if result_path.is_file():
            result = read_json(result_path)
            self._validate_cycle_result(result, workflow_id, architecture)
            return result
        if cycle_id > 0:
            predecessor = self._cycle_result(workflow_id, cycle_id - 1)
        else:
            predecessor = None

        cycle_dir.mkdir(parents=True, exist_ok=True)
        try:
            runtime, model_status, effective_model_cycle, fit_result, policy = (
                self._runtime_for_cycle(
                    workflow,
                    architecture,
                    cycle,
                    predecessor,
                )
            )
            checkpoint_path = self._checkpoint_for_cycle(
                workflow,
                cycle,
                runtime,
                model_status=model_status,
                predecessor=predecessor,
            )
            calibrator = self._fit_calibrator(runtime, workflow)
            scores = self._score_cycle(runtime, workflow, cycle)
            risk_scores = calibrator.transform(scores)
            segment_mask = pd.Series(True, index=risk_scores.index, dtype=bool)
            maintenance_risk = build_segmented_maintenance_risk(
                risk_scores,
                [segment_mask],
                exceedance_quantile=self.config.calibration.exceedance_quantile,
                risk_window_minutes=self.config.evaluation.risk_window_minutes,
            )
            predictions = pd.DataFrame(
                {
                    "timestamp": scores.index,
                    "cycle_id": cycle.cycle_id,
                    "operation_phase": self.prepared.operation_phase.reindex(
                        scores.index
                    ).to_numpy(),
                    "anomaly_score": scores.to_numpy(dtype=float),
                    "risk_score": risk_scores.to_numpy(dtype=float),
                    "maintenance_risk": maintenance_risk.reindex(scores.index).to_numpy(
                        dtype=float
                    ),
                }
            )
            prediction_path = atomic_write_csv(cycle_dir / "predictions.csv", predictions)
            calibration_payload = {
                "method": calibrator.method,
                "reference_scope": self.config.calibration.reference_scope,
                "reference_count": len(calibrator.sorted_reference_scores),
                "reference_index_hash": calibrator.reference_index_hash,
                "calibrator_fingerprint": calibrator.fingerprint,
                "minimum_score": float(calibrator.sorted_reference_scores[0]),
                "maximum_score": float(calibrator.sorted_reference_scores[-1]),
            }
            calibration_path = atomic_write_json(
                cycle_dir / "calibration.json", calibration_payload
            )
            result = {
                "schema_version": "1.0",
                "study_id": self.config.artifacts.study_id,
                "workflow_id": workflow_id,
                "cycle_id": cycle.cycle_id,
                "source_event_id": cycle.source_event_id,
                "status": "completed",
                "model_status": model_status,
                "effective_model_cycle": effective_model_cycle,
                "architecture_hash": architecture.architecture_hash,
                "checkpoint": self.store.relative(checkpoint_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "predictions": self.store.relative(prediction_path),
                "predictions_sha256": sha256_file(prediction_path),
                "calibration": self.store.relative(calibration_path),
                "calibration_sha256": sha256_file(calibration_path),
                "prediction_count": len(predictions),
                "evaluation_status": ("scored" if len(predictions) else "no_evaluation_anchors"),
                "fit_result": asdict(fit_result) if fit_result is not None else None,
                "training_policy": policy,
            }
            atomic_write_json(result_path, result)
            return result
        except Exception as error:
            self.store.mark_failed(workflow_id, error)
            raise

    def finalize_workflow(self, workflow_id: str) -> dict[str, Any]:
        with self.store.exclusive_lock(f"{workflow_id}-finalize", timeout_seconds=1.0):
            return self._finalize_workflow_locked(workflow_id)

    def _finalize_workflow_locked(self, workflow_id: str) -> dict[str, Any]:
        self._validate_workflow_enabled(workflow_id)
        workflow = WorkflowSpec.from_id(workflow_id)
        architecture = self._architecture(workflow)
        manifest = self._ensure_workflow_manifest(workflow, architecture)
        if manifest.get("status") == "completed":
            summary_path = self.store.workflow_dir(workflow_id) / "workflow_summary.json"
            if not summary_path.is_file():
                raise FileNotFoundError(
                    f"Completed workflow {workflow_id} is missing its summary: {summary_path}"
                )
            return read_json(summary_path)
        results = [
            self._cycle_result(workflow_id, cycle.cycle_id) for cycle in self.prepared.cycles
        ]
        frames: list[pd.DataFrame] = []
        for result in results:
            path = self.store.root / str(result["predictions"])
            frame = pd.read_csv(path, parse_dates=["timestamp"])
            frames.append(frame)
        scored_frames = [frame for frame in frames if not frame.empty]
        if not scored_frames:
            raise ValueError(f"Workflow {workflow_id} has no predictions in any cycle.")
        predictions = pd.concat(scored_frames, ignore_index=True).sort_values("timestamp")
        if predictions["timestamp"].duplicated().any():
            duplicates = (
                predictions.loc[predictions["timestamp"].duplicated(), "timestamp"].head().tolist()
            )
            raise ValueError(f"Cycle predictions overlap at timestamps: {duplicates}")

        risk = pd.Series(np.nan, index=self.prepared.scaled_features.index, dtype=float)
        risk.loc[pd.DatetimeIndex(predictions["timestamp"])] = predictions[
            "maintenance_risk"
        ].to_numpy(dtype=float)
        sweep_rows = evaluate_risk_thresholds(
            risk,
            self.prepared.events,
            eval_mask=self.prepared.evaluation_mask,
            config=self.config.evaluation,
        )
        selected = select_operating_point(sweep_rows, self.config.evaluation)
        alarm = (risk >= float(selected["maintenance_risk_theta"])) & self.prepared.evaluation_mask
        predictions["alarm"] = alarm.reindex(pd.DatetimeIndex(predictions["timestamp"])).to_numpy(
            dtype=bool
        )
        metrics = evaluate_maintenance_prediction(
            alarm,
            self.prepared.events,
            self.config.evaluation.fixed_lead_minutes,
            method_name=workflow_id,
            eval_mask=self.prepared.evaluation_mask,
            lead_step_minutes=self.config.evaluation.lead_step_minutes,
            sensitivity_leads=self.config.evaluation.sensitivity_leads,
        )
        islands, island_summary = analyze_alarm_islands(
            alarm,
            self.prepared.events,
            early_warning_minutes=self.config.evaluation.fixed_lead_minutes,
            eval_mask=self.prepared.evaluation_mask,
        )

        directory = self.store.workflow_dir(workflow_id)
        predictions_path = atomic_write_csv(directory / "predictions.csv", predictions)
        sweep_path = atomic_write_csv(
            directory / "theta_sweep_maintenance_risk.csv", pd.DataFrame(sweep_rows)
        )
        selected_path = atomic_write_json(directory / "selected_operating_point.json", selected)
        metrics_path = atomic_write_json(directory / "event_metrics.json", metrics)
        islands_path = atomic_write_csv(directory / "alarm_islands.csv", islands)
        island_summary_path = atomic_write_json(
            directory / "alarm_island_summary.json", island_summary
        )
        summary = _workflow_summary(workflow_id, selected, metrics, island_summary)
        summary_path = atomic_write_json(directory / "workflow_summary.json", summary)
        outputs = {
            "predictions": self.store.relative(predictions_path),
            "theta_sweep": self.store.relative(sweep_path),
            "selected_operating_point": self.store.relative(selected_path),
            "event_metrics": self.store.relative(metrics_path),
            "alarm_islands": self.store.relative(islands_path),
            "alarm_island_summary": self.store.relative(island_summary_path),
            "workflow_summary": self.store.relative(summary_path),
        }
        if self.config.artifacts.save_plots:
            timeline_path = plot_workflow_timeline(
                predictions,
                events=self.prepared.events,
                selected_theta=float(selected["maintenance_risk_theta"]),
                coverage_percent=float(metrics["coverage"]["alarm_coverage_percent"]),
                output=directory / "maintenance_risk_timeline.png",
            )
            tradeoff_path = plot_theta_tradeoff(
                pd.DataFrame(sweep_rows),
                selected_theta=float(selected["maintenance_risk_theta"]),
                output=directory / "theta_tradeoff.png",
            )
            outputs["maintenance_risk_timeline"] = self.store.relative(timeline_path)
            outputs["theta_tradeoff"] = self.store.relative(tradeoff_path)
        self.store.complete_workflow(
            workflow_id,
            manifest,
            outputs=outputs,
            cycle_lineage=results,
        )
        return summary

    def _runtime_for_cycle(
        self,
        workflow: WorkflowSpec,
        architecture: ArchitectureSpec,
        cycle: CyclePlan,
        predecessor: dict[str, Any] | None,
    ) -> tuple[Runtime, str, int, FitResult | None, dict[str, Any]]:
        if cycle.cycle_id == 0:
            runtime, fit = self._train_initial(workflow, architecture)
            return (
                runtime,
                "trained_initial",
                0,
                fit,
                {
                    "mode": "initial_baseline",
                    "preprocessing_fit_scope": self.config.preprocessing.fit_scope,
                },
            )

        assert predecessor is not None
        if workflow.strategy == "static":
            initial = self._cycle_result(workflow.workflow_id, 0)
            return (
                self._load_runtime(workflow, architecture, self.store.root / initial["checkpoint"]),
                "reused_static",
                0,
                None,
                {"mode": "static_no_update"},
            )

        if not cycle.update_possible:
            return (
                self._load_runtime(
                    workflow, architecture, self.store.root / predecessor["checkpoint"]
                ),
                "alias_no_trainable_local_window",
                int(predecessor["effective_model_cycle"]),
                None,
                {
                    "mode": "alias",
                    "reason": "post_maintenance_interval_has_no_complete_sequence_and_score_gap",
                },
            )

        local_segments = self._local_segments(cycle)
        if workflow.model_kind == "iforest":
            runtime = IsolationForestRuntime(seed=self.config.training.seed)
            training = pd.concat(
                [
                    self.prepared.scaled_features.loc[self.prepared.baseline_train_mask],
                    *local_segments,
                ]
            )
            runtime.fit(training)
            return (
                runtime,
                "refitted",
                cycle.cycle_id,
                None,
                {
                    "mode": "initial_baseline_plus_current_local_refit",
                    "baseline_rows": int(self.prepared.baseline_train_mask.sum()),
                    "local_rows": int(sum(len(segment) for segment in local_segments)),
                },
            )

        runtime = self._load_runtime(
            workflow, architecture, self.store.root / predecessor["checkpoint"]
        )
        local_train, local_validation, policy = self._split_local_segments(local_segments)
        fallback = bool(policy["short_local_fallback_applied"])
        fit = runtime.fine_tune(
            baseline_segments=contiguous_frames(
                self.prepared.scaled_features,
                self.prepared.baseline_train_mask,
            ),
            local_train_segments=local_train,
            local_validation_segments=local_validation,
            baseline_replay_fraction=self.config.adaptation.baseline_replay_fraction,
            learning_rate_scale=self.config.adaptation.learning_rate_scale,
            min_epochs=self.config.adaptation.min_epochs,
            max_epochs=(
                self.config.adaptation.min_epochs if fallback else self.config.adaptation.max_epochs
            ),
            early_stopping=not fallback,
        )
        return runtime, "fine_tuned", cycle.cycle_id, fit, policy

    def _train_initial(
        self,
        workflow: WorkflowSpec,
        architecture: ArchitectureSpec,
    ) -> tuple[Runtime, FitResult | None]:
        if workflow.model_kind == "iforest":
            runtime = IsolationForestRuntime(seed=self.config.training.seed)
            runtime.fit(self.prepared.scaled_features.loc[self.prepared.baseline_train_mask])
            return runtime, None
        runtime = RecurrentRuntime(architecture, self.config.training)
        fit = runtime.fit(
            contiguous_frames(self.prepared.scaled_features, self.prepared.baseline_train_mask),
            contiguous_frames(
                self.prepared.scaled_features, self.prepared.baseline_validation_mask
            ),
            min_epochs=self.config.training.min_epochs,
            max_epochs=self.config.training.max_epochs,
            early_stopping=True,
        )
        return runtime, fit

    def _fit_calibrator(self, runtime: Runtime, workflow: WorkflowSpec) -> EmpiricalCDFCalibrator:
        if workflow.model_kind == "iforest":
            assert isinstance(runtime, IsolationForestRuntime)
            scores = runtime.score(
                self.prepared.scaled_features.loc[self.prepared.calibration_mask]
            )
        else:
            assert isinstance(runtime, RecurrentRuntime)
            scores = runtime.score_segments(
                contiguous_frames(self.prepared.scaled_features, self.prepared.baseline_mask)
            ).reindex(self.prepared.calibration_mask[self.prepared.calibration_mask].index)
        expected = self.prepared.calibration_mask[self.prepared.calibration_mask].index
        if not scores.index.equals(expected):
            missing = expected.difference(scores.index)
            raise ValueError(
                "Model calibration did not score the fixed reference population; "
                f"missing={len(missing)}."
            )
        return EmpiricalCDFCalibrator.fit(scores)

    def _score_cycle(self, runtime: Runtime, workflow: WorkflowSpec, cycle: CyclePlan) -> pd.Series:
        source_mask, anchor_mask = self._cycle_masks(cycle)
        expected = anchor_mask[anchor_mask].index
        if expected.empty:
            return pd.Series(index=expected, dtype=float, name="anomaly_score")
        if workflow.model_kind == "iforest":
            assert isinstance(runtime, IsolationForestRuntime)
            scores = runtime.score(self.prepared.scaled_features.loc[anchor_mask])
        else:
            assert isinstance(runtime, RecurrentRuntime)
            scores = runtime.score_segments(
                contiguous_frames(self.prepared.scaled_features, source_mask)
            ).reindex(expected)
        if not scores.index.equals(expected) or scores.isna().any():
            missing = expected.difference(scores.dropna().index)
            raise ValueError(
                f"Workflow {workflow.workflow_id} cycle {cycle.cycle_id} failed the shared "
                f"evaluation-population contract; expected={len(expected)}, missing={len(missing)}."
            )
        return scores.astype(float)

    def _cycle_masks(self, cycle: CyclePlan) -> tuple[pd.Series, pd.Series]:
        return cycle_source_and_anchor_masks(
            self.prepared,
            cycle,
            self.config.data.test_phases,
        )

    def _local_segments(self, cycle: CyclePlan) -> list[pd.DataFrame]:
        if cycle.update_start is None or cycle.update_end is None:
            raise ValueError(f"Cycle {cycle.cycle_id} has no local update interval.")
        index = self.prepared.scaled_features.index
        mask = pd.Series(
            (index > cycle.update_start) & (index <= cycle.update_end),
            index=index,
            dtype=bool,
        )
        mask &= self.prepared.operation_phase.isin(self.config.data.train_phases)
        segments = contiguous_frames(self.prepared.scaled_features, mask)
        if (
            sum(max(0, len(segment) - self.config.data.sequence_length + 1) for segment in segments)
            < 1
        ):
            raise ValueError(f"Cycle {cycle.cycle_id} produced no local sequence windows.")
        return segments

    def _split_local_segments(
        self,
        segments: list[pd.DataFrame],
    ) -> tuple[list[pd.DataFrame], list[pd.DataFrame], dict[str, Any]]:
        sequence_length = self.config.data.sequence_length
        segment_windows = [max(0, len(segment) - sequence_length + 1) for segment in segments]
        usable_segments = [
            (segment, windows)
            for segment, windows in zip(segments, segment_windows, strict=True)
            if windows > 0
        ]
        total_windows = sum(windows for _segment, windows in usable_segments)
        if total_windows < 1:
            raise ValueError("Local update segments produced no complete sequence windows.")
        requested_validation = max(
            1,
            int(np.floor(total_windows * self.config.adaptation.local_validation_fraction)),
        )
        requested_embargo = sequence_length - 1 if self.config.adaptation.validation_embargo else 0

        validation_windows_by_segment = [0] * len(usable_segments)
        remaining_validation = requested_validation
        for index in range(len(usable_segments) - 1, -1, -1):
            windows = usable_segments[index][1]
            selected = min(windows, remaining_validation)
            validation_windows_by_segment[index] = selected
            remaining_validation -= selected
            if remaining_validation == 0:
                break

        train_segments: list[pd.DataFrame] = []
        validation_segments: list[pd.DataFrame] = []
        local_train_windows = 0
        local_validation_windows = 0
        applied_embargo = 0
        unused_rows = 0
        for (segment, windows), validation_windows in zip(
            usable_segments,
            validation_windows_by_segment,
            strict=True,
        ):
            if validation_windows == 0:
                train_segments.append(segment)
                local_train_windows += windows
                continue
            if validation_windows == windows:
                validation_segments.append(segment)
                local_validation_windows += windows
                continue

            validation_start_window = windows - validation_windows
            segment_embargo = min(requested_embargo, validation_start_window)
            train_windows = validation_start_window - segment_embargo
            if train_windows > 0:
                train_raw_end = (train_windows - 1) + sequence_length
                train_segments.append(segment.iloc[:train_raw_end].copy())
                unused_rows += validation_start_window - train_raw_end
            else:
                unused_rows += validation_start_window
            validation_segments.append(segment.iloc[validation_start_window:].copy())
            local_train_windows += train_windows
            local_validation_windows += validation_windows
            applied_embargo += segment_embargo

        fallback = local_train_windows < 1
        if fallback:
            if self.config.adaptation.short_local_fallback != "train_all_fixed_min_epochs":
                raise ValueError(
                    "Local update interval is too short for non-overlapping "
                    "train/validation windows."
                )
            train_segments = [segment for segment, _windows in usable_segments]
            validation_segments = []
            local_train_windows = total_windows
            local_validation_windows = 0
            applied_embargo = 0
            strategy = "disabled_short_local_fallback"
            unused_rows = 0
        else:
            strategy = (
                "chronological_non_overlapping_local"
                if len(usable_segments) == 1
                else "chronological_non_overlapping_local_segments"
            )
        baseline_windows = sum(
            max(0, len(segment) - sequence_length + 1)
            for segment in contiguous_frames(
                self.prepared.scaled_features,
                self.prepared.baseline_train_mask,
            )
        )
        replay = (
            int(
                round(
                    local_train_windows
                    * self.config.adaptation.baseline_replay_fraction
                    / (1.0 - self.config.adaptation.baseline_replay_fraction)
                )
            )
            if self.config.adaptation.baseline_replay_fraction > 0.0
            else 0
        )
        replay = min(baseline_windows, replay)
        return (
            train_segments,
            validation_segments,
            {
                "mode": "local_train_with_frozen_baseline_replay",
                "local_segment_count": len(segments),
                "usable_local_segment_count": len(usable_segments),
                "local_train_segment_count": len(train_segments),
                "local_validation_segment_count": len(validation_segments),
                "local_total_rows": sum(len(segment) for segment in segments),
                "local_total_windows": total_windows,
                "local_train_windows": local_train_windows,
                "local_validation_windows": local_validation_windows,
                "requested_validation_windows": requested_validation,
                "requested_embargo_windows": requested_embargo,
                "applied_embargo_windows": applied_embargo,
                "unused_boundary_rows": unused_rows,
                "validation_strategy": strategy,
                "early_stopping_eligible": not fallback,
                "short_local_fallback_applied": fallback,
                "baseline_total_windows": baseline_windows,
                "baseline_replay_windows": replay,
                "baseline_replay_fraction": self.config.adaptation.baseline_replay_fraction,
                "learning_rate": self.config.training.learning_rate
                * self.config.adaptation.learning_rate_scale,
            },
        )

    def _checkpoint_for_cycle(
        self,
        workflow: WorkflowSpec,
        cycle: CyclePlan,
        runtime: Runtime,
        *,
        model_status: str,
        predecessor: dict[str, Any] | None,
    ) -> Path:
        if model_status.startswith("reused") or model_status.startswith("alias"):
            assert predecessor is not None
            return self.store.root / str(predecessor["checkpoint"])
        suffix = ".joblib" if workflow.model_kind == "iforest" else ".pt"
        target = (
            self.store.workflow_dir(workflow.workflow_id)
            / "models"
            / f"cycle_{cycle.cycle_id:02d}{suffix}"
        )
        runtime.save(target)
        return target

    def _load_runtime(
        self,
        workflow: WorkflowSpec,
        architecture: ArchitectureSpec,
        checkpoint: Path,
    ) -> Runtime:
        if workflow.model_kind == "iforest":
            return IsolationForestRuntime.load(checkpoint)
        return RecurrentRuntime.load(checkpoint, architecture, self.config.training)

    def _architecture(self, workflow: WorkflowSpec) -> ArchitectureSpec:
        input_dim = len(self.prepared.feature_names)
        sequence_length = self.config.data.sequence_length
        if workflow.model_kind == "iforest":
            return iforest_spec(input_dim)
        if workflow.workflow_id == "sae_static":
            return handcrafted_sae_spec(input_dim, sequence_length)
        if workflow.workflow_id == "vae_static":
            return handcrafted_vae_spec(input_dim, sequence_length)
        selected_path = self.store.search_dir / "selected_architecture.json"
        search_manifest_path = self.store.search_dir / "search_manifest.json"
        if not selected_path.is_file():
            raise FileNotFoundError(
                f"NiaNetVAE requires a fresh shared-core search result: {selected_path}."
            )
        if not search_manifest_path.is_file():
            raise FileNotFoundError(
                f"NiaNetVAE search manifest is missing: {search_manifest_path}."
            )
        payload = read_json(selected_path)
        if payload.get("study_id") != self.config.artifacts.study_id:
            raise ValueError("Selected architecture belongs to another study_id.")
        from ..search.engine import search_contract

        expected_search = search_contract(self.config, self.prepared)["search_contract_fingerprint"]
        if payload.get("search_contract_fingerprint") != expected_search:
            raise ValueError("Selected architecture belongs to another search contract.")
        search_manifest = read_json(search_manifest_path)
        selected_hash = search_manifest.get("output_sha256", {}).get("selected_architecture")
        if (
            search_manifest.get("status") != "completed"
            or search_manifest.get("search_contract_fingerprint") != expected_search
            or not selected_hash
            or sha256_file(selected_path) != selected_hash
        ):
            raise ValueError("Selected architecture failed search-manifest validation.")
        architecture = ArchitectureSpec.from_dict(payload["architecture"])
        if architecture.input_dim != input_dim or architecture.sequence_length != sequence_length:
            raise ValueError("Selected architecture does not match the current data contract.")
        return architecture

    def _ensure_workflow_manifest(
        self,
        workflow: WorkflowSpec,
        architecture: ArchitectureSpec,
    ) -> dict[str, Any]:
        path = self.store.workflow_manifest_path(workflow.workflow_id)
        if not path.is_file():
            return self.store.begin_workflow(
                workflow.workflow_id,
                config=self.config,
                prepared=self.prepared,
                architecture=architecture.as_dict(),
            )
        manifest = read_json(path)
        if manifest.get("study_config_fingerprint") != self.config.fingerprint():
            raise ValueError(f"Workflow {workflow.workflow_id} uses a different study config.")
        recorded = ArchitectureSpec.from_dict(manifest["architecture"])
        if recorded.architecture_hash != architecture.architecture_hash:
            raise ValueError(f"Workflow {workflow.workflow_id} architecture changed during resume.")
        return manifest

    def _cycle(self, cycle_id: int) -> CyclePlan:
        if cycle_id < 0 or cycle_id >= len(self.prepared.cycles):
            raise ValueError(f"cycle_id must be in [0,{len(self.prepared.cycles) - 1}].")
        return self.prepared.cycles[cycle_id]

    def _cycle_result(self, workflow_id: str, cycle_id: int) -> dict[str, Any]:
        path = (
            self.store.workflow_dir(workflow_id)
            / "cycles"
            / f"cycle_{cycle_id:02d}"
            / "cycle_result.json"
        )
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing predecessor/result for {workflow_id} cycle {cycle_id}: {path}"
            )
        return read_json(path)

    def _validate_cycle_result(
        self,
        result: dict[str, Any],
        workflow_id: str,
        architecture: ArchitectureSpec,
    ) -> None:
        if result.get("schema_version") != "1.0" or result.get("status") != "completed":
            raise ValueError(f"Invalid saved cycle result for {workflow_id}.")
        if result.get("workflow_id") != workflow_id:
            raise ValueError("Saved cycle result belongs to another workflow.")
        if result.get("architecture_hash") != architecture.architecture_hash:
            raise ValueError("Saved cycle result architecture does not match the active workflow.")
        for key in ("checkpoint", "predictions", "calibration"):
            artifact = self.store.root / str(result[key])
            if not artifact.is_file():
                raise FileNotFoundError(f"Saved cycle result is missing {key}: {result[key]}")
            expected_hash = result.get(f"{key}_sha256")
            if not expected_hash or sha256_file(artifact) != expected_hash:
                raise ValueError(f"Saved cycle result has an invalid {key} hash.")
        cycle = self._cycle(int(result.get("cycle_id", -1)))
        expected_index = self._cycle_masks(cycle)[1]
        expected_index = expected_index[expected_index].index
        expected_status = "scored" if len(expected_index) else "no_evaluation_anchors"
        if int(result.get("prediction_count", -1)) != len(expected_index):
            raise ValueError("Saved cycle result has an invalid prediction count.")
        if result.get("evaluation_status") != expected_status:
            raise ValueError("Saved cycle result has an invalid evaluation status.")
        predictions = pd.read_csv(
            self.store.root / str(result["predictions"]),
            usecols=["timestamp", "cycle_id"],
            parse_dates=["timestamp"],
        )
        if len(predictions) != len(expected_index):
            raise ValueError("Saved cycle predictions have an invalid row count.")
        if not predictions.empty and not predictions["cycle_id"].eq(cycle.cycle_id).all():
            raise ValueError("Saved cycle predictions contain another cycle id.")
        if not pd.DatetimeIndex(predictions["timestamp"]).equals(expected_index):
            raise ValueError("Saved cycle predictions changed the evaluation timestamps.")

    def _validate_workflow_enabled(self, workflow_id: str) -> None:
        if workflow_id not in self.config.workflows:
            raise ValueError(f"Workflow {workflow_id!r} is not enabled in this study.")


def _workflow_summary(
    workflow_id: str,
    selected: dict[str, Any],
    metrics: dict[str, Any],
    island_summary: dict[str, Any],
) -> dict[str, Any]:
    event = metrics["event_scores"]
    return {
        "schema_version": "1.0",
        "workflow_id": workflow_id,
        "selection_scope": selected["selection_scope"],
        "selection_mode": selected["selection_mode"],
        "maintenance_risk_theta": selected["maintenance_risk_theta"],
        "tp": event["tp"],
        "fp": event["fp"],
        "fn": event["fn"],
        "precision": event["precision"],
        "recall": event["recall"],
        "f1": event["f1"],
        "ttd_minutes": metrics["ttd"]["mean_ttd"],
        "faa": metrics["first_alarm_accuracy"]["first_alarm_accuracy"],
        "far_per_day": metrics["far"]["far_per_day"],
        "coverage": metrics["coverage"]["alarm_coverage"],
        "coverage_percent": metrics["coverage"]["alarm_coverage_percent"],
        "mtia_minutes": metrics["mtia"]["mtia_minutes"],
        "nab_standard": metrics["nab"]["standard"]["nab_score_normalized"],
        "nab_low_fp": metrics["nab"]["low_fp"]["nab_score_normalized"],
        "nab_low_fn": metrics["nab"]["low_fn"]["nab_score_normalized"],
        "alarm_islands": island_summary["island_count"],
        "false_alarm_islands": island_summary["false_alarm_island_count"],
    }


def build_comparison(config: StudyConfig, store: StudyArtifactStore) -> pd.DataFrame:
    with store.exclusive_lock("workflow-comparison", timeout_seconds=1.0):
        return _build_comparison_locked(config, store)


def _build_comparison_locked(config: StudyConfig, store: StudyArtifactStore) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for workflow_id in config.workflows:
        path = store.workflow_dir(workflow_id) / "workflow_summary.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing workflow summary: {path}")
        rows.append(read_json(path))
    frame = pd.DataFrame(rows)
    csv_path = atomic_write_csv(store.comparison_dir / "workflow_comparison.csv", frame)
    json_path = atomic_write_json(
        store.comparison_dir / "workflow_comparison.json",
        {
            "schema_version": "1.0",
            "selection_scope": config.evaluation.selection_scope,
            "rows": rows,
        },
    )
    outputs = {
        "workflow_comparison_csv": store.relative(csv_path),
        "workflow_comparison_json": store.relative(json_path),
    }
    if config.artifacts.save_plots:
        plot_path = plot_workflow_comparison(
            frame, store.comparison_dir / "workflow_comparison.png"
        )
        outputs["workflow_comparison_plot"] = store.relative(plot_path)
    comparison_manifest = {
        "schema_version": "1.0",
        "study_id": config.artifacts.study_id,
        "created_at": utc_now(),
        "selection_scope": config.evaluation.selection_scope,
        "workflows": list(config.workflows),
        "outputs": outputs,
        "output_sha256": {
            label: sha256_file(store.root / relative) for label, relative in outputs.items()
        },
    }
    manifest_path = atomic_write_json(
        store.comparison_dir / "comparison_manifest.json", comparison_manifest
    )

    def update(study: dict[str, Any]) -> None:
        study["comparison"] = {
            "status": "completed",
            "manifest": store.relative(manifest_path),
        }
        study["status"] = "comparison_completed"
        study["updated_at"] = utc_now()

    store.update_study_manifest(update)
    return frame

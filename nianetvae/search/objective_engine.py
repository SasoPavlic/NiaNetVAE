import math
import time

import numpy as np
import torch

from log import Log

DEFAULT_PENALTY = int(9e10)


def _safe_float(value) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _resolve_objective_contract(cfg: dict | None = None) -> dict:
    cfg = cfg or {}
    objectives = dict(cfg.get("objectives") or {})
    error_metric = str(((objectives.get("error") or {}).get("metric") or "SMAPE")).strip().upper()
    efficiency_metric = str(((objectives.get("efficiency") or {}).get("metric") or "macs")).strip().lower()
    pdm_metric = str(((objectives.get("pdm") or {}).get("metric") or "auprc_premaint")).strip().lower()
    return {
        "error_metric": error_metric,
        "efficiency_metric": efficiency_metric,
        "pdm_metric": pdm_metric,
    }


def _penalty_objective_bundle(
    reason: str,
    objective_contract: dict | None = None,
    cfg: dict | None = None,
    penalty: int | float = DEFAULT_PENALTY,
) -> dict:
    penalty_value = float(penalty)
    return {
        "valid": False,
        "reason": reason,
        "objective_contract": objective_contract or _resolve_objective_contract(cfg),
        "obj_error": penalty_value,
        "obj_efficiency": penalty_value,
        "obj_pdm": penalty_value,
        "pdm_signal_quality": None,
        "fitness": penalty_value,
    }


def _model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cpu")


def _model_forward(model, signal_batch: torch.Tensor):
    try:
        return model({"signal": signal_batch})
    except Exception:
        return model(signal_batch)


def _estimate_model_macs(model, seq_len: int, n_features: int) -> tuple[float | None, str | None]:
    was_training = bool(model.training) if hasattr(model, "training") else False
    device = _model_device(model)
    dummy_signal = torch.zeros((1, int(seq_len), int(n_features)), dtype=torch.float32, device=device)
    try:
        if hasattr(model, "eval"):
            model.eval()
        try:
            from thop import profile as thop_profile

            macs, _ = thop_profile(model, inputs=({"signal": dummy_signal},), verbose=False)
            macs_value = _safe_float(macs)
            if macs_value is not None and macs_value > 0:
                return macs_value, None
        except Exception:
            pass

        if not hasattr(torch, "profiler") or not hasattr(torch.profiler, "profile"):
            return None, "macs_profiler_unavailable"
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
            torch.cuda.synchronize()
        with torch.inference_mode():
            with torch.profiler.profile(activities=activities, with_flops=True) as prof:
                _model_forward(model, dummy_signal)
        if device.type == "cuda":
            torch.cuda.synchronize()

        total_flops = 0.0
        for event in prof.key_averages():
            flops_value = _safe_float(getattr(event, "flops", None))
            if flops_value is not None and flops_value > 0:
                total_flops += flops_value
        if total_flops <= 0:
            return None, "macs_flops_not_reported"
        return float(total_flops / 2.0), None
    except Exception as exc:
        return None, f"macs_estimation_failed:{exc.__class__.__name__}"
    finally:
        try:
            if hasattr(model, "train"):
                model.train(was_training)
        except Exception:
            pass


def _estimate_model_latency_ms(
    model,
    seq_len: int,
    n_features: int,
    warmup_steps: int = 3,
    measure_steps: int = 7,
) -> tuple[float | None, str | None]:
    was_training = bool(model.training) if hasattr(model, "training") else False
    device = _model_device(model)
    dummy_signal = torch.zeros((1, int(seq_len), int(n_features)), dtype=torch.float32, device=device)
    try:
        if hasattr(model, "eval"):
            model.eval()
        with torch.inference_mode():
            for _ in range(max(1, int(warmup_steps))):
                _model_forward(model, dummy_signal)
                if device.type == "cuda":
                    torch.cuda.synchronize()

            durations_ms = []
            for _ in range(max(1, int(measure_steps))):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                started = time.perf_counter()
                _model_forward(model, dummy_signal)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                ended = time.perf_counter()
                durations_ms.append((ended - started) * 1000.0)

        if not durations_ms:
            return None, "latency_no_samples"
        latency_ms = _safe_float(float(np.median(np.asarray(durations_ms, dtype=np.float64))))
        if latency_ms is None or latency_ms <= 0:
            return None, "latency_non_finite"
        return latency_ms, None
    except Exception as exc:
        return None, f"latency_estimation_failed:{exc.__class__.__name__}"
    finally:
        try:
            if hasattr(model, "train"):
                model.train(was_training)
        except Exception:
            pass


def _compute_efficiency_objective(model, metric_name: str, seq_len: int, n_features: int) -> tuple[float | None, str | None]:
    metric = str(metric_name).strip().lower()
    if metric == "params":
        try:
            value = float(sum(int(p.numel()) for p in model.parameters()))
        except Exception as exc:
            return None, f"params_count_failed:{exc.__class__.__name__}"
        if value <= 0 or not math.isfinite(value):
            return None, "params_non_finite"
        return value, None

    if metric == "macs":
        return _estimate_model_macs(model, seq_len=seq_len, n_features=n_features)

    if metric == "latency_ms":
        return _estimate_model_latency_ms(model, seq_len=seq_len, n_features=n_features)

    return None, f"unsupported_efficiency_metric:{metric}"


def _metrics_payload_from_cached_entry(entry: dict | None) -> dict:
    entry = entry or {}
    metric_keys = ("MAE", "MSE", "RMSE", "MAPE", "RMAPE", "SMAPE")
    return {metric_key: entry.get(metric_key) for metric_key in metric_keys}


def _anomaly_payload_from_cached_entry(entry: dict | None) -> dict:
    entry = entry or {}
    keys = (
        "precision",
        "recall",
        "f1_score",
        "pr_auc_mean",
        "pr_auc_std",
        "roc_auc_mean",
        "roc_auc_std",
    )
    return {key: entry.get(key) for key in keys}


def calculate_objective_bundle(
    model,
    metrics_payload: dict | None,
    anomaly_metrics: dict | None,
    seq_len: int,
    n_features: int,
    cfg: dict | None = None,
    penalty: int | float = DEFAULT_PENALTY,
) -> dict:
    cfg = cfg or {}
    objective_contract = _resolve_objective_contract(cfg)
    metrics_payload = metrics_payload or {}
    anomaly_metrics = anomaly_metrics or {}

    error_metric = objective_contract["error_metric"]
    obj_error = _safe_float(metrics_payload.get(error_metric))
    if obj_error is None:
        return _penalty_objective_bundle(
            reason=f"missing_or_invalid_error_metric:{error_metric}",
            objective_contract=objective_contract,
            penalty=penalty,
        )

    obj_efficiency, eff_reason = _compute_efficiency_objective(
        model=model,
        metric_name=objective_contract["efficiency_metric"],
        seq_len=int(seq_len),
        n_features=int(n_features),
    )
    if obj_efficiency is None:
        Log.warning(
            "OBJECTIVE_EFFICIENCY_FALLBACK "
            f"metric={objective_contract['efficiency_metric']} "
            f"reason={eff_reason or 'invalid_efficiency_objective'} penalty=true"
        )
        return _penalty_objective_bundle(
            reason=eff_reason or "invalid_efficiency_objective",
            objective_contract=objective_contract,
            penalty=penalty,
        )

    pdm_signal_quality = _safe_float(anomaly_metrics.get("pr_auc_mean"))
    if pdm_signal_quality is None:
        Log.warning(
            "OBJECTIVE_PDM_FALLBACK "
            "metric=auprc_premaint reason=missing_or_invalid_pdm_signal_quality penalty=true"
        )
        return _penalty_objective_bundle(
            reason="missing_or_invalid_pdm_signal_quality",
            objective_contract=objective_contract,
            penalty=penalty,
        )

    obj_pdm = _safe_float(1.0 - pdm_signal_quality)
    if obj_pdm is None:
        return _penalty_objective_bundle(
            reason="invalid_obj_pdm",
            objective_contract=objective_contract,
            penalty=penalty,
        )

    fitness = _safe_float(obj_error + obj_efficiency)
    if fitness is None:
        return _penalty_objective_bundle(
            reason="invalid_compatibility_fitness",
            objective_contract=objective_contract,
            penalty=penalty,
        )

    return {
        "valid": True,
        "reason": None,
        "objective_contract": objective_contract,
        "obj_error": float(obj_error),
        "obj_efficiency": float(obj_efficiency),
        "obj_pdm": float(obj_pdm),
        "pdm_signal_quality": float(pdm_signal_quality),
        # Compatibility fields retained until winner selector migration finalization.
        "fitness": float(fitness),
    }


def calculate_objective_bundle_from_experiment(
    model,
    experiment,
    seq_len: int,
    n_features: int,
    cfg: dict | None = None,
    penalty: int | float = DEFAULT_PENALTY,
):
    cfg = cfg or {}
    objective_contract = _resolve_objective_contract(cfg)
    if experiment is None or getattr(experiment, "metrics", None) is None:
        return _penalty_objective_bundle(
            "missing_experiment_metrics",
            objective_contract=objective_contract,
            penalty=penalty,
        )

    try:
        if not experiment.metrics.are_metrics_complete():
            return _penalty_objective_bundle(
                reason="incomplete_metrics",
                objective_contract=objective_contract,
                penalty=penalty,
            )
        metrics_payload = experiment.metrics.compute()
    except Exception as exc:
        return _penalty_objective_bundle(
            reason=f"metrics_compute_failed:{exc.__class__.__name__}",
            objective_contract=objective_contract,
            penalty=penalty,
        )

    anomaly_metrics = getattr(experiment, "anomaly_metrics", {}) or {}
    return calculate_objective_bundle(
        model=model,
        metrics_payload=metrics_payload,
        anomaly_metrics=anomaly_metrics,
        seq_len=seq_len,
        n_features=n_features,
        cfg=cfg,
        penalty=penalty,
    )


def calculate_objective_bundle_from_cached_row(
    model,
    cached_row,
    seq_len: int,
    n_features: int,
    cfg: dict | None = None,
    penalty: int | float = DEFAULT_PENALTY,
):
    return calculate_objective_bundle(
        model=model,
        metrics_payload=_metrics_payload_from_cached_entry(cached_row),
        anomaly_metrics=_anomaly_payload_from_cached_entry(cached_row),
        seq_len=seq_len,
        n_features=n_features,
        cfg=cfg,
        penalty=penalty,
    )

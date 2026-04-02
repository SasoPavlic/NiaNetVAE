import json
import math
from datetime import datetime

import numpy as np

from .objective_engine import _safe_float
from .runtime_artifacts import _as_jsonable


def _resolve_winner_selection_contract(cfg: dict | None = None) -> dict:
    cfg = cfg or {}
    objectives = dict(cfg.get("objectives") or {})
    selection_cfg = dict(objectives.get("selection") or {})

    method = str(selection_cfg.get("method", "weighted_ideal_distance")).strip().lower()
    if method != "weighted_ideal_distance":
        raise ValueError(
            f"Unsupported objectives.selection.method={method!r}. "
            "Allowed value: weighted_ideal_distance."
        )

    default_weights = {"error": 0.30, "efficiency": 0.20, "pdm": 0.50}
    raw_weights_cfg = dict(selection_cfg.get("weights") or {})
    resolved_weights = {}
    for key in ("error", "efficiency", "pdm"):
        raw_value = raw_weights_cfg.get(key, default_weights[key])
        try:
            value = float(raw_value)
        except Exception:
            raise ValueError(
                f"Invalid objectives.selection.weights.{key}={raw_value!r}. Expected finite float >= 0."
            ) from None
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                f"Invalid objectives.selection.weights.{key}={raw_value!r}. Expected finite float >= 0."
            )
        resolved_weights[key] = value

    total = resolved_weights["error"] + resolved_weights["efficiency"] + resolved_weights["pdm"]
    if total <= 0:
        raise ValueError(
            "Invalid objectives.selection.weights: sum(error, efficiency, pdm) must be > 0."
        )
    weights_normalized = {
        "error": resolved_weights["error"] / total,
        "efficiency": resolved_weights["efficiency"] / total,
        "pdm": resolved_weights["pdm"] / total,
    }
    return {
        "method": method,
        "weights": resolved_weights,
        "weights_normalized": weights_normalized,
    }


def _parse_solution_array(raw_value):
    if raw_value is None:
        return None
    parsed = None
    if isinstance(raw_value, str):
        try:
            parsed = json.loads(raw_value)
        except Exception:
            return None
    else:
        parsed = raw_value

    try:
        arr = np.asarray(parsed, dtype=float)
    except Exception:
        return None
    if arr.size == 0:
        return None
    if not np.isfinite(arr).all():
        return None
    return arr


def _parse_timestamp_sort_key(raw_value):
    if raw_value is None:
        return float("inf")
    if isinstance(raw_value, datetime):
        dt = raw_value
    else:
        text = str(raw_value).strip()
        if not text:
            return float("inf")
        dt = None
        try:
            dt = datetime.fromisoformat(text)
        except Exception:
            dt = None
        if dt is None:
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
                try:
                    dt = datetime.strptime(text, fmt)
                    break
                except Exception:
                    dt = None
    if dt is None:
        return float("inf")
    try:
        return float(dt.timestamp())
    except Exception:
        return float("inf")


def _pareto_mask_minimize(objectives: np.ndarray) -> np.ndarray:
    count = int(objectives.shape[0])
    keep = np.ones(count, dtype=bool)
    for i in range(count):
        if not keep[i]:
            continue
        for j in range(count):
            if i == j:
                continue
            dominates = np.all(objectives[j] <= objectives[i]) and np.any(objectives[j] < objectives[i])
            if dominates:
                keep[i] = False
                break
    return keep


def _select_deterministic_pareto_winner(
    candidates_df,
    selection_contract: dict,
    dataset_name: str,
    penalty: int | float,
):
    penalty_value = float(penalty)

    if candidates_df is None:
        raise ValueError("Winner selection failed: no candidate rows were returned from DB.")

    try:
        records = candidates_df.to_dict(orient="records")
    except Exception:
        records = []
    candidate_count = len(records)
    if candidate_count == 0:
        raise ValueError(
            f"Winner selection failed for {dataset_name}: no DB candidates found in cycle-scoped pool."
        )

    valid = []
    for row in records:
        obj_error = _safe_float(row.get("error"))
        obj_efficiency = _safe_float(row.get("complexity"))
        pr_auc_mean = _safe_float(row.get("pr_auc_mean"))
        obj_pdm = _safe_float(1.0 - pr_auc_mean) if pr_auc_mean is not None else None
        solution = _parse_solution_array(row.get("solution_array"))
        if obj_error is None or obj_efficiency is None or obj_pdm is None or solution is None:
            continue
        if obj_error >= penalty_value or obj_efficiency >= penalty_value or obj_pdm >= penalty_value:
            continue
        valid.append(
            {
                "id": int(row.get("id")) if row.get("id") is not None else int(9e18),
                "hash_id": str(row.get("hash_id", "")),
                "algorithm_name": str(row.get("algorithm_name", "NSGA3")),
                "timestamp_sort_key": _parse_timestamp_sort_key(row.get("timestamp")),
                "obj_error": float(obj_error),
                "obj_efficiency": float(obj_efficiency),
                "obj_pdm": float(obj_pdm),
                "pdm_signal_quality": float(pr_auc_mean),
                "solution": solution,
            }
        )

    if not valid:
        raise ValueError(
            f"Winner selection failed for {dataset_name}: no valid objective candidates after filtering."
        )

    def _tie_break_key(item):
        return (
            float(item["obj_pdm"]),
            float(item["obj_error"]),
            float(item["obj_efficiency"]),
            float(item["timestamp_sort_key"]),
            int(item["id"]),
        )

    dedup = {}
    for item in valid:
        key = item["hash_id"]
        if key not in dedup or _tie_break_key(item) < _tie_break_key(dedup[key]):
            dedup[key] = item
    deduped = list(dedup.values())
    if not deduped:
        raise ValueError(
            f"Winner selection failed for {dataset_name}: no candidates after hash de-duplication."
        )

    objective_matrix = np.array(
        [[row["obj_error"], row["obj_efficiency"], row["obj_pdm"]] for row in deduped],
        dtype=float,
    )
    mask = _pareto_mask_minimize(objective_matrix)
    pareto_rows = [deduped[i] for i in range(len(deduped)) if bool(mask[i])]
    if not pareto_rows:
        raise ValueError(
            f"Winner selection failed for {dataset_name}: Pareto set is empty after filtering."
        )

    pareto_matrix = np.array(
        [[row["obj_error"], row["obj_efficiency"], row["obj_pdm"]] for row in pareto_rows],
        dtype=float,
    )
    mins = np.min(pareto_matrix, axis=0)
    maxs = np.max(pareto_matrix, axis=0)
    spans = maxs - mins
    normalized = np.zeros_like(pareto_matrix, dtype=float)
    positive_span = spans > 0
    if np.any(positive_span):
        normalized[:, positive_span] = (
            (pareto_matrix[:, positive_span] - mins[positive_span]) / spans[positive_span]
        )

    weights = selection_contract["weights_normalized"]
    distances = np.sqrt(
        normalized[:, 0] ** 2 * float(weights["error"])
        + normalized[:, 1] ** 2 * float(weights["efficiency"])
        + normalized[:, 2] ** 2 * float(weights["pdm"])
    )
    best_distance = float(np.min(distances))
    tie_indices = [i for i, d in enumerate(distances) if abs(float(d) - best_distance) <= 1e-12]
    tied_rows = [pareto_rows[i] for i in tie_indices]
    tied_rows.sort(key=_tie_break_key)
    selected = tied_rows[0]

    return {
        "method": selection_contract["method"],
        "weights": _as_jsonable(selection_contract["weights"]),
        "weights_normalized": _as_jsonable(selection_contract["weights_normalized"]),
        "candidate_count": int(candidate_count),
        "valid_candidate_count": int(len(valid)),
        "deduplicated_candidate_count": int(len(deduped)),
        "pareto_candidate_count": int(len(pareto_rows)),
        "selected_hash": selected["hash_id"],
        "selected_id": int(selected["id"]),
        "selected_algorithm": selected["algorithm_name"],
        "selected_objectives": {
            "obj_error": float(selected["obj_error"]),
            "obj_efficiency": float(selected["obj_efficiency"]),
            "obj_pdm": float(selected["obj_pdm"]),
        },
        "selected_pdm_signal_quality": float(selected["pdm_signal_quality"]),
        "selected_distance": float(best_distance),
        "selected_solution": _as_jsonable(selected["solution"]),
    }

#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER_SCRIPT="${SCRIPT_DIR}/run_per_maint_job.sbatch"
IMAGE_DIR="/d/hpc/home/sasop/images"
IMAGE_REF="docker://spartan300/nianet:vaepymoo"
IMAGE_LATEST="${IMAGE_DIR}/nianet-vaepymoo-latest.sif"
IMAGE_CURRENT="${IMAGE_DIR}/nianet-vaepymoo-current.sif"
IMAGE_SYNC="${IMAGE_SYNC:-1}"
IMAGE_BUILD_FAKEROOT="${IMAGE_BUILD_FAKEROOT:-0}"

# Normal human workflow: edit CONFIG_PATH, then run this script.
# START_CYCLE/END_CYCLE/RESUME_FROM are advanced operational overrides.
CONFIG_PATH="${CONFIG_PATH:-configs/main_config.yaml}"
START_CYCLE="${START_CYCLE:-0}"
END_CYCLE="${END_CYCLE:-21}"
RESUME_FROM="${RESUME_FROM:-auto}"
DETACHED_SUBMIT="${DETACHED_SUBMIT:-0}"
CHAIN_DEPENDENCY_TYPE="${CHAIN_DEPENDENCY_TYPE:-afterany}"

# Cluster policy constants. Research/runtime settings should come from CONFIG_PATH.
SEARCH_WALLTIME_BUFFER="10:00:00"
FINETUNE_WALLTIME="08:00:00"
MANIFEST_WALLTIME="00:30:00"
SAFE_MAX_WALLTIME="4-00:00:00"

# Optional detached mode:
#   ./submit_per_maint_pipeline.sh --detach
# This re-launches the script under nohup so SSH disconnects do not stop submission.
if [ "${1:-}" = "--detach" ] && [ "${DETACHED_SUBMIT}" != "1" ]; then
    mkdir -p outputs
    ts=$(date +%Y%m%d_%H%M%S)
    detach_log="outputs/submit_per_maint_pipeline_${ts}.log"
    nohup env DETACHED_SUBMIT=1 \
        CONFIG_PATH="${CONFIG_PATH}" \
        START_CYCLE="${START_CYCLE}" \
        END_CYCLE="${END_CYCLE}" \
        RESUME_FROM="${RESUME_FROM}" \
        CHAIN_DEPENDENCY_TYPE="${CHAIN_DEPENDENCY_TYPE}" \
        IMAGE_SYNC="${IMAGE_SYNC}" \
        IMAGE_BUILD_FAKEROOT="${IMAGE_BUILD_FAKEROOT}" \
        PROOT_NO_SECCOMP="${PROOT_NO_SECCOMP:-}" \
        bash "$0" > "${detach_log}" 2>&1 < /dev/null &
    detach_pid=$!
    echo "Detached submission started."
    echo "  pid=${detach_pid}"
    echo "  log=${detach_log}"
    echo "Track progress with: tail -f ${detach_log}"
    exit 0
fi

if [ ! -f "${WORKER_SCRIPT}" ]; then
    echo "Missing ${WORKER_SCRIPT}"
    exit 1
fi

if [ ! -f "${CONFIG_PATH}" ]; then
    echo "Missing CONFIG_PATH=${CONFIG_PATH} in $(pwd)"
    exit 1
fi

if ! [[ "${START_CYCLE}" =~ ^[0-9]+$ && "${END_CYCLE}" =~ ^[0-9]+$ ]]; then
    echo "START_CYCLE and END_CYCLE must be integers. Received START_CYCLE=${START_CYCLE}, END_CYCLE=${END_CYCLE}"
    exit 1
fi

if [ "${START_CYCLE}" -gt "${END_CYCLE}" ]; then
    echo "Invalid cycle range: START_CYCLE=${START_CYCLE} > END_CYCLE=${END_CYCLE}"
    exit 1
fi

if [ "${CHAIN_DEPENDENCY_TYPE}" != "afterok" ] && [ "${CHAIN_DEPENDENCY_TYPE}" != "afterany" ]; then
    echo "CHAIN_DEPENDENCY_TYPE must be afterany or afterok. Received: ${CHAIN_DEPENDENCY_TYPE}"
    exit 1
fi

if ! command -v singularity >/dev/null 2>&1; then
    echo "singularity command not found in PATH."
    exit 1
fi

if ! command -v sbatch >/dev/null 2>&1; then
    echo "sbatch command not found in PATH."
    exit 1
fi

if [ ! -d "configs" ]; then
    echo "Missing ./configs directory in $(pwd)"
    exit 1
fi
if [ ! -d "data" ]; then
    echo "Missing ./data directory in $(pwd)"
    exit 1
fi

mkdir -p "${IMAGE_DIR}" outputs logs

if [ "${IMAGE_SYNC}" = "1" ]; then
    echo "Syncing image from ${IMAGE_REF} ..."
    if [ "${IMAGE_BUILD_FAKEROOT}" = "1" ]; then
        singularity build --force --fakeroot "${IMAGE_LATEST}" "${IMAGE_REF}"
    elif [ "${IMAGE_BUILD_FAKEROOT}" = "0" ]; then
        singularity pull --force "${IMAGE_LATEST}" "${IMAGE_REF}"
    else
        echo "Invalid IMAGE_BUILD_FAKEROOT=${IMAGE_BUILD_FAKEROOT}. Use 1 for build --fakeroot or 0 for pull."
        exit 1
    fi
    ln -sfn "${IMAGE_LATEST}" "${IMAGE_CURRENT}"
elif [ "${IMAGE_SYNC}" = "0" ]; then
    echo "Skipping image sync because IMAGE_SYNC=0."
    if [ ! -e "${IMAGE_CURRENT}" ] && [ -f "${IMAGE_LATEST}" ]; then
        ln -sfn "${IMAGE_LATEST}" "${IMAGE_CURRENT}"
    fi
else
    echo "Invalid IMAGE_SYNC=${IMAGE_SYNC}. Use IMAGE_SYNC=1 to pull or IMAGE_SYNC=0 to use the existing SIF."
    exit 1
fi

if [ ! -f "${IMAGE_CURRENT}" ]; then
    echo "Missing active SIF image at ${IMAGE_CURRENT}."
    echo "Either allow image sync with IMAGE_SYNC=1 or place a prebuilt SIF there and rerun with IMAGE_SYNC=0."
    exit 1
fi
echo "Active image symlink: ${IMAGE_CURRENT} -> $(readlink -f "${IMAGE_CURRENT}")"

read_config_assignments() {
    singularity exec \
        -e \
        --pwd /app \
        -B "$(pwd)/configs:/app/configs" \
        "${IMAGE_CURRENT}" \
        env \
            CONFIG_PATH="${CONFIG_PATH}" \
            SEARCH_WALLTIME_BUFFER="${SEARCH_WALLTIME_BUFFER}" \
            FINETUNE_WALLTIME="${FINETUNE_WALLTIME}" \
            MANIFEST_WALLTIME="${MANIFEST_WALLTIME}" \
            SAFE_MAX_WALLTIME="${SAFE_MAX_WALLTIME}" \
            python - <<'PY'
import os
import shlex
from pathlib import Path
import yaml


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.load(handle, Loader=yaml.Loader) or {}


def load_merged_config(config_path: Path) -> dict:
    config = load_yaml(config_path)
    seen = set()
    for _ in range(5):
        dataset_cfg = (config.get("dataset") or {}).get("config_file")
        if not dataset_cfg:
            break
        candidate = Path(str(dataset_cfg))
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        candidate = candidate.resolve()
        key = str(candidate)
        if key in seen:
            raise SystemExit(f"Recursive dataset config_file reference detected: {dataset_cfg}")
        seen.add(key)
        payload = load_yaml(candidate)
        config.update(payload)
        next_cfg = (payload.get("dataset") or {}).get("config_file")
        if not next_cfg:
            if isinstance(config.get("dataset"), dict):
                config["dataset"].pop("config_file", None)
            break
    config.setdefault("data_params", {})
    config["data_params"].update(config.get("data_loader_params", {}) or {})
    return config


def parse_time_seconds(value: str) -> int:
    text = str(value).strip()
    if not text:
        raise ValueError("empty time value")
    days = 0
    if "-" in text:
        day_part, text = text.split("-", 1)
        days = int(day_part)
    parts = text.split(":")
    if len(parts) != 3:
        raise ValueError(f"time must be HH:MM:SS or D-HH:MM:SS, got {value!r}")
    hours, minutes, seconds = (int(part) for part in parts)
    if minutes < 0 or minutes >= 60 or seconds < 0 or seconds >= 60 or hours < 0 or days < 0:
        raise ValueError(f"invalid time value {value!r}")
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def format_slurm_time(total_seconds: int) -> str:
    total_seconds = int(total_seconds)
    days, rem = divmod(total_seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    if days:
        return f"{days}-{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


config_path = Path(os.environ["CONFIG_PATH"])
if not config_path.is_absolute():
    config_path = Path.cwd() / config_path
config = load_merged_config(config_path)
workflow_mode = str((config.get("workflow") or {}).get("mode") or "per_maint_baseline_search").strip().lower()
nia_search = config.get("nia_search") or {}
termination_cfg = nia_search.get("termination") or {}
termination_type = str(termination_cfg.get("type") or "time").strip().lower()
termination_n_gen = termination_cfg.get("n_gen", "")
search_time = str(termination_cfg.get("time") or nia_search.get("time") or "01:00:00").strip()
data_params = config.get("data_params") or {}
logging_params = config.get("logging_params") or {}
dataset_name = str(data_params.get("dataset_name") or "MetroPT").strip() or "MetroPT"
export_root = str(logging_params.get("model_export_dir") or "logs/per_maint_models").strip() or "logs/per_maint_models"
db_table_name = str(logging_params.get("db_table_name") or "").strip()

search_seconds = parse_time_seconds(search_time)
buffer_seconds = parse_time_seconds(os.environ["SEARCH_WALLTIME_BUFFER"])
safe_max_seconds = parse_time_seconds(os.environ["SAFE_MAX_WALLTIME"])
derived_search_seconds = min(search_seconds + buffer_seconds, safe_max_seconds)

values = {
    "WORKFLOW_MODE": workflow_mode,
    "NIA_SEARCH_TIME": search_time,
    "NIA_TERMINATION_TYPE": termination_type,
    "NIA_TERMINATION_N_GEN": termination_n_gen,
    "DERIVED_SEARCH_WALLTIME": format_slurm_time(derived_search_seconds),
    "FINETUNE_WALLTIME_RESOLVED": os.environ["FINETUNE_WALLTIME"],
    "MANIFEST_WALLTIME_RESOLVED": os.environ["MANIFEST_WALLTIME"],
    "SAFE_MAX_WALLTIME_RESOLVED": os.environ["SAFE_MAX_WALLTIME"],
    "DATASET_NAME_RESOLVED": dataset_name,
    "EXPORT_ROOT_RESOLVED": export_root,
    "DB_TABLE_NAME_RESOLVED": db_table_name,
}
for key, value in values.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
}

CONFIG_ASSIGNMENTS="$(read_config_assignments)"
eval "${CONFIG_ASSIGNMENTS}"

cycle_complete() {
    local cycle_id="$1"
    local cycle_key
    cycle_key=$(printf "%02d" "${cycle_id}")
    local cycle_dir="${EXPORT_ROOT_RESOLVED}/${DATASET_NAME_RESOLVED}/cycle_${cycle_key}"
    local status_file="${cycle_dir}/cycle_status.json"

    # Trained cycle is complete when model + metadata exist.
    if [ -f "${cycle_dir}/model.pt" ] && [ -f "${cycle_dir}/model_meta.json" ]; then
        return 0
    fi

    # Non-trainable cycle is also treated as complete when skip marker exists.
    if [ -f "${status_file}" ] && grep -q '"status"[[:space:]]*:[[:space:]]*"skipped_non_trainable"' "${status_file}"; then
        return 0
    fi

    return 1
}

# Auto-resume: find the first incomplete cycle in the requested range.
SUBMIT_FROM="${START_CYCLE}"
if [ "${RESUME_FROM}" = "auto" ]; then
    found_incomplete="false"
    for ((cid=START_CYCLE; cid<=END_CYCLE; cid++)); do
        if ! cycle_complete "${cid}"; then
            SUBMIT_FROM="${cid}"
            found_incomplete="true"
            break
        fi
    done
    if [ "${found_incomplete}" = "false" ]; then
        SUBMIT_FROM=$((END_CYCLE + 1))
        echo "All cycles ${START_CYCLE}-${END_CYCLE} already have model.pt + model_meta.json or skipped_non_trainable status."
    fi
else
    if ! [[ "${RESUME_FROM}" =~ ^[0-9]+$ ]]; then
        echo "RESUME_FROM must be 'auto' or integer. Received: ${RESUME_FROM}"
        exit 1
    fi
    SUBMIT_FROM="${RESUME_FROM}"
fi

walltime_for_cycle() {
    local cycle_id="$1"
    case "${WORKFLOW_MODE}" in
        per_maint_finetune_search)
            if [ "${cycle_id}" -eq 0 ]; then
                echo "${DERIVED_SEARCH_WALLTIME}"
            else
                echo "${FINETUNE_WALLTIME_RESOLVED}"
            fi
            ;;
        per_maint_warmstart_search|per_maint_baseline_search)
            echo "${DERIVED_SEARCH_WALLTIME}"
            ;;
        *)
            echo "Unsupported workflow.mode='${WORKFLOW_MODE}' from ${CONFIG_PATH}."
            exit 1
            ;;
    esac
}

echo "Sequential submission settings:"
echo "  config_path=${CONFIG_PATH}"
echo "  workflow_mode=${WORKFLOW_MODE}"
echo "  dataset_name=${DATASET_NAME_RESOLVED}"
echo "  db_table=${DB_TABLE_NAME_RESOLVED:-n/a}"
echo "  export_root=${EXPORT_ROOT_RESOLVED}"
echo "  requested_range=${START_CYCLE}-${END_CYCLE}"
echo "  resume_from=${RESUME_FROM}"
echo "  submit_from=${SUBMIT_FROM}"
echo "  nia_search_time=${NIA_SEARCH_TIME}"
echo "  nia_termination_type=${NIA_TERMINATION_TYPE}"
echo "  nia_termination_n_gen=${NIA_TERMINATION_N_GEN:-n/a}"
echo "  search_walltime_buffer=${SEARCH_WALLTIME_BUFFER}"
echo "  derived_search_walltime=${DERIVED_SEARCH_WALLTIME}"
echo "  finetune_walltime=${FINETUNE_WALLTIME_RESOLVED}"
echo "  manifest_walltime=${MANIFEST_WALLTIME_RESOLVED}"
echo "  safe_max_walltime=${SAFE_MAX_WALLTIME_RESOLVED}"
echo "  chain_dependency_type=${CHAIN_DEPENDENCY_TYPE}"

submitted_job_ids=()
prev_job_id=""

# Sequential order: cycle k starts after cycle k-1 ends. The default afterany
# keeps the campaign moving if one cycle fails; rerun missing cycles later with
# RESUME_FROM=auto. Use CHAIN_DEPENDENCY_TYPE=afterok for strict fail-fast runs.
for ((cid=SUBMIT_FROM; cid<=END_CYCLE; cid++)); do
    dep_args=()
    dep_info=""
    job_time="$(walltime_for_cycle "${cid}")"
    if [ -n "${prev_job_id}" ]; then
        dep_args+=(--dependency="${CHAIN_DEPENDENCY_TYPE}:${prev_job_id}")
        dep_info=" (depends on ${CHAIN_DEPENDENCY_TYPE}:${prev_job_id})"
    fi
    job_id=$(
        sbatch --parsable \
            --job-name="nianetvae-metropt" \
            --time="${job_time}" \
            "${dep_args[@]}" \
            --export="ALL,JOB_MODE=train,CONFIG_PATH=${CONFIG_PATH},CYCLE_ID=${cid}" \
            "${WORKER_SCRIPT}"
    )
    submitted_job_ids+=("${job_id}")
    prev_job_id="${job_id}"
    echo "Submitted cycle ${cid}: ${job_id} (time=${job_time})${dep_info}"
done

if [ "${#submitted_job_ids[@]}" -gt 0 ]; then
    dependency_chain=$(IFS=:; echo "${submitted_job_ids[*]}")
    manifest_dep="afterany:${dependency_chain}"
    MANIFEST_JOB_ID=$(
        sbatch --parsable \
            --job-name="nianetvae-manifest" \
            --time="${MANIFEST_WALLTIME_RESOLVED}" \
            --dependency="${manifest_dep}" \
            --export="ALL,JOB_MODE=manifest,CONFIG_PATH=${CONFIG_PATH},CYCLE_SPEC=${START_CYCLE}-${END_CYCLE}" \
            "${WORKER_SCRIPT}"
    )
    echo "Submitted training jobs: ${dependency_chain}"
    echo "Submitted manifest job: ${MANIFEST_JOB_ID} (time=${MANIFEST_WALLTIME_RESOLVED}, dependency: ${manifest_dep})"
else
    # If everything is already complete, still rebuild manifest to confirm trained/alias/missing statuses.
    MANIFEST_JOB_ID=$(
        sbatch --parsable \
            --job-name="nianetvae-manifest" \
            --time="${MANIFEST_WALLTIME_RESOLVED}" \
            --export="ALL,JOB_MODE=manifest,CONFIG_PATH=${CONFIG_PATH},CYCLE_SPEC=${START_CYCLE}-${END_CYCLE}" \
            "${WORKER_SCRIPT}"
    )
    echo "No cycle jobs submitted. Submitted manifest-only job: ${MANIFEST_JOB_ID} (time=${MANIFEST_WALLTIME_RESOLVED})"
fi

echo "Check status with: squeue --me"

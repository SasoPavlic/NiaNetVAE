#!/usr/bin/env python3
"""
Recreate, train, and evaluate a single NiaNetVAE model from a given solution vector and dataset,
then visualize metrics (e.g., training/validation loss) from the CSV logs.

Usage (PowerShell one-liner example):
  python recreate_model_from_solution.py `
    --solution "[...7 floats...]" `
    --dataset SWAT `
    --config configs/main_config.yaml `
    --outdir logs/manual_replays `
    --plot-loss

Key features:
- Auto-detects and chdir's to the repo root (folder containing `nianetvae` and `configs`),
  unless --no-chdir is passed.
- Builds DataLoader and model exactly like main.py does.
- Supports manual encoder/decoder injection for legacy architectures (see --force-manual).
- Logs to CSV + TensorBoard.
- Visualizations are integrated (no subprocess); easily extensible (see viz_* functions).
"""

import argparse
import ast
import json
import os
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
import matplotlib.pyplot as plt
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

# --- Project imports ---
from log import Log
from nianetvae.experiments.rnn_vae_experiment import RNNVAExperiment
from nianetvae.models.rnn_vae import RNNVAE

# Dataloaders (same mapping as main.py)
from nianetvae.dataloaders.kpi_dataloader import KPIDataLoader
from nianetvae.dataloaders.nab_dataloader import NABDataLoader
from nianetvae.dataloaders.smap_and_msl_dataloader import SMABandMSDataLoader
from nianetvae.dataloaders.smd_dataloader import SMDDataLoader
from nianetvae.dataloaders.swat_dataloader import SWATDataLoader
from nianetvae.dataloaders.ucr_dataloader import UCRDataLoader
from nianetvae.dataloaders.wadi_dataloader import WADIDataLoader
from nianetvae.dataloaders.yahoo_dataloader import YahooA1DataLoader
from nianetvae.dataloaders.metropt_dataloader import MetroPTDataLoader


# ============================================================
# Helpers: config, dataset, repo root
# ============================================================

def parse_solution(arg: str) -> np.ndarray:
    """Accept JSON list or comma-separated string for --solution and enforce 7 genes."""
    s = arg.strip()
    try:
        arr = np.array(json.loads(s), dtype=float) if s.startswith("[") else np.array([float(x) for x in s.split(",")], dtype=float)
    except Exception as e:
        raise ValueError(f"Could not parse --solution: {e}")
    if arr.shape[0] != 7:
        raise ValueError(f"--solution must contain exactly 7 values (got {arr.shape[0]}).")
    return arr


def dataset_config_path_from_name(name: str) -> str:
    """Mapping from dataset name to its config path under configs/."""
    mapping = {
        "KPI": "configs/kpi_config.yaml",
        "MSL": "configs/msl_config.yaml",
        "SMAP": "configs/smap_config.yaml",
        "SMD": "configs/smd_config.yaml",
        "SWAT": "configs/swat_config.yaml",
        "UCR": "configs/ucr_config.yaml",
        "WADI": "configs/wadi_config.yaml",
        "YahooA1": "configs/yahoo_config.yaml",
        "NAB": "configs/nab_config.yaml",
    }
    return mapping.get(name, f"configs/{name.lower()}_config.yaml")


def select_dataloader(config: dict):
    """Same mapping logic as in main.py."""
    dataset_name = config["data_params"].get("dataset_name", "")
    dataset_key = str(dataset_name).strip().lower()
    dataloader_switch = {
        "yahooa1": YahooA1DataLoader,
        "kpi": KPIDataLoader,
        "msl": SMABandMSDataLoader,
        "smap": SMABandMSDataLoader,
        "smd": SMDDataLoader,
        "ucr": UCRDataLoader,
        "swat": SWATDataLoader,
        "wadi": WADIDataLoader,
        "nab": NABDataLoader,
        "metropt": MetroPTDataLoader,
    }
    DataLoaderClass = dataloader_switch.get(dataset_key)
    if DataLoaderClass is None:
        raise ValueError(
            f"Unsupported dataset name: {dataset_name!r}. "
            f"Expected one of: {sorted(dataloader_switch.keys())}"
        )
    return DataLoaderClass(**config["data_params"])


def find_repo_root(start_dir: Path) -> Path | None:
    """
    Heuristically find project root by walking up until we see both:
    - a 'nianetvae' package directory, and
    - a 'configs' directory.
    """
    cur = start_dir.resolve()
    while True:
        if (cur / "nianetvae").is_dir() and (cur / "configs").is_dir():
            return cur
        if cur.parent == cur:
            return None
        cur = cur.parent


# ============================================================
# Manual layer override (for legacy architectures)
# ============================================================

ALLOWED_LAYERS = {
    "RNN": nn.RNN,
    "LSTM": nn.LSTM,
    "GRU": nn.GRU,
    "Linear": nn.Linear,
    "Dropout": nn.Dropout,
    "ReLU": nn.ReLU,
    "ELU": nn.ELU,
    "LeakyReLU": nn.LeakyReLU,
    "GELU": nn.GELU,
    "Tanh": nn.Tanh,
    "BatchNorm1d": nn.BatchNorm1d,
}

def _parse_one_call(call_src: str):
    """Parse 'Linear(11, 11, bias=True)' safely into (name, args, kwargs)."""
    node = ast.parse(call_src.strip(), mode="eval").body
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        raise ValueError(f"Unsupported layer syntax: {call_src}")
    name = node.func.id
    args = [ast.literal_eval(a) for a in node.args]
    kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords}
    return name, args, kwargs

def _build_layer(name: str, args, kwargs):
    if name not in ALLOWED_LAYERS:
        raise ValueError(f"Layer '{name}' not in allowed set: {list(ALLOWED_LAYERS)}")
    return ALLOWED_LAYERS[name](*args, **kwargs)

def _read_text_or_file(s: str) -> str:
    """If s points to a file, read it; else return s as the spec text."""
    if s and os.path.exists(s) and os.path.isfile(s):
        with open(s, "r", encoding="utf-8") as f:
            return f.read()
    return s or ""

def parse_layers_spec(spec_text: str) -> nn.ModuleList:
    """
    Accepts lines like:
      RNN(1, 11, batch_first=True)
      2 x Linear(11, 11)
      Linear(in_features=197, out_features=1, bias=True)
    Or a pasted ModuleList repr:
      (0): RNN(1, 11, batch_first=True)
      (1-2): 2 x Linear(in_features=11, out_features=11, bias=True)
    Returns nn.ModuleList([...]).
    """
    # strip ModuleList(...) wrappers and indices if pasted from repr
    spec = []
    for raw in spec_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("ModuleList"):
            continue
        line = re.sub(r"^\(\d+(?:-\d+)?\):\s*", "", line)
        if line:
            spec.append(line)

    layers = []
    for line in spec:
        m = re.match(r"^(\d+)\s*[xX]\s*(.+)$", line)
        repeat = 1
        inner = line
        if m:
            repeat = int(m.group(1))
            inner = m.group(2).strip()
        name, args, kwargs = _parse_one_call(inner)
        for _ in range(repeat):
            layers.append(_build_layer(name, args, kwargs))
    return nn.ModuleList(layers)

def validate_first_last_dims(encoder: nn.ModuleList, decoder: nn.ModuleList, n_features: int):
    """Lightweight shape checks for first encoder and last decoder layers."""
    if isinstance(encoder[0], (nn.RNN, nn.GRU, nn.LSTM)):
        assert encoder[0].input_size == n_features, \
            f"Encoder first RNN input_size={encoder[0].input_size} != n_features={n_features}"
    elif isinstance(encoder[0], nn.Linear):
        assert encoder[0].in_features == n_features, \
            f"Encoder first Linear in_features={encoder[0].in_features} != n_features={n_features}"
    last = decoder[-1]
    if isinstance(last, nn.Linear):
        assert last.out_features == n_features, \
            f"Decoder last Linear out_features={last.out_features} != n_features={n_features}"


# ============================================================
# Visualization utilities (extensible: add more viz_* funcs)
# ============================================================

def find_metrics_csv(run_dir: Path) -> Path:
    # Typical PL CSVLogger path: <run_dir>/pl_logs/version_*/metrics.csv
    candidates = sorted(run_dir.glob("pl_logs/version_*/metrics.csv"))
    if not candidates:
        raise FileNotFoundError(f"No metrics.csv found under {run_dir}/pl_logs/version_*/")
    return candidates[-1]  # latest version

def pick_loss_series(df: pd.DataFrame, prefer_epoch=True):
    # training: prefer epoch-aggregated if available
    if prefer_epoch and "train_loss_epoch" in df.columns:
        tr = df[["epoch", "train_loss_epoch"]].dropna().drop_duplicates(subset=["epoch"], keep="last")
        tr.rename(columns={"train_loss_epoch": "train_loss"}, inplace=True)
    elif "train_loss" in df.columns:
        tr = df[["epoch", "train_loss"]].dropna().groupby("epoch", as_index=False).mean()
    else:
        tr = pd.DataFrame(columns=["epoch", "train_loss"])
    # validation
    if "val_loss" in df.columns:
        va = df[["epoch", "val_loss"]].dropna().drop_duplicates(subset=["epoch"], keep="last")
    elif "val_loss_epoch" in df.columns:
        va = df[["epoch", "val_loss_epoch"]].dropna().drop_duplicates(subset=["epoch"], keep="last")
        va.rename(columns={"val_loss_epoch": "val_loss"}, inplace=True)
    else:
        va = pd.DataFrame(columns=["epoch", "val_loss"])
    return tr, va

def viz_training_validation_loss(run_dir: Path, out: Path | None = None, show: bool = False) -> Path:
    run_dir = Path(run_dir).resolve()
    csv_path = find_metrics_csv(run_dir)
    df = pd.read_csv(csv_path)

    tr, va = pick_loss_series(df)

    plt.figure()
    if not tr.empty:
        plt.plot(tr["epoch"], tr["train_loss"], label="Training Loss")
    if not va.empty:
        plt.plot(va["epoch"], va["val_loss"], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()

    out_path = out if out else (run_dir / "training_validation_loss.png")
    plt.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close()
    print(f"[VIZ] Saved: {out_path}")
    return out_path


# ============================================================
# Main training/eval pipeline
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Recreate & evaluate a single NiaNetVAE model, then visualize.")
    # core
    parser.add_argument("--solution", required=True, help="7-gene vector in [0,1] (JSON list or comma-separated).")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g., SWAT, KPI, YahooA1, ...).")
    parser.add_argument("--config", default="configs/main_config.yaml", help="Path to main_config.yaml.")
    parser.add_argument("--dataset-config", default=None, help="(Optional) Explicit dataset config path.")
    parser.add_argument("--outdir", default="logs/manual_replays", help="Base directory to save run artifacts.")
    parser.add_argument("--alg-name", default="manual", help="Name to tag this run with (for logs only).")
    parser.add_argument("--no-chdir", action="store_true", help="Do not auto-chdir to repo root.")
    # manual (legacy) override
    parser.add_argument("--force-manual", action="store_true",
                        help="Force manual encoder/decoder override (legacy solutions).")
    parser.add_argument("--manual-encoder", default=None,
                        help="Manual encoder spec (file path or inline text).")
    parser.add_argument("--manual-decoder", default=None,
                        help="Manual decoder spec (file path or inline text).")
    # viz flags
    parser.add_argument("--plot-loss", action="store_true",
                        help="After training/testing, save Training/Validation loss plot from CSV logs.")
    parser.add_argument("--show-plots", action="store_true",
                        help="Display plots interactively (in addition to saving).")
    args = parser.parse_args()

    # ---------- Auto-set working dir to repo root ----------
    if not args.no_chdir:
        repo = find_repo_root(Path.cwd())
        if repo:
            os.chdir(repo)
        else:
            print("[WARN] Could not locate repo root automatically. Continuing with current CWD.")

    solution = parse_solution(args.solution)
    dataset_name = args.dataset

    # ---------- Load main config ----------
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    # ---------- Resolve dataset config path ----------
    if args.dataset_config:
        dataset_cfg_path = args.dataset_config
    else:
        mapped = dataset_config_path_from_name(dataset_name)
        dataset_cfg_path = mapped if os.path.exists(mapped) else config.get("dataset", {}).get("config_file")
        if not dataset_cfg_path or not os.path.exists(dataset_cfg_path):
            raise FileNotFoundError(f"Dataset config not found. Tried mapped '{mapped}' and main_config dataset.config_file.")

    with open(dataset_cfg_path, "r") as f:
        dataset_config = yaml.load(f, Loader=yaml.Loader)

    # Force the dataset name to match the user request
    dataset_config.setdefault("data_params", {})
    dataset_config["data_params"]["dataset_name"] = dataset_name

    # ---------- Merge dataset config into main config (shallow merge like main.py) ----------
    config.update(dataset_config)

    # Merge shared data loader parameters like main.py
    shared_data_loader_params = config.get("data_loader_params", {})
    config.setdefault("data_params", {})
    config["data_params"].update(shared_data_loader_params)

    # Allow dataset configs to control anomaly-metrics computation without overriding exp_params.
    if "compute_anomaly_metrics" in config.get("data_params", {}):
        config.setdefault("exp_params", {})
        compute_flag = config["data_params"].get("compute_anomaly_metrics")
        config["exp_params"]["compute_anomaly_metrics"] = bool(compute_flag)
        config["data_params"].pop("compute_anomaly_metrics", None)

    # ---------- Validate/echo key paths ----------
    data_path = config["data_params"].get("data_path")
    print(f"[INFO] Using dataset config: {dataset_cfg_path}")
    print(f"[INFO] dataset_name: {dataset_name}")
    print(f"[INFO] data_path: {data_path} (exists: {os.path.isdir(data_path) if data_path else 'N/A'})")

    # ---------- Logging / run directory ----------
    run_uuid = uuid.uuid4().hex
    save_dir = os.path.join(config["logging_params"]["save_dir"], "manual", run_uuid)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    config["logging_params"]["save_dir"] = save_dir  # keep all artifacts under this run folder

    Log.enable(config["logging_params"])
    Log.info(f"Recreate run UUID: {run_uuid}")
    Log.info(f"Dataset: {dataset_name}")
    Log.info(f"CUDA available: {'Yes' if torch.cuda.is_available() else 'No'}")

    # ---------- Seeding ----------
    seed_everything(config["exp_params"]["manual_seed"], True)

    # ---------- DataModule ----------
    datamodule = select_dataloader(config)
    datamodule.setup()

    # ---------- Model from solution (with optional manual fallback) ----------
    use_manual = args.force_manual or (args.manual_encoder and args.manual_decoder)
    if use_manual:
        enc_text = _read_text_or_file(args.manual_encoder)
        dec_text = _read_text_or_file(args.manual_decoder)
        manual_encoder_layers = parse_layers_spec(enc_text)
        manual_decoder_layers = parse_layers_spec(dec_text)
        n_features = config["data_params"]["n_features"]
        validate_first_last_dims(manual_encoder_layers, manual_decoder_layers, n_features)
        model = RNNVAE(
            solution,
            manual_override=True,
            manual_encoder_layers=manual_encoder_layers,
            manual_decoder_layers=manual_decoder_layers,
            **config
        )
    else:
        model = RNNVAE(solution, **config)
        if not getattr(model, "is_valid", True):
            Log.error("The provided solution maps to an invalid model configuration. "
                      "Re-run with --force-manual and provide --manual-encoder/--manual-decoder.")
            print(json.dumps({"status": "error", "reason": "invalid_model_configuration"}, indent=2))
            return

    # Output path for this exact model/hash
    model_path = os.path.join(save_dir, f"{dataset_name}_{args.alg_name}_{model.get_hash()[:12]}")
    Path(model_path).mkdir(parents=True, exist_ok=True)
    config["logging_params"]["model_path"] = model_path

    # ---------- Lightning Experiment ----------
    experiment = RNNVAExperiment(model, model_path, dataset_name, **config)

    loggers = [
        CSVLogger(save_dir=model_path, name="pl_logs"),
        TensorBoardLogger(save_dir=model_path, name="tb")
    ]

    trainer = Trainer(
        enable_progress_bar=True,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        default_root_dir=model_path,
        log_every_n_steps=50,
        logger=loggers,
        **config["trainer_params"],
    )

    # ---------- Train & Test ----------
    start_time = datetime.now()
    Log.info(f"Training start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    trainer.fit(experiment, datamodule=datamodule)
    Log.info("Training finished.")

    Log.info("Testing start.")
    trainer.test(experiment, datamodule=datamodule)
    end_time = datetime.now()
    Log.info(f"Test end: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # ---------- Collect metrics ----------
    recon_metrics = experiment.metrics.compute() if experiment.metrics is not None else {}
    anomaly_metrics = getattr(experiment, "anomaly_metrics", None)

    # ---------- Persist artifacts ----------
    torch.save(model.state_dict(), os.path.join(model_path, "model.pt"))
    with open(os.path.join(model_path, "effective_config.yaml"), "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    summary = {
        "run_uuid": run_uuid,
        "dataset": dataset_name,
        "model_hash": model.get_hash(),
        "solution": solution.tolist(),
        "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
        "reconstruction_metrics": recon_metrics,
        "anomaly_metrics": anomaly_metrics if isinstance(anomaly_metrics, dict) else getattr(experiment, "anomaly_metrics", {}),
        "artifacts": {
            "model_pt": os.path.join(model_path, "model.pt"),
            "config_yaml": os.path.join(model_path, "effective_config.yaml"),
            "log_dir": model_path,
        },
    }
    with open(os.path.join(model_path, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))

    # ---------- Visualizations (extensible) ----------
    if args.plot_loss:
        try:
            viz_training_validation_loss(Path(model_path), show=args.show_plots)
        except Exception as e:
            print(f"[VIZ] Training/validation loss plot failed: {e}")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    main()

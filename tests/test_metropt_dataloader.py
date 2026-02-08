
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from log import Log
from nianetvae.dataloaders.metropt_dataloader import MetroPTDataLoader


def _ensure_test_logger(tmp_path: Path) -> None:
    if hasattr(Log, "logger"):
        return
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    Log.enable(
        {
            "name": f"pytest-{uuid.uuid4().hex}",
            "logger_file": "test.log",
            "save_dir": str(logs_dir) + "/",
        }
    )


def _write_synth_metropt_csv(tmp_path: Path) -> Path:
    start = pd.Timestamp("2020-04-11 00:00:00")
    end = pd.Timestamp("2020-04-18 00:00:00")
    ts = pd.date_range(start, end, freq="30min")
    rng = np.random.RandomState(0)
    data = rng.randn(len(ts), 3).astype(np.float32)
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "TP2": data[:, 0],
            "TP3": data[:, 1],
            "H1": data[:, 2],
        }
    )
    path = tmp_path / "MetroPT3.csv"
    df.to_csv(path, index=False)
    return path


def test_metropt_dataloader_single_smoke(tmp_path: Path) -> None:
    _ensure_test_logger(tmp_path)
    csv_path = _write_synth_metropt_csv(tmp_path)

    dm = MetroPTDataLoader(
        dataset_name="MetroPT",
        data_path=str(csv_path),
        batch_size=16,
        seq_len=10,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
        val_size=20,
        data_percentage=100,
        rolling_window="2h",
        train_minutes=12 * 60,
        post_train_minutes=12 * 60,
        pre_maint_minutes=120,
        regime="single",
        cycle_id=1,
        stride=2,
        timestamp_col="timestamp",
        drop_unnamed_index=True,
        train_phases=(0, 1),
        test_phases=(0,),
    )
    dm.setup()

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    batch = next(iter(train_loader))
    assert batch["signal"].ndim == 3
    assert batch["signal"].shape[1] == 10
    assert batch["signal"].shape[2] == dm.n_features
    assert int(batch["target"].sum().item()) == 0
    assert "ts_id" in batch

    assert next(iter(val_loader))["signal"].shape[1:] == (10, dm.n_features)
    assert next(iter(test_loader))["signal"].shape[1:] == (10, dm.n_features)


def test_metropt_dataloader_per_maint_cycle_1_splits(tmp_path: Path) -> None:
    _ensure_test_logger(tmp_path)
    csv_path = _write_synth_metropt_csv(tmp_path)

    dm = MetroPTDataLoader(
        dataset_name="MetroPT",
        data_path=str(csv_path),
        batch_size=16,
        seq_len=10,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
        val_size=20,
        data_percentage=100,
        rolling_window="2h",
        train_minutes=12 * 60,
        post_train_minutes=12 * 60,
        pre_maint_minutes=120,
        regime="per_maint",
        cycle_id=1,
        stride=2,
        timestamp_col="timestamp",
        drop_unnamed_index=True,
        train_phases=(0, 1),
        test_phases=(0,),
    )
    dm.setup()

    split = dm.split_info
    assert split["regime"] == "per_maint"
    assert split["cycle_id"] == 1

    # For cycle_id=1, post_train_end = end(#1) + post_train_minutes, and test starts there.
    assert pd.to_datetime(split["test_start"]) == pd.to_datetime(split["post_train_end"])

    # Test interval ends at start of W2 (#2) by definition.
    assert pd.to_datetime(split["test_end"]) == pd.Timestamp("2020-04-18 00:00:00")

    # Training mask should split into at least baseline + post-train segments.
    assert int(split.get("train_segments", 0)) >= 2


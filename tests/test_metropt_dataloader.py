
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from torch.utils.data import RandomSampler

from log import Log
from nianetvae.dataloaders.metropt_dataloader import (
    MetroPTDataLoader,
    MetroPTSegmentedSequenceDataset,
    build_feature_hash,
)


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


def _write_synth_metropt_csv_long(tmp_path: Path) -> Path:
    start = pd.Timestamp("2020-04-11 00:00:00")
    end = pd.Timestamp("2020-07-20 00:00:00")
    ts = pd.date_range(start, end, freq="30min")
    rng = np.random.RandomState(7)
    data = rng.randn(len(ts), 3).astype(np.float32)
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "TP2": data[:, 0],
            "TP3": data[:, 1],
            "H1": data[:, 2],
        }
    )
    path = tmp_path / "MetroPT3_long.csv"
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
    )
    dm.setup()

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    batch = next(iter(train_loader))
    assert dm.base_feature_names == ["TP2", "TP3", "H1"]
    assert len(dm.rolling_feature_names) == dm.n_features
    assert dm.feature_hash == build_feature_hash(dm.rolling_feature_names)
    assert dm.scaler is not None
    assert int(dm.scaler.n_features_in_) == int(dm.n_features)
    assert dm.train_segment_metadata
    assert dm.test_segment_metadata
    assert dm.split_info["rolling_feature_names"] == dm.rolling_feature_names
    assert dm.split_info["feature_hash"] == dm.feature_hash
    assert batch["signal"].ndim == 3
    assert batch["signal"].shape[1] == 10
    assert batch["signal"].shape[2] == dm.n_features
    assert int(batch["target"].sum().item()) == 0
    assert int(batch["operation_phase"].sum().item()) == 0
    assert "ts_id" in batch

    assert next(iter(val_loader))["signal"].shape[1:] == (10, dm.n_features)
    test_batch = next(iter(test_loader))
    assert test_batch["signal"].shape[1:] == (10, dm.n_features)
    unique_targets = set(test_batch["target"].detach().cpu().numpy().astype(int).tolist())
    unique_phases = set(test_batch["operation_phase"].detach().cpu().numpy().astype(int).tolist())
    assert unique_targets.issubset({0, 1})
    assert unique_phases.issubset({0, 1})


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


def test_metropt_finetune_policy_trains_on_local_windows_with_balanced_baseline_replay(
    tmp_path: Path,
) -> None:
    _ensure_test_logger(tmp_path)
    csv_path = _write_synth_metropt_csv(tmp_path)

    dm = MetroPTDataLoader(
        dataset_name="MetroPT",
        data_path=str(csv_path),
        batch_size=4,
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
        test_phases=(0, 1),
        workflow_mode="per_maint_finetune_search",
        finetune_data_policy={
            "enabled": True,
            "baseline_replay_fraction": 0.5,
            "local_validation_fraction": 0.2,
            "validation_embargo": True,
            "shuffle_train": True,
            "random_seed": 123,
        },
    )
    dm.setup()

    policy = dm.split_info["fine_tune_data_policy"]
    assert policy["mode"] == "local_train_with_baseline_replay"
    assert int(policy["local_train_windows"]) > 0
    assert int(policy["local_validation_windows"]) > 0
    assert int(policy["local_validation_embargo_windows"]) > 0
    assert int(policy["baseline_replay_windows"]) == int(policy["local_train_windows"])
    assert float(policy["effective_baseline_replay_fraction"]) == pytest.approx(0.5)
    assert len(dm.train_dataset) == (
        int(policy["baseline_replay_windows"]) + int(policy["local_train_windows"])
    )
    assert len(dm.val_dataset) == int(policy["local_validation_windows"])
    assert isinstance(dm.train_dataloader().sampler, RandomSampler)


def test_metropt_finetune_policy_uses_all_short_local_windows_without_validation(
    tmp_path: Path,
) -> None:
    _ensure_test_logger(tmp_path)
    dm = MetroPTDataLoader(
        dataset_name="MetroPT",
        data_path=str(tmp_path / "unused.csv"),
        batch_size=4,
        seq_len=200,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
        val_size=20,
        data_percentage=100,
        regime="per_maint",
        cycle_id=10,
        stride=1,
        workflow_mode="per_maint_finetune_search",
        finetune_data_policy={
            "enabled": True,
            "baseline_replay_fraction": 0.5,
            "local_validation_fraction": 0.2,
            "validation_embargo": True,
            "short_local_fallback": "train_all_fixed_min_epochs",
            "shuffle_train": True,
            "random_seed": 123,
        },
    )

    # Reproduce the production cycle-10 shape from HPC: 295 local rows yield
    # 96 windows at seq_len=200 and stride=1, fewer than the 199-window embargo.
    baseline = np.zeros((600, 3), dtype=np.float32)
    local = np.zeros((295, 3), dtype=np.float32)
    dm._build_finetune_datasets([baseline, local])

    policy = dm.split_info["fine_tune_data_policy"]
    assert policy["short_local_fallback_applied"] is True
    assert policy["validation_strategy"] == "disabled_short_local_fallback"
    assert policy["early_stopping_eligible"] is False
    assert policy["local_total_windows"] == 96
    assert policy["local_train_windows"] == 96
    assert policy["local_validation_windows"] == 0
    assert policy["requested_local_validation_embargo_windows"] == 199
    assert policy["local_validation_embargo_windows"] == 0
    assert policy["baseline_replay_windows"] == 96
    assert len(dm.train_dataset) == 192
    assert dm.val_dataset is None


def test_metropt_finetune_policy_allows_unsupervised_cycle_without_positive_test_windows(
    tmp_path: Path,
) -> None:
    _ensure_test_logger(tmp_path)
    csv_path = _write_synth_metropt_csv_long(tmp_path)

    dm = MetroPTDataLoader(
        dataset_name="MetroPT",
        data_path=str(csv_path),
        batch_size=4,
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
        cycle_id=21,
        stride=2,
        timestamp_col="timestamp",
        drop_unnamed_index=True,
        train_phases=(0, 1),
        test_phases=(0, 1),
        workflow_mode="per_maint_finetune_search",
        finetune_data_policy={
            "enabled": True,
            "baseline_replay_fraction": 0.5,
            "local_validation_fraction": 0.2,
            "validation_embargo": True,
            "shuffle_train": True,
            "random_seed": 123,
        },
    )
    dm.setup()

    assert dm.split_info["test_informative_for_pdm_objective"] is False
    assert int(dm.split_info["fine_tune_data_policy"]["local_train_windows"]) > 0


def test_metropt_dataloader_per_maint_cycle_0_uses_phase01_test_filter_defaults(tmp_path: Path) -> None:
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
        cycle_id=0,
        stride=2,
        timestamp_col="timestamp",
        drop_unnamed_index=True,
        train_phases=(0, 1),
    )
    dm.setup()

    split = dm.split_info
    assert split["regime"] == "per_maint"
    assert split["cycle_id"] == 0
    assert split["maintenance_id"] == "pre_W1"
    assert split["test_phases"] == [0, 1]

    assert pd.to_datetime(split["test_start"]) == pd.to_datetime(split["baseline_end"])
    assert pd.to_datetime(split["test_end"]) == pd.Timestamp("2020-04-12 11:50:00")
    assert int(split["test_rows"]) > 0
    assert int(split["test_phase2_rows"]) == 0
    assert int(split["test_label_pos_windows"]) >= 0
    assert int(split["test_label_neg_windows"]) > 0


def test_metropt_dataloader_per_maint_raises_on_zero_positive_windows_when_phase1_expected(tmp_path: Path) -> None:
    _ensure_test_logger(tmp_path)
    csv_path = _write_synth_metropt_csv_long(tmp_path)

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
        cycle_id=21,
        stride=2,
        timestamp_col="timestamp",
        drop_unnamed_index=True,
        train_phases=(0, 1),
        test_phases=(0, 1),
    )

    with pytest.raises(ValueError, match="zero positive windows after phase filtering"):
        dm.setup()


def test_metropt_segmented_dataset_uses_end_anchor_phase_labels() -> None:
    signal_segment = np.arange(8, dtype=np.float32).reshape(-1, 1)
    phase_segment = np.array([0, 0, 1, 1, 0, 1, 0, 0], dtype=np.int8)

    ds = MetroPTSegmentedSequenceDataset(
        segments=[signal_segment],
        phase_segments=[phase_segment],
        seq_len=3,
        stride=2,
    )

    assert len(ds) == 3
    sample0 = ds[0]
    sample1 = ds[1]
    sample2 = ds[2]

    # Anchors are at indices 2, 4, 6 for seq_len=3 and stride=2.
    assert int(sample0["operation_phase"]) == 1
    assert int(sample0["target"]) == 1
    assert int(sample1["operation_phase"]) == 0
    assert int(sample1["target"]) == 0
    assert int(sample2["operation_phase"]) == 0
    assert int(sample2["target"]) == 0

    assert int(ds.window_positive_count) == 1
    assert int(ds.window_negative_count) == 2

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nianetvae.config import PreprocessingConfig, StudyConfig, load_study_config
from nianetvae.dataloaders.preprocessing import FrozenPreprocessor


def test_controlled_config_rejects_model_specific_preprocessing() -> None:
    config = replace(StudyConfig(), preprocessing=PreprocessingConfig(policy="standard_scaler_v1"))
    with pytest.raises(ValueError, match="binary_passthrough_v1"):
        config.validate()


def test_search_termination_budget_does_not_change_study_identity() -> None:
    config = StudyConfig().validate()
    extended = replace(
        config,
        search=replace(config.search, max_generations=600, max_time="90:00:00"),
    ).validate()
    assert extended.fingerprint() == config.fingerprint()
    assert extended.resolved_fingerprint() != config.resolved_fingerprint()


def test_binary_derived_features_are_passthrough_and_contract_is_frozen() -> None:
    frame = pd.DataFrame(
        {
            "TP2__mean": [1.0, 2.0, 3.0],
            "COMP__mean": [0.0, 1.0, 0.0],
            "COMP__std": [0.0, 0.5, 0.0],
        }
    )
    preprocessor = FrozenPreprocessor.fit(frame, binary_feature_names=("COMP",))
    transformed = preprocessor.transform(frame)
    assert np.allclose(transformed["COMP__mean"], frame["COMP__mean"])
    assert np.allclose(transformed["COMP__std"], frame["COMP__std"])
    assert np.isclose(float(transformed["TP2__mean"].mean()), 0.0)
    assert len(preprocessor.fingerprint) == 64

    with pytest.raises(ValueError, match="Feature names/order"):
        preprocessor.transform(frame[["COMP__mean", "TP2__mean", "COMP__std"]])


REPOSITORY = Path(__file__).resolve().parents[1]
V5_LADDER = (
    ("configs/search_ladder/metropt_study_v5_gen025.yaml", 25),
    ("configs/search_ladder/metropt_study_v5_gen050.yaml", 50),
    ("configs/search_ladder/metropt_study_v5_gen075.yaml", 75),
    ("configs/metropt_study_v5.yaml", 100),
)


def test_v5_search_ladder_shares_one_study_contract() -> None:
    """Ladder steps must differ only in the replaceable search budget.

    Each step resumes the previous NSGA-III checkpoint, which the engine accepts
    only while the study contract is unchanged. A step that drifted in any
    controlled constant would be rejected mid-run, days into the search.
    """
    configs = [
        (load_study_config(REPOSITORY / relative), relative, generations)
        for relative, generations in V5_LADDER
    ]

    reference, reference_relative, _ = configs[0]
    for config, relative, generations in configs:
        assert config.artifacts.study_id == "metropt_controlled_v5", relative
        assert config.search.enabled, relative
        assert config.search.max_generations == generations, relative
        assert config.fingerprint() == reference.fingerprint(), (
            f"{relative} does not share the study contract of {reference_relative}"
        )

    budgets = [config.search.max_generations for config, _relative, _gens in configs]
    assert budgets == sorted(set(budgets)), "ladder budgets must strictly increase"

    resolved = {config.resolved_fingerprint() for config, _relative, _gens in configs}
    assert len(resolved) == len(configs), "each ladder step must record a distinct budget"

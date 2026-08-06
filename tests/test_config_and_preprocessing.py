from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from nianetvae.config import PreprocessingConfig, StudyConfig
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

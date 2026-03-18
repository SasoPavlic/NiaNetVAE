from __future__ import annotations

import numpy as np

from nianetvae.models.rnn_vae import RNNVAE


def _model_kwargs(n_features: int = 90, seq_len: int = 200) -> dict:
    return {
        "data_params": {
            "n_features": n_features,
            "seq_len": seq_len,
            "batch_size": 16,
        }
    }


def test_default_constructor_uses_solution_mapping():
    solution = np.array([0.1, 0.01, 0.01, 0.01, 0.01, 0.2, 0.3], dtype=float)
    model = RNNVAE(solution, **_model_kwargs())
    assert isinstance(model.mapping_context, dict)


def test_mapping_builds_monotone_valid_architecture():
    solution = np.array([0.25, 0.8, 0.5, 0.75, 0.4, 0.2, 0.3], dtype=float)
    model = RNNVAE(solution, **_model_kwargs(n_features=90, seq_len=200))

    assert model.is_valid
    assert model.bottleneck_size == model.hidden_dims[-1]
    assert len(model.hidden_dims) == model.encoder_num_layers
    assert len(model.decoder_hidden_dims) == model.decoder_num_layers
    assert 1 <= model.bottleneck_size < 90

    encoder_path = [90] + list(model.hidden_dims)
    assert all(left > right for left, right in zip(encoder_path[:-1], encoder_path[1:]))

    decoder_path = [model.bottleneck_size] + list(model.decoder_hidden_dims)
    assert all(left < right for left, right in zip(decoder_path[:-1], decoder_path[1:]))


def test_invalid_rate_is_zero_on_random_solutions():
    rng = np.random.default_rng(42)
    solutions = rng.uniform(0.0, 1.0, size=(200, 7))

    invalid_count = 0
    for solution in solutions:
        if not RNNVAE(solution, **_model_kwargs(n_features=90, seq_len=200)).is_valid:
            invalid_count += 1

    assert invalid_count == 0


def test_dense_ratio_grid_and_half_up_rounding():
    # Gene y3 is mapped from a dense ratio grid [0.04..0.50 step 0.01].
    # Pick index 31 => ratio 0.35 and verify 90 * 0.35 rounds to 32 (half-up).
    y3_gene = (31.1 / 47.0)
    solution = np.array([0.2, 0.3, y3_gene, 0.4, 0.5, 0.1, 0.2], dtype=float)
    model = RNNVAE(solution, **_model_kwargs(n_features=90, seq_len=200))

    assert model.is_valid
    assert model.mapping_context["bottleneck_ratio"] == 0.35
    assert model.bottleneck_size == 32

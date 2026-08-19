"""Pure six-gene to ArchitectureSpec decoding."""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

import numpy as np

from ..contracts import ArchitectureSpec

GENOME_DIMENSION = 6
MAPPING_VERSION = "metropt_six_gene_v1"


def _option(gene: float, options):
    index = max(0, min(int(float(gene) * len(options)), len(options) - 1))
    return options[index]


def _round_half_up(value: float) -> int:
    return int(Decimal(str(value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _allocate_steps(total_delta: int, count: int, curvature: float) -> list[int]:
    if total_delta < count or count < 1:
        raise ValueError("Architecture does not have enough width for strict monotone layers.")
    positions = np.arange(1, count + 1, dtype=float)
    raw = np.power(positions, float(curvature))
    raw = raw / raw.sum() * total_delta
    steps = np.maximum(np.floor(raw).astype(int), 1)
    while int(steps.sum()) > total_delta:
        changed = False
        for index in np.argsort(-steps):
            if steps[index] > 1 and int(steps.sum()) > total_delta:
                steps[index] -= 1
                changed = True
        if not changed:
            raise ValueError("Could not allocate strict architecture steps.")
    fractional = raw - np.floor(raw)
    order = np.argsort(-fractional)
    remainder = total_delta - int(steps.sum())
    for offset in range(remainder):
        steps[order[offset % count]] += 1
    return [int(value) for value in steps]


def _monotone_dims(start: int, end: int, depth: int, curvature: float) -> tuple[int, ...]:
    decreasing = start > end
    steps = _allocate_steps(abs(start - end), depth, curvature)
    current = int(start)
    output: list[int] = []
    for step in steps:
        current = current - step if decreasing else current + step
        output.append(current)
    return tuple(output)


def decode_genome(
    genome,
    *,
    input_dim: int,
    sequence_length: int,
    source: str = "nsga3",
) -> ArchitectureSpec:
    genes = tuple(float(value) for value in np.asarray(genome, dtype=float).reshape(-1))
    if len(genes) != GENOME_DIMENSION:
        raise ValueError(f"Expected {GENOME_DIMENSION} genes, received {len(genes)}.")
    if any(not np.isfinite(value) or value < 0.0 or value > 1.0 for value in genes):
        raise ValueError("Every architecture gene must be finite and in [0,1].")
    cell = _option(genes[0], ("LSTM", "GRU", "RNN_TANH"))
    encoder_depth = int(_option(genes[1], (1, 2, 3, 4, 5)))
    ratio = float(_option(genes[2], tuple(round(value / 100.0, 2) for value in range(4, 51))))
    encoder_curvature = float(_option(genes[3], (0.7, 1.0, 1.3, 1.8)))
    decoder_offset = int(_option(genes[4], (-1, 0, 1, 2)))
    activation = _option(
        genes[5],
        ("elu", "relu", "leaky_relu", "rrelu", "selu", "celu", "gelu", "tanh"),
    )
    latent_dim = max(1, min(int(input_dim) - 1, _round_half_up(int(input_dim) * ratio)))
    maximum_depth = max(1, int(input_dim) - latent_dim)
    encoder_depth = max(1, min(encoder_depth, maximum_depth))
    decoder_depth = max(1, min(encoder_depth + decoder_offset, maximum_depth))
    encoder_dims = _monotone_dims(input_dim, latent_dim, encoder_depth, encoder_curvature)
    decoder_curvature = max(0.4, 2.0 - encoder_curvature)
    decoder_dims = _monotone_dims(latent_dim, input_dim, decoder_depth, decoder_curvature)
    return ArchitectureSpec(
        model_kind="recurrent_vae",
        source=source,  # type: ignore[arg-type]
        input_dim=int(input_dim),
        sequence_length=int(sequence_length),
        recurrent_cell=cell,
        encoder_hidden_dims=encoder_dims,
        decoder_hidden_dims=decoder_dims,
        latent_dim=latent_dim,
        activation=activation,
        genome=genes,
        mapping_version=MAPPING_VERSION,
        label=f"nsga3_{cell.lower()}_{len(encoder_dims)}x{len(decoder_dims)}_latent{latent_dim}",
    ).validate()

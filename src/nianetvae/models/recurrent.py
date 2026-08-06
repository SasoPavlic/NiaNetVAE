"""One recurrent autoencoder implementation for handcrafted and searched models."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..contracts import ArchitectureSpec


def _activation(name: str) -> Callable[[Tensor], Tensor]:
    normalized = str(name).strip().lower().replace(" ", "_")
    mapping: dict[str, Callable[[Tensor], Tensor]] = {
        "identity": lambda value: value,
        "elu": F.elu,
        "relu": F.relu,
        "leaky_relu": F.leaky_relu,
        "rrelu": F.rrelu,
        "selu": F.selu,
        "celu": F.celu,
        "gelu": F.gelu,
        "tanh": torch.tanh,
    }
    try:
        return mapping[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported activation={name!r}.") from exc


def _recurrent_layer(cell: str, input_size: int, hidden_size: int) -> nn.Module:
    if cell == "LSTM":
        return nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
    if cell == "GRU":
        return nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True)
    if cell == "RNN_TANH":
        return nn.RNN(input_size, hidden_size, num_layers=1, nonlinearity="tanh", batch_first=True)
    raise ValueError(f"Unsupported recurrent cell={cell!r}.")


class RecurrentAutoencoder(nn.Module):
    def __init__(self, architecture: ArchitectureSpec) -> None:
        super().__init__()
        self.architecture = architecture.validate()
        if architecture.model_kind not in {"recurrent_vae", "recurrent_sae"}:
            raise ValueError("RecurrentAutoencoder requires a recurrent architecture.")
        self.activation = _activation(architecture.activation)
        self.encoder_layers = nn.ModuleList()
        input_size = architecture.input_dim
        for hidden_size in architecture.encoder_hidden_dims:
            self.encoder_layers.append(
                _recurrent_layer(architecture.recurrent_cell or "", input_size, int(hidden_size))
            )
            input_size = int(hidden_size)
        latent_dim = int(architecture.latent_dim or 0)
        if architecture.model_kind == "recurrent_vae":
            self.fc_mu = nn.Linear(input_size, latent_dim)
            self.fc_logvar = nn.Linear(input_size, latent_dim)
            self.fc_latent = None
        else:
            self.fc_mu = None
            self.fc_logvar = None
            self.fc_latent = nn.Linear(input_size, latent_dim)
        self.decoder_layers = nn.ModuleList()
        input_size = latent_dim
        for hidden_size in architecture.decoder_hidden_dims:
            self.decoder_layers.append(
                _recurrent_layer(architecture.recurrent_cell or "", input_size, int(hidden_size))
            )
            input_size = int(hidden_size)
        self.output = nn.Linear(input_size, architecture.input_dim)

    def _encode_hidden(self, signal: Tensor) -> Tensor:
        value = signal
        for layer in self.encoder_layers:
            value, _state = layer(value)
            value = self.activation(value)
        return value[:, -1, :]

    def encode(self, signal: Tensor):
        hidden = self._encode_hidden(signal)
        if self.architecture.model_kind == "recurrent_vae":
            assert self.fc_mu is not None and self.fc_logvar is not None
            return self.fc_mu(hidden), self.fc_logvar(hidden)
        assert self.fc_latent is not None
        return torch.sigmoid(self.fc_latent(hidden))

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, latent: Tensor) -> Tensor:
        value = latent.unsqueeze(1).repeat(1, self.architecture.sequence_length, 1)
        for layer in self.decoder_layers:
            value, _state = layer(value)
            value = self.activation(value)
        return self.output(value)

    def forward(self, signal: Tensor) -> dict[str, Tensor]:
        if self.architecture.model_kind == "recurrent_vae":
            mu, logvar = self.encode(signal)
            latent = self.reparameterize(mu, logvar)
            return {
                "reconstructed": self.decode(latent),
                "latent": latent,
                "mu": mu,
                "logvar": logvar,
            }
        latent = self.encode(signal)
        return {"reconstructed": self.decode(latent), "latent": latent}

    def reconstruct_deterministic(self, signal: Tensor) -> Tensor:
        if self.architecture.model_kind == "recurrent_vae":
            mu, _logvar = self.encode(signal)
            return self.decode(mu)
        return self.decode(self.encode(signal))


def build_recurrent_model(architecture: ArchitectureSpec) -> RecurrentAutoencoder:
    return RecurrentAutoencoder(architecture)


def recurrent_training_loss(
    model: RecurrentAutoencoder,
    batch: Tensor,
    *,
    kl_beta: float,
    sparsity_beta: float,
    sparsity_rho: float,
) -> Tensor:
    output = model(batch)
    reconstruction_loss = F.mse_loss(output["reconstructed"], batch, reduction="mean")
    if model.architecture.model_kind == "recurrent_vae":
        mu = output["mu"]
        logvar = output["logvar"]
        kl = -0.5 * torch.mean(torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        return reconstruction_loss + float(kl_beta) * kl
    latent = output["latent"]
    epsilon = 1e-7
    rho = torch.tensor(float(sparsity_rho), device=latent.device, dtype=latent.dtype)
    rho_hat = torch.clamp(latent.mean(dim=0), epsilon, 1.0 - epsilon)
    sparse_kl = rho * torch.log(rho / rho_hat) + (1.0 - rho) * torch.log(
        (1.0 - rho) / (1.0 - rho_hat)
    )
    return reconstruction_loss + float(sparsity_beta) * sparse_kl.sum()


def recurrent_anomaly_score(model: RecurrentAutoencoder, batch: Tensor) -> Tensor:
    reconstructed = model.reconstruct_deterministic(batch)
    return ((reconstructed - batch) ** 2).mean(dim=(1, 2))

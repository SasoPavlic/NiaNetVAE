import time
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import hashlib

from log import Log
from .base import BaseVAE
from .types_ import *


class RNNVAE(BaseVAE, nn.Module):
    def __init__(self, solution, **kwargs) -> None:
        super(RNNVAE, self).__init__()
        """
        Solution vector dimensionality (length 7):
            y1: recurrent layer family,
            y2..y5: mapping-specific architecture genes,
            y6: activation function,
            y7: optimizer algorithm.
        """

        self.is_valid = True

        # Extract data parameters
        n_features = kwargs['data_params']['n_features']
        seq_len = kwargs['data_params']['seq_len']
        batch_size = kwargs['data_params']['batch_size']
        data_shape = kwargs['data_params'].get('data_shape', None)

        # Determine if univariate or multivariate
        if data_shape is not None:
            if len(data_shape) == 3:
                n_features = data_shape[2]
                seq_len = data_shape[1]
                is_univariate = False
            elif len(data_shape) == 2:
                n_features = 1
                seq_len = data_shape[1]
                is_univariate = True
            else:
                raise ValueError(f"Unsupported data shape: {data_shape}")
        else:
            is_univariate = (n_features == 1)

        self.is_univariate = is_univariate
        self.id = str(int(time.time())).strip()
        self.dataset_shape = [n_features, seq_len]
        self.encoding_layers = nn.ModuleList()
        self.decoding_layers = nn.ModuleList()

        self.n_features = n_features
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.mapping_version = self.resolve_mapping_version(kwargs)
        self.mapping_context = {}
        # Keep defaults for robust hashing/logging even if decode fails early.
        self.encoder_layer_step = 1
        self.decoder_layer_step = 1
        self.encoder_num_layers = 1
        self.decoder_num_layers = 1
        self.bottleneck_size = 1

        # Map solution vector (length 7)
        y1, y2, y3, y4, y5, y6, y7 = solution

        self.layer_type = self.map_layer_type(y1)  # Shared for encoder & decoder
        self.activation = self.map_activation(y6)
        self.optimizer_name = self.map_optimizer(y7, self)

        self.hidden_dims = []
        self.decoder_hidden_dims = []
        decode_ok = self._decode_solution_v2(y2, y3, y4, y5) if self.mapping_version == "v2" else self._decode_solution_v1(y2, y3, y4, y5)
        if not decode_ok:
            self.is_valid = False
            Log.debug(f"Invalid architecture configuration for mapping_version={self.mapping_version}.")
            self.bottleneck_size = None
            self.hidden_dims = []
            self.encoding_layers = None
            self.decoding_layers = None
            self.get_hash()
            return

        if self.mapping_version == "v2":
            # Mapping v2 builds a deterministic decoder profile directly.
            self.generate_autoencoder(
                self.layer_type,
                self.dataset_shape,
                self.hidden_dims,
                symmetrical=False,
                decoder_hidden_dims_override=self.decoder_hidden_dims,
            )
        else:
            self.generate_autoencoder(
                self.layer_type,
                self.dataset_shape,
                self.hidden_dims,
                symmetrical=(
                    self.encoder_layer_step == self.decoder_layer_step and
                    self.encoder_num_layers == self.decoder_num_layers
                )
            )

        if not self.is_valid:
            self.get_hash()
            return

        self.get_hash()
        Log.debug(
            f"MODEL_DECODE hash={self.hash_id} mapping_version={self.mapping_version} layer_type={self.layer_type} "
            f"enc_step={self.encoder_layer_step} enc_layers={self.encoder_num_layers} "
            f"dec_layers={self.decoder_num_layers} dec_step={self.decoder_layer_step} "
            f"activation={self.activation_name} optimizer={self.optimizer_name} "
            f"bottleneck_size={self.bottleneck_size}"
        )

    def get_hash(self):
        hash_parts = [
            self.layer_type,
            str(self.encoder_layer_step),
            str(self.encoder_num_layers),
            str(self.decoder_num_layers),
            str(self.decoder_layer_step),
            str(self.activation_name),
            str(self.optimizer_name),
            str(self.bottleneck_size),
        ]
        # Keep legacy v1 hash representation stable; include mapping payload for v2+.
        if self.mapping_version != "v1":
            hash_parts.extend([
                str(self.mapping_version),
                str(getattr(self, "encoder_hidden_dims", [])),
                str(getattr(self, "decoder_hidden_dims", [])),
            ])
        self.hash_id = hashlib.sha1(
            "".join(hash_parts).encode('utf-8')
        ).hexdigest()
        return self.hash_id

    def encode(self, x: Tensor) -> list:
        for i, layer in enumerate(self.encoding_layers[:-2]):
            if isinstance(layer, (nn.LSTM, nn.GRU, nn.RNN)):
                x, _ = layer(x)
                x = self.activation(x)
            else:
                x = layer(x)
                x = self.activation(x)
        x_last = x[:, -1, :]
        mu = self.encoding_layers[-2](x_last)
        log_var = self.encoding_layers[-1](x_last)
        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        decoder_input = z.unsqueeze(1).repeat(1, self.seq_len, 1)
        x = decoder_input
        for i, layer in enumerate(self.decoding_layers[:-1]):
            if isinstance(layer, (nn.LSTM, nn.GRU, nn.RNN)):
                x, _ = layer(x)
                x = self.activation(x)
            else:
                x = layer(x)
                x = self.activation(x)
        batch_size, seq_len, hidden_dim = x.size()
        x = x.contiguous().view(-1, hidden_dim)
        x = self.decoding_layers[-1](x)
        x = x.view(batch_size, seq_len, self.n_features)
        return x

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, input: dict, **kwargs) -> dict:
        signal = input['signal']
        mu, log_var = self.encode(signal)
        z = self.reparameterize(mu, log_var)
        reconstructed = self.decode(z)
        return {'signal': signal, 'reconstructed': reconstructed, 'mu': mu, 'log_var': log_var}

    def loss_function(self, curr_device: str = 'cuda', **kwargs) -> dict:
        input = kwargs['signal']
        recons = kwargs['reconstructed']
        mu = kwargs['mu']
        log_var = kwargs['log_var']
        kld_weight = kwargs['M_N']
        recons_loss = F.mse_loss(recons, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': -kld_loss.detach()}

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x)['reconstructed']

    def get_layer_object(self, input_size, hidden_size, num_layers, batch_first):
        hidden_size = int(hidden_size) if isinstance(hidden_size, (np.integer, np.int64, float)) else hidden_size
        if self.layer_type == 'LSTM':
            return nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=batch_first)
        elif self.layer_type == 'GRU':
            return nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=batch_first)
        elif self.layer_type == 'RNN_TANH':
            return nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, nonlinearity='tanh', batch_first=batch_first)

    def map_layer_type(self, gene):
        gene = float(gene)
        layer_types = ["LSTM", "GRU", "RNN_TANH"]
        index = int(gene * len(layer_types))
        if index >= len(layer_types):
            index = len(layer_types) - 1
        return layer_types[index]

    def map_activation(self, gene):
        gene = float(gene)
        activation_functions = [F.elu, F.relu, F.leaky_relu, F.rrelu, F.selu, F.celu, F.gelu, torch.tanh]
        activation_names = ["ELU", "ReLU", "Leaky ReLU", "RReLU", "SELU", "CELU", "GELU", "Tanh"]
        index = int(gene * len(activation_functions))
        if index >= len(activation_functions):
            index = len(activation_functions) - 1
        self.activation_name = activation_names[index]
        return activation_functions[index]

    def map_optimizer(self, gene, architecture):
        gene = float(gene)
        optimizer_names = ["Adam", "Adagrad", "SGD", "RAdam", "ASGD", "RPROP"]
        index = int(gene * len(optimizer_names))
        if index >= len(optimizer_names):
            index = len(optimizer_names) - 1
        self.optimizer_name = optimizer_names[index]
        return optimizer_names[index]

    def map_layer_step(self, gene, ref_dim):
        gene = float(gene)
        min_step = 1
        max_step = ref_dim
        layer_step = int(min_step + gene * (max_step - min_step))
        return max(min(layer_step, max_step), min_step)

    def map_num_layers(self, gene, ref_dim):
        gene = float(gene)
        min_layers = 1
        max_layers = ref_dim
        num_layers = int(min_layers + gene * (max_layers - min_layers))
        return max(min(num_layers, max_layers), min_layers)

    def resolve_mapping_version(self, kwargs):
        model_params = kwargs.get("model_params", {}) if isinstance(kwargs, dict) else {}
        version = str(model_params.get("mapping_version", "v1")).strip().lower()
        if version in {"v2", "2"}:
            return "v2"
        return "v1"

    @staticmethod
    def _map_from_options(gene, options):
        if not options:
            raise ValueError("Options for gene mapping cannot be empty.")
        idx = int(float(gene) * len(options))
        idx = max(0, min(idx, len(options) - 1))
        return options[idx]

    def _reference_dim(self):
        return int(self.seq_len if self.is_univariate else self.n_features)

    @staticmethod
    def _ratio_to_bottleneck_size(ref_dim, ratio):
        raw = Decimal(str(ref_dim)) * Decimal(str(ratio))
        rounded = raw.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        return int(rounded)

    @staticmethod
    def _allocate_strict_steps(total_delta, num_layers, curvature):
        total_delta = int(total_delta)
        num_layers = int(num_layers)
        if total_delta <= 0 or num_layers <= 0:
            return None
        if num_layers > total_delta:
            return None

        positions = np.arange(1, num_layers + 1, dtype=float)
        raw = np.power(positions, float(curvature))
        raw = raw / raw.sum() * float(total_delta)

        steps = np.floor(raw).astype(int)
        steps = np.maximum(steps, 1)
        current_sum = int(steps.sum())

        if current_sum > total_delta:
            for idx in np.argsort(-steps):
                if current_sum <= total_delta:
                    break
                if steps[idx] > 1:
                    steps[idx] -= 1
                    current_sum -= 1
            if current_sum > total_delta:
                return None

        remainder = total_delta - int(steps.sum())
        if remainder > 0:
            frac = raw - np.floor(raw)
            order = np.argsort(-frac)
            for i in range(remainder):
                steps[order[i % num_layers]] += 1

        return steps.tolist()

    def _build_monotone_hidden_dims(self, start_dim, end_dim, num_layers, curvature):
        start_dim = int(start_dim)
        end_dim = int(end_dim)
        num_layers = int(num_layers)
        if start_dim <= 0 or end_dim <= 0 or num_layers <= 0:
            return None
        if start_dim == end_dim:
            return None

        decreasing = start_dim > end_dim
        delta = abs(start_dim - end_dim)
        steps = self._allocate_strict_steps(delta, num_layers, curvature)
        if steps is None:
            return None

        dims = []
        current = start_dim
        for step in steps:
            current = current - step if decreasing else current + step
            dims.append(int(current))
        return dims

    @staticmethod
    def _estimate_step(start_dim, dims):
        if not dims:
            return 1
        prev = int(start_dim)
        diffs = []
        for dim in dims:
            dim = int(dim)
            diffs.append(abs(prev - dim))
            prev = dim
        avg = int(round(float(np.mean(diffs)))) if diffs else 1
        return max(avg, 1)

    def _decode_solution_v1(self, y2, y3, y4, y5):
        ref_dim = self._reference_dim()

        self.encoder_layer_step = self.map_layer_step(y2, ref_dim)
        self.decoder_layer_step = self.map_layer_step(y4, ref_dim)

        max_enc = max(1, ref_dim // self.encoder_layer_step)
        max_dec = max(1, ref_dim // self.decoder_layer_step)

        self.encoder_num_layers = self.map_num_layers(y5, max_enc)
        self.decoder_num_layers = self.map_num_layers(y3, max_dec)

        if self.is_univariate:
            encoder_hidden_dims = self.calculate_univariate_hidden_dims(
                self.seq_len,
                self.encoder_layer_step,
                self.encoder_num_layers
            )
        else:
            encoder_hidden_dims = self.calculate_hidden_dims(
                self.n_features,
                self.encoder_layer_step,
                self.encoder_num_layers
            )

        if encoder_hidden_dims is None:
            return False

        self.hidden_dims = encoder_hidden_dims
        self.bottleneck_size = encoder_hidden_dims[-1]
        return True

    def _decode_solution_v2(self, y2, y3, y4, y5):
        ref_dim = self._reference_dim()
        if ref_dim <= 1:
            return False

        encoder_depth = self._map_from_options(y2, [1, 2, 3, 4, 5])
        # Dense, bounded ratio grid keeps mapping simple while improving coverage.
        bottleneck_ratio = self._map_from_options(y3, [round(r / 100.0, 2) for r in range(4, 51)])
        encoder_curvature = self._map_from_options(y4, [0.7, 1.0, 1.3, 1.8])
        decoder_depth_offset = self._map_from_options(y5, [-1, 0, 1, 2])

        bottleneck_size = self._ratio_to_bottleneck_size(ref_dim, bottleneck_ratio)
        bottleneck_size = max(1, min(ref_dim - 1, bottleneck_size))

        max_depth = max(1, ref_dim - bottleneck_size)
        encoder_depth = max(1, min(int(encoder_depth), max_depth))

        encoder_hidden_dims = self._build_monotone_hidden_dims(
            start_dim=ref_dim,
            end_dim=bottleneck_size,
            num_layers=encoder_depth,
            curvature=encoder_curvature,
        )
        if not encoder_hidden_dims:
            return False

        decoder_target = ref_dim
        decoder_depth = max(1, min(int(encoder_depth + decoder_depth_offset), max_depth))
        decoder_curvature = max(0.4, 2.0 - float(encoder_curvature))
        decoder_hidden_dims = self._build_monotone_hidden_dims(
            start_dim=bottleneck_size,
            end_dim=decoder_target,
            num_layers=decoder_depth,
            curvature=decoder_curvature,
        )
        if not decoder_hidden_dims:
            return False

        self.hidden_dims = encoder_hidden_dims
        self.decoder_hidden_dims = decoder_hidden_dims
        self.bottleneck_size = bottleneck_size
        self.encoder_num_layers = int(len(self.hidden_dims))
        self.decoder_num_layers = int(len(self.decoder_hidden_dims))
        self.encoder_layer_step = self._estimate_step(ref_dim, self.hidden_dims)
        self.decoder_layer_step = self._estimate_step(self.bottleneck_size, self.decoder_hidden_dims)
        self.mapping_context = {
            "mapping_version": "v2",
            "ref_dim": int(ref_dim),
            "bottleneck_ratio": float(bottleneck_ratio),
            "encoder_curvature": float(encoder_curvature),
            "decoder_curvature": float(decoder_curvature),
            "decoder_depth_offset": int(decoder_depth_offset),
        }
        return True

    def calculate_hidden_dims(self, input_dim, layer_step, num_layers):
        hidden_dims = []
        current_dim = input_dim
        if current_dim - layer_step * num_layers <= 0:
            Log.debug("Invalid encoder configuration: non-positive hidden dimension.")
            return None
        for _ in range(num_layers):
            current_dim -= layer_step
            if current_dim <= 0:
                Log.debug("Invalid encoder configuration: layer_step too large.")
                return None
            hidden_dims.append(int(current_dim))
        Log.debug(f"Encoder hidden dims: {hidden_dims}")
        return hidden_dims

    def calculate_univariate_hidden_dims(self, h_init, layer_step, num_layers):
        return self.calculate_hidden_dims(h_init, layer_step, num_layers)

    def calculate_decoder_hidden_dims(self, start_dim, end_dim, layer_step, num_layers):
        hidden_dims = []
        current_dim = start_dim
        if current_dim + layer_step * num_layers < end_dim:
            Log.debug("Invalid decoder configuration: cannot reach output dimension.")
            return None
        for idx in range(num_layers):
            current_dim += layer_step
            if current_dim >= end_dim and idx != num_layers - 1:
                current_dim = end_dim
            hidden_dims.append(int(current_dim))
        if hidden_dims and hidden_dims[-1] < end_dim:
            hidden_dims.append(end_dim)
        Log.debug(f"Decoder hidden dims: {hidden_dims}")
        return hidden_dims

    def generate_autoencoder(self, layer_type, dataset_shape, hidden_dims, symmetrical=True, decoder_hidden_dims_override=None):
        # Build encoder
        self.encoder_hidden_dims = hidden_dims
        encoder_input_size = self.n_features
        for hidden_dim in self.encoder_hidden_dims:
            self.encoding_layers.append(self.get_layer_object(
                input_size=encoder_input_size,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True
            ))
            encoder_input_size = hidden_dim

        self.encoding_layers.append(nn.Linear(self.bottleneck_size, self.bottleneck_size))
        self.encoding_layers.append(nn.Linear(self.bottleneck_size, self.bottleneck_size))

        # Build decoder
        self.decoding_layers = nn.ModuleList()
        if decoder_hidden_dims_override is not None:
            self.decoder_hidden_dims = [int(v) for v in decoder_hidden_dims_override]
        elif symmetrical:
            self.decoder_hidden_dims = self.encoder_hidden_dims[::-1]
        else:
            self.decoder_hidden_dims = self.calculate_decoder_hidden_dims(
                start_dim=self.bottleneck_size,
                end_dim=self.n_features,
                layer_step=self.decoder_layer_step,
                num_layers=self.decoder_num_layers
            )
            if self.decoder_hidden_dims is None:
                self.is_valid = False
                Log.debug("Invalid asymmetrical decoder configuration.")
                return

        decoder_input_size = self.bottleneck_size
        for hidden_dim in self.decoder_hidden_dims:
            self.decoding_layers.append(self.get_layer_object(
                input_size=decoder_input_size,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True
            ))
            decoder_input_size = hidden_dim

        self.decoding_layers.append(nn.Linear(decoder_input_size, self.n_features))

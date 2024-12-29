import os
import time
from datetime import datetime
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import random
import hashlib
from tabulate import tabulate

from log import Log
from .base import BaseVAE
from .types_ import *


class RNNVAE(BaseVAE, nn.Module):
    def __init__(self, solution, **kwargs) -> None:
        super(RNNVAE, self).__init__()
        """
        Dimensionality:
        y1: topology shape,
        y2: layer type,
        y3: layer step,
        y4: number of layers,
        y5: activation function
        y6: optimizer algorithm.
        """

        #TODO Remove on HPC
        #solution = [0.57584263, 0.63118431, 0.34364192, 0.02515069, 0.20950322, 0.33386429]
        # Initialize validity flag
        self.is_valid = True

        # Extract data parameters
        n_features = kwargs['data_params']['n_features']
        seq_len = kwargs['data_params']['seq_len']
        batch_size = kwargs['data_params']['batch_size']
        data_shape = kwargs['data_params'].get('data_shape', None)

        # Determine if the data is univariate or multivariate
        if data_shape is not None:
            if len(data_shape) == 3:
                # Multivariate data
                n_features = data_shape[2]
                seq_len = data_shape[1]
                is_univariate = False
            elif len(data_shape) == 2:
                # Univariate data
                n_features = 1
                seq_len = data_shape[1]
                is_univariate = True
            else:
                raise ValueError(f"Unsupported data shape: {data_shape}")
        else:
            # Assume multivariate if data_shape is not provided
            is_univariate = n_features == 1

        self.is_univariate = is_univariate  # Store for use in other methods

        self.id = str(int(time.time())).strip()
        self.dataset_shape = [n_features, seq_len]
        self.encoding_layers = nn.ModuleList()
        self.decoding_layers = nn.ModuleList()

        self.n_features = n_features
        self.seq_len = seq_len
        self.batch_size = batch_size

        # Corrected indices for solution
        y1 = solution[0]  # topology shape
        y2 = solution[1]  # layer type
        y3 = solution[2]  # layer step
        y4 = solution[3]  # number of layers
        y5 = solution[4]  # activation function
        y6 = solution[5]  # optimizer algorithm

        self.shape = self.map_shape(y1)
        self.layer_type = self.map_layer_type(y2)
        self.activation = self.map_activation(y5)
        self.symmetrical = self.shape == "SYMMETRICAL"

        # Map optimizer regardless of validity
        self.optimizer_name = self.map_optimizer(y6, self)

        # Adjust calculations based on univariate or multivariate data
        if self.is_univariate:
            # For univariate data, set an initial hidden dimension
            # Compute layer_step
            self.layer_step = self.map_layer_step_univariate(y3, seq_len)
            # Compute num_layers
            self.num_layers = self.map_num_layers_univariate(y4, seq_len)
            # Calculate encoder hidden dimensions
            encoder_hidden_dims = self.calculate_univariate_hidden_dims(seq_len, self.layer_step, self.num_layers)
            if encoder_hidden_dims is None:
                self.is_valid = False
                Log.error("Invalid model configuration detected during encoder hidden dimensions calculation.")
                # Set default values for attributes
                self.bottleneck_size = None
                self.hidden_dims = []
                self.encoding_layers = None
                self.decoding_layers = None
                self.get_hash()
                return

            self.hidden_dims = encoder_hidden_dims
            self.bottleneck_size = encoder_hidden_dims[-1]
        else:
            # For multivariate data, compute hidden_dim, layer_step, num_layers
            self.layer_step = self.map_layer_step(y3, self.n_features)
            self.num_layers = self.map_num_layers(y4, self.n_features)
            # Calculate encoder hidden dimensions
            encoder_hidden_dims = self.calculate_hidden_dims(self.n_features, self.layer_step, self.num_layers)
            if encoder_hidden_dims is None:
                self.is_valid = False
                Log.error("Invalid model configuration detected during encoder hidden dimensions calculation.")
                # Set default values for attributes
                self.bottleneck_size = None
                self.hidden_dims = []
                self.encoding_layers = None
                self.decoding_layers = None
                self.get_hash()
                return

            self.hidden_dims = encoder_hidden_dims
            self.bottleneck_size = encoder_hidden_dims[-1]

        # If the model is invalid, exit the initialization
        if not self.is_valid:
            # Ensure all necessary attributes are set before returning
            self.get_hash()
            return

        # Generate the autoencoder with dynamic parameters
        self.generate_autoencoder(
            self.layer_type,
            self.num_layers,
            self.dataset_shape,
            self.layer_step,
            self.hidden_dims,
            symmetrical=self.symmetrical  # Use the mapped shape
        )

        self.get_hash()
        outputs = []

        outputs.append([self.hash_id,
                        self.shape,
                        self.layer_type,
                        self.layer_step,
                        self.num_layers,
                        self.activation_name,
                        self.optimizer_name,
                        self.bottleneck_size,
                        self.encoding_layers,
                        self.decoding_layers])

        Log.info(tabulate(outputs, headers=["ID",
                                            "Shape (y1)",
                                            "Layer type (y2)",
                                            "Layer step (y3)",
                                            "Layers (y4)",
                                            "Activation func. (y5)",
                                            "Optimizer (y6)",
                                            "Bottleneck size",
                                            "Encoder",
                                            "Decoder"], tablefmt="pretty"))

    def get_hash(self):
        self.hash_id = hashlib.sha1(str(str(self.shape) +
                                        str(self.layer_type) +
                                        str(self.layer_step) +
                                        str(self.num_layers) +
                                        str(self.activation_name) +
                                        str(self.optimizer_name) +
                                        str(self.bottleneck_size)).encode('utf-8')).hexdigest()
        return self.hash_id

    def encode(self, x: Tensor) -> List[Tensor]:
        #print(f"Input to encoder: {x.shape}")

        # Pass through encoder layers
        for i, layer in enumerate(self.encoding_layers[:-2]):
            if isinstance(layer, (nn.LSTM, nn.GRU, nn.RNN)):
                x, _ = layer(x)
                x = self.activation(x)
                #print(f"Shape after encoder layer {i}: {x.shape}")
            else:
                x = layer(x)
                x = self.activation(x)
                #print(f"Shape after encoder layer {i}: {x.shape}")

        # Use the output from the last time step
        x_last = x[:, -1, :]
        #print(f"Encoder output at last time step: {x_last.shape}")

        # Apply the final linear layers to obtain mu and log_var
        mu = self.encoding_layers[-2](x_last)
        #print(f"Shape after Linear layer for mu: {mu.shape}")

        log_var = self.encoding_layers[-1](x_last)
        #print(f"Shape after Linear layer for log_var: {log_var.shape}")

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        #print(f"Latent input shape: {z.shape}")

        # Map latent vector to decoder input
        decoder_input = z
        #print(f"Decoder input: {decoder_input.shape}")
        decoder_input = decoder_input.unsqueeze(1).repeat(1, self.seq_len, 1)
        #print(f"Decoder input after unsqueeze and repeat: {decoder_input.shape}")

        x = decoder_input

        # Pass through decoder layers
        for i, layer in enumerate(self.decoding_layers[:-1]):
            if isinstance(layer, (nn.LSTM, nn.GRU, nn.RNN)):
                x, _ = layer(x)
                x = self.activation(x)
                #print(f"Shape after decoder layer {i}: {x.shape}")
            else:
                x = layer(x)
                x = self.activation(x)
                #print(f"Shape after decoder layer {i}: {x.shape}")

        # Apply decoder output layer to map to n_features
        batch_size, seq_len, hidden_dim = x.size()
        x = x.contiguous().view(-1, hidden_dim)  # [batch_size*seq_len, hidden_dim]
        x = self.decoding_layers[-1](x)  # [batch_size*seq_len, n_features]
        x = x.view(batch_size, seq_len, self.n_features)
        #print(f"Reconstructed shape: {x.shape}")
        return x

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        #print(f"Sampled latent vector z: {z.shape}")
        return z

    def forward(self, input: dict, **kwargs) -> dict:
        # Extract signal from the input dictionary.
        signal = input['signal']



        # Encode the input signal to obtain the latent distribution parameters (mu and log_var).
        mu, log_var = self.encode(signal)

        # Sample the latent variable z from the latent distribution using the reparameterization trick.
        z = self.reparameterize(mu, log_var)

        # Decode the latent variable z to reconstruct the original input.
        reconstructed = self.decode(z)

        # Create a dictionary containing the original signal, reconstructed signal, and latent distribution parameters.
        response = {
            'signal': signal,
            'reconstructed': reconstructed,
            'mu': mu,
            'log_var': log_var
        }

        # Return the response dictionary.
        return response

    def loss_function(self, curr_device: str = 'cuda', **kwargs) -> dict:
        input = kwargs['signal']
        recons = kwargs['reconstructed']
        mu = kwargs['mu']
        log_var = kwargs['log_var']

        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        #print(f"Input shape: {input.shape}")
        #print(f"Reconstructed shape: {recons.shape}")
        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight * kld_loss
        details = {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': -kld_loss.detach()}
        return details

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        reconstructed = self.forward(x)['reconstructed']
        return reconstructed

    def get_layer_object(self, input_size, hidden_size, num_layers, batch_first):
        hidden_size = int(hidden_size) if isinstance(hidden_size, (np.integer, np.int64, float)) else hidden_size

        if self.layer_type == 'LSTM':
            return nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=batch_first
            )

        elif self.layer_type == 'GRU':
            return nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=batch_first
            )

        elif self.layer_type == 'RNN_TANH':
            return nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                nonlinearity='tanh',
                batch_first=batch_first
            )

    def map_shape(self, gene):
        gene = float(gene)
        shapes = ["SYMMETRICAL", "A-SYMMETRICAL"]
        num_shapes = len(shapes)
        index = int(gene * num_shapes)
        if index >= num_shapes:
            index = num_shapes - 1  # Adjust index if gene is 1.0 or slightly above due to floating-point precision

        if 0 <= index < num_shapes:
            return shapes[index]
        else:
            raise ValueError(f"Gene value {gene} is out of bounds [0.0, 1.0].")

    def map_layer_type(self, gene):
        gene = float(gene)
        layer_types = ["LSTM", "GRU", "RNN_TANH"]
        num_types = len(layer_types)
        index = int(gene * num_types)
        if index >= num_types:
            index = num_types - 1  # Adjust index if gene is 1.0 or slightly above due to floating-point precision

        if 0 <= index < num_types:
            return layer_types[index]
        else:
            raise ValueError(f"Gene value {gene} is out of bounds [0.0, 1.0].")

    def map_activation(self, gene):
        gene = float(gene)
        activation_functions = [
            F.elu,
            F.relu,
            F.leaky_relu,
            F.rrelu,
            F.selu,
            F.celu,
            F.gelu,
            torch.tanh
        ]
        activation_names = [
            "ELU",
            "ReLU",
            "Leaky ReLU",
            "RReLU",
            "SELU",
            "CELU",
            "GELU",
            "Tanh"
        ]

        num_activations = len(activation_functions)
        index = int(gene * num_activations)
        if index >= num_activations:
            index = num_activations - 1  # Adjust index if gene is 1.0 or slightly above due to floating-point precision

        if 0 <= index < num_activations:
            self.activation_name = activation_names[index]
            return activation_functions[index]
        else:
            raise ValueError(f"Gene value {gene} is out of bounds [0.0, 1.0).")

    def map_layer_step(self, gene, n_features):
        gene = float(gene)
        min_step = 1
        max_step = n_features
        layer_step = int(min_step + gene * (max_step - min_step))
        layer_step = max(min(layer_step, max_step), min_step)
        Log.debug(f"Mapped layer_step: {layer_step}")
        return layer_step

    def map_layer_step_univariate(self, gene, seq_len):
        gene = float(gene)
        min_step = 1
        max_step = max(1, seq_len)
        layer_step = int(min_step + gene * (max_step - min_step))
        layer_step = max(min(layer_step, max_step), min_step)
        Log.debug(f"Mapped layer_step (univariate): {layer_step}")
        return layer_step

    def map_num_layers(self, gene, n_features):
        gene = float(gene)
        min_layers = 1
        max_layers = n_features  # Set maximum number of layers to n_features
        num_layers = int(min_layers + gene * (max_layers - min_layers))
        num_layers = max(min(num_layers, max_layers), min_layers)
        Log.debug(f"Mapped num_layers: {num_layers}")
        return num_layers

    def map_num_layers_univariate(self, gene, seq_len):
        gene = float(gene)
        min_layers = 1
        max_layers = max(2, seq_len)
        num_layers = int(min_layers + gene * (max_layers - min_layers))
        num_layers = max(min(num_layers, max_layers), min_layers)
        Log.debug(f"Mapped num_layers (univariate): {num_layers}")
        return num_layers

    def calculate_hidden_dims(self, input_dim, layer_step, num_layers):
        hidden_dims = []
        current_dim = input_dim

        # Pre-check if the given layer_step and num_layers will result in positive hidden dimensions
        min_hidden_dim = current_dim - layer_step * num_layers
        if min_hidden_dim <= 0:
            # Configuration is invalid
            Log.error("Invalid configuration: layer_step and num_layers combination results in non-positive hidden dimensions.")
            return None

        for idx in range(num_layers):
            current_dim -= layer_step
            if current_dim <= 0:
                Log.error("Invalid configuration: layer_step is too large, resulting in non-positive hidden dimensions.")
                return None
            hidden_dims.append(int(current_dim))

        Log.debug((f"encoder_hidden_dims: {hidden_dims}"))
        return hidden_dims

    def calculate_univariate_hidden_dims(self, h_init, layer_step, num_layers):
        hidden_dims = []
        current_dim = h_init

        # Pre-check if the given layer_step and num_layers will result in positive hidden dimensions
        min_hidden_dim = current_dim - layer_step * num_layers
        if min_hidden_dim <= 0:
            # Configuration is invalid
            Log.error("Invalid configuration: layer_step and num_layers combination results in non-positive hidden dimensions.")
            return None

        for idx in range(num_layers):
            current_dim -= layer_step
            if current_dim <= 0:
                Log.error("Invalid configuration: layer_step is too large, resulting in non-positive hidden dimensions.")
                return None
            hidden_dims.append(int(current_dim))
        Log.debug((f"encoder_hidden_dims: {hidden_dims}"))
        return hidden_dims

    def calculate_decoder_hidden_dims(self, start_dim, end_dim, layer_step, num_layers):
        hidden_dims = []
        current_dim = start_dim

        # Pre-check if the given layer_step and num_layers will reach or exceed the end_dim
        max_possible_dim = current_dim + layer_step * num_layers
        if max_possible_dim < end_dim:
            # Configuration is invalid
            Log.error("Invalid configuration: Decoder cannot reach end_dim with the given layer_step and num_layers.")
            return None

        for idx in range(num_layers):
            current_dim += layer_step
            if current_dim >= end_dim and idx != num_layers - 1:
                # Adjust current_dim to not exceed end_dim before the last layer
                current_dim = end_dim
            hidden_dims.append(int(current_dim))
        # Ensure the last layer dimension is at least end_dim
        if hidden_dims and hidden_dims[-1] < end_dim:
            hidden_dims.append(end_dim)
        return hidden_dims

    def generate_autoencoder(self, layer_type, num_layers, dataset_shape, layer_step, hidden_dims, symmetrical=True):
        self.encoder_hidden_dims = hidden_dims
        encoder_input_size = self.n_features
        # Build the encoder layers
        for hidden_dim in self.encoder_hidden_dims:
            self.encoding_layers.append(self.get_layer_object(
                input_size=encoder_input_size,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True
            ))
            encoder_input_size = hidden_dim

        # Add the linear layers to the encoder
        self.encoding_layers.append(nn.Linear(self.bottleneck_size, self.bottleneck_size))
        self.encoding_layers.append(nn.Linear(self.bottleneck_size, self.bottleneck_size))

        # Build the decoder layers
        self.decoding_layers = nn.ModuleList()
        if symmetrical:
            # Symmetrical decoder
            self.decoder_hidden_dims = self.encoder_hidden_dims[::-1]
        else:
            # Asymmetrical decoder
            # Define different hidden dimensions for decoder
            self.decoder_hidden_dims = self.calculate_decoder_hidden_dims(
                start_dim=self.bottleneck_size,
                end_dim=self.n_features,
                layer_step=layer_step,
                num_layers=num_layers
            )
            if self.decoder_hidden_dims is None:
                self.is_valid = False
                Log.error("Invalid model configuration detected during decoder hidden dimensions calculation.")
                return

        # Define the mapping from latent space to decoder input
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

    def map_optimizer(self, gene, architecture):
        gene = float(gene)
        optimizer_names = ["Adam", "Adagrad", "SGD", "RAdam", "ASGD", "RPROP"]
        num_optimizers = len(optimizer_names)
        index = int(gene * num_optimizers)
        if index >= num_optimizers:
            index = num_optimizers - 1  # Adjust index if gene is 1.0 or slightly above due to floating-point precision

        if 0 <= index < num_optimizers:
            self.optimizer_name = optimizer_names[index]
            return optimizer_names[index]
        else:
            raise ValueError(f"Gene value {gene} is out of bounds [0.0, 1.0].")

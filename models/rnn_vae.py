import os
import time
from datetime import datetime
import numpy as np
import torch
import torchmetrics
from torch import tensor

from .base import BaseVAE
from .types_ import *
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import random
import hashlib
from tabulate import tabulate


class RNNVAE(BaseVAE, nn.Module):
    def __init__(self, solution, **kwargs) -> None:
        super(RNNVAE, self).__init__()

        """
        Dimensionality:
        y1: topology shape,
        y2: layer type
        y3: number of neurons per layer,
        y4: number of layers,
        y5: activation function
        y6: number of epochs,
        y7: learning rate
        y8: optimizer algorithm.
        """
        n_features = kwargs['model_params']['n_features']
        seq_len = kwargs['model_params']['seq_len']
        batch_size = kwargs['data_params']['batch_size']

        self.id = str(int(time.time())).strip()
        self.dataset_shape = [n_features, seq_len]
        self.encoding_layers = nn.ModuleList()
        self.decoding_layers = nn.ModuleList()

        self.topology_shape = self.map_shape(solution[0])
        self.layer_type = self.map_layer_type(solution[1])
        self.layer_step = self.map_layer_step(solution[2], self.dataset_shape)
        self.num_layers = self.map_num_layers(solution[3], self.layer_step, self.dataset_shape)
        # https://ai.stackexchange.com/questions/3156/how-to-select-number-of-hidden-layers-and-number-of-memory-cells-in-an-lstm
        self.activation = self.map_activation(solution[4])
        self.num_epochs = self.map_num_epochs(solution[5])
        self.learning_rate = self.map_learning_rate(solution[6])

        self.bottleneck_size = 0
        self.seq_len = seq_len
        self.n_features = n_features
        self.batch_size = batch_size

        self.generate_autoencoder(self.topology_shape,
                                  self.layer_type,
                                  self.num_layers,
                                  self.dataset_shape,
                                  self.layer_step)

        """For testing:"""
        # self.encoding_layers.append(self.get_layer_object(
        #     input_size=1,
        #     hidden_size=140,
        #     num_layers=1,
        #     batch_first=True
        # ))
        #
        # self.encoding_layers.append(self.get_layer_object(
        #     input_size=140,
        #     hidden_size=70,
        #     num_layers=1,
        #     batch_first=True
        # ))
        #
        # self.encoding_layers.append(self.get_layer_object(
        #     input_size=70,
        #     hidden_size=35,
        #     num_layers=1,
        #     batch_first=True
        # ))
        # self.bottleneck_size = 35
        # self.encoding_layers.append(nn.Linear(self.bottleneck_size, self.bottleneck_size))
        # self.encoding_layers.append(nn.Linear(self.bottleneck_size, self.bottleneck_size))
        #
        # self.decoding_layers.append(self.get_layer_object(
        #     input_size=35,
        #     hidden_size=70,
        #     num_layers=1,
        #     batch_first=True
        # ))
        #
        # self.decoding_layers.append(self.get_layer_object(
        #     input_size=70,
        #     hidden_size=140,
        #     num_layers=1,
        #     batch_first=True
        # ))
        #
        # self.decoding_layers.append(self.get_layer_object(
        #     input_size=140,
        #     hidden_size=140,
        #     num_layers=1,
        #     batch_first=True
        # ))
        # self.decoding_layers.append(nn.Linear(140, self.seq_len))

        self.optimizer = self.map_optimizer(solution[7])
        self.get_hash()
        outputs = []

        outputs.append([self.hash_id,
                        self.topology_shape,
                        self.layer_type,
                        self.layer_step,
                        self.num_layers,
                        self.activation_name,
                        self.num_epochs,
                        self.learning_rate,
                        self.optimizer_name,
                        self.bottleneck_size,
                        self.encoding_layers,
                        self.decoding_layers])

        print(tabulate(outputs, headers=["ID",
                                         "Shape (y1)",
                                         "Layer type (y2)",
                                         "Layer step (y3)",
                                         "Layers (y4)",
                                         "Activation func. (y5)",
                                         "Epochs (y6)",
                                         "Learning rate (y7)",
                                         "Optimizer (y8)",
                                         "Bottleneck size",
                                         "Encoder",
                                         "Decoder", ], tablefmt="pretty"))

    def get_hash(self):

        self.hash_id = hashlib.sha1(str(str(self.topology_shape) +
                                        str(self.layer_type) +
                                        str(self.layer_step) +
                                        str(self.num_layers) +
                                        str(self.activation_name) +
                                        str(self.num_epochs) +
                                        str(self.learning_rate) +
                                        str(self.optimizer_name) +
                                        str(self.bottleneck_size) +
                                        str(self.encoding_layers) +
                                        str(self.decoding_layers)).encode('utf-8')).hexdigest()
        return self.hash_id

    def encode(self, x: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """

        x = x.reshape((self.batch_size, self.seq_len, self.n_features))
        x, (hidden_n, cell_n) = x, (None, None)

        for layer in self.encoding_layers[:-2]:
            if layer.mode == 'LSTM':
                x, (hidden_n, cell_n) = layer(x)
            elif layer.mode == 'GRU':
                x, hidden_n = layer(x)
            elif layer.mode == 'RNN_TANH':
                x, hidden_n = layer(x)

        # # TODO Why hidden state needs to be passed
        # # https://github.com/chrisvdweth/ml-toolkit\
        # # https://discuss.pytorch.org/t/lstm-autoencoders-in-pytorch/139727
        hidden_n = hidden_n.reshape((self.batch_size, self.bottleneck_size))
        mu = self.encoding_layers[-2](hidden_n)
        log_var = self.encoding_layers[-1](hidden_n)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """

        x = z.reshape((self.batch_size, self.bottleneck_size))

        for layer in self.decoding_layers[:-1]:
            if layer.mode == 'LSTM':
                x, (hidden_n, cell_n) = layer(x)
            elif layer.mode == 'GRU':
                x, hidden_n = layer(x)
            elif layer.mode == 'RNN_TANH':
                x, hidden_n = layer(x)

        reconstructed = self.decoding_layers[-1](x)

        return reconstructed

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return (eps * std) + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        # TODO Try not to use tensor.reshape
        # https://discuss.pytorch.org/t/for-beginners-do-not-use-view-or-reshape-to-swap-dimensions-of-tensors/75524

        input = input.reshape(input.shape[1], input.shape[0])
        mu, log_var = self.encode(input)

        z = self.reparameterize(mu, log_var)

        input = input.reshape(input.shape[1], input.shape[0])
        reconstructed = self.decode(z)

        return [reconstructed, input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight * kld_loss
        details = {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': -kld_loss.detach()}
        return details

    def sample(self,
               num_samples: int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.bottleneck_size)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        reconstructed = self.forward(x)[0]
        return reconstructed

    def get_layer_object(self, input_size, hidden_size, num_layers, batch_first):
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
                batch_first=batch_first
            )

    def map_shape(self, gene):
        gene = np.array([gene])
        bins = np.array([0.0, 0.5])
        inds = np.digitize(gene, bins)

        if inds[0] - 1 == 0:
            return "SYMMETRICAL"

        elif inds[0] - 1 == 1:
            return "A-SYMMETRICAL"

        else:
            raise ValueError(f"Value not between boundaries 0.0 and 1.0. Value is: {inds[0] - 1}")

    def map_layer_type(self, gene):
        gene = np.array([gene])
        bins = np.array([0.33, 0.66, 1.01])
        inds = np.digitize(gene, bins)
        bin = inds[0]

        if bin == 0:
            return "LSTM"

        elif bin == 1:
            return "GRU"

        elif bin == 2:
            return "RNN_TANH"

        else:
            raise ValueError(f"Value not between boundaries 0.0 and 1.0. Value is: {inds[0] - 1}")

    def map_layer_step(self, gene, dataset_shape):
        gene = np.array([gene])
        bins = []
        value = 1 / dataset_shape[1]
        step = value
        for col in range(0, dataset_shape[1]):
            bins.append(step)
            step += value
        bins[-1] = 1.01
        inds = np.digitize(gene, bins)
        return inds[0]

    def map_num_layers(self, gene, layer_step, dataset_shape):
        if layer_step == 0:
            max_layers = dataset_shape[1]
            return max_layers

        else:
            max_layers = round(dataset_shape[1] / layer_step)

        if max_layers == 1:
            return 1

        else:
            gene = np.array([gene])

            bins = []
            value = 1 / max_layers
            step = value
            for col in range(0, max_layers):
                bins.append(step)
                step += value
            bins[-1] = 1.01
            inds = np.digitize(gene, bins)

            return int(inds[0])

    def map_activation(self, gene):
        gene = np.array([gene])
        bins = np.array([0.0, 0.125, 0.25, 0.375, 0.500, 0.625, 0.750, 0.875, 1.01])
        inds = np.digitize(gene, bins)

        if inds[0] - 1 == 0:
            self.activation_name = "ELU"
            return F.elu

        elif inds[0] - 1 == 1:
            self.activation_name = "RELU"
            return F.relu

        elif inds[0] - 1 == 2:
            self.activation_name = "Leaky RELU"
            return F.leaky_relu

        elif inds[0] - 1 == 3:
            self.activation_name = "RRELU"
            return F.rrelu

        elif inds[0] - 1 == 4:
            self.activation_name = "SELU"
            return F.selu

        elif inds[0] - 1 == 5:
            self.activation_name = "CELU"
            return F.celu

        elif inds[0] - 1 == 6:
            self.activation_name = "GELU"
            return F.gelu

        elif inds[0] - 1 == 7:
            self.activation_name = "TANH"
            return torch.tanh

        else:

            raise ValueError(f"Value not between boundaries 0.0 and 1.0. Value is: {inds[0] - 1}")

    def map_num_epochs(self, gene):
        gene = np.array([gene])
        bins = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.60, 0.7, 0.8, 0.9, 1.01])
        inds = np.digitize(gene, bins)

        return int(inds[0]) * 10 + 100

    def map_learning_rate(self, gene):
        # https://www.jeremyjordan.me/nn-learning-rate/
        gene = np.array([gene])
        bins = []
        value = 1 / 100
        step = value
        for col in range(0, 100):
            bins.append(step)
            step += value
        bins[-1] = 1.01
        inds = np.digitize(gene, bins)
        lr = np.array(bins)[inds[0]]

        return round(lr, 2)

    def generate_autoencoder(self, shape, layer_type, layers, dataset_shape, layer_step):

        if shape == "SYMMETRICAL":

            i = dataset_shape[1]
            z = dataset_shape[1] - layer_step
            num_of_layers = layers
            input = self.n_features
            hidden_dim = self.seq_len
            last_decoder_layer_flag = True

            while layers != 0:
                """Minimum depth reached"""
                # TODO Check negatives
                if hidden_dim < 1:
                    break

                if num_of_layers == 1:
                    self.encoding_layers.append(self.get_layer_object(
                        input_size=input,
                        hidden_size=hidden_dim,
                        num_layers=1,
                        batch_first=True
                    ))

                    self.encoding_layers.append(self.get_layer_object(
                        input_size=hidden_dim,
                        hidden_size=hidden_dim - layer_step,
                        num_layers=1,
                        batch_first=True
                    ))

                    self.decoding_layers.insert(0, self.get_layer_object(
                        input_size=hidden_dim - layer_step,
                        hidden_size=hidden_dim,
                        num_layers=1,
                        batch_first=True
                    ))

                    break

                self.encoding_layers.append(self.get_layer_object(
                    input_size=input,
                    hidden_size=hidden_dim,
                    num_layers=1,
                    batch_first=True
                ))

                if last_decoder_layer_flag:
                    """ Last layer needs to have same input and hidden dims"""
                    input = input * hidden_dim
                    last_decoder_layer_flag = False

                self.decoding_layers.insert(0, self.get_layer_object(
                    input_size=hidden_dim,
                    hidden_size=input,
                    num_layers=1,
                    batch_first=True
                ))

                input = hidden_dim
                hidden_dim = hidden_dim - layer_step

                i = i - layer_step
                z = z - layer_step
                layers = layers - 1

            if len(self.encoding_layers) == 0:
                self.bottleneck_size = 0
            else:
                self.bottleneck_size = int(self.encoding_layers[-1].hidden_size)
                self.encoding_layers.append(nn.Linear(self.bottleneck_size, self.bottleneck_size))
                self.encoding_layers.append(nn.Linear(self.bottleneck_size, self.bottleneck_size))
                self.decoding_layers.append(nn.Linear(dataset_shape[1], self.seq_len))

        elif shape == "A-SYMMETRICAL":
            i = dataset_shape[1]
            z = dataset_shape[1] - layer_step

            input_dimension = 1
            hidden_dimension = dataset_shape[1]

            if layers == 1:
                self.encoding_layers.append(self.get_layer_object(
                    input_size=input_dimension,
                    hidden_size=hidden_dimension,
                    num_layers=1,
                    batch_first=True
                ))

                self.decoding_layers.insert(0, self.get_layer_object(
                    input_size=hidden_dimension,
                    hidden_size=hidden_dimension,
                    num_layers=1,
                    batch_first=True
                ))

            if layers >= 2:
                random.seed(datetime.now())
                layers_encoder = random.randint(1, layers)
                layers_decoder = layers - layers_encoder

                encoder_counter = layers_encoder
                decoder_counter = layers_decoder

                if layers_decoder == 0:
                    layers_encoder = layers_encoder - 1
                    layers_decoder = 1

                    encoder_counter = layers_encoder
                    decoder_counter = layers_decoder

                while encoder_counter != 0:

                    if hidden_dimension < 1:
                        hidden_dimension += layer_step
                        break

                    self.encoding_layers.append(self.get_layer_object(
                        input_size=input_dimension,
                        hidden_size=hidden_dimension,
                        num_layers=1,
                        batch_first=True
                    ))
                    if encoder_counter > 1:
                        input_dimension = hidden_dimension
                        hidden_dimension -= layer_step
                    else:
                        break
                    i = i - layer_step
                    z = z - layer_step
                    encoder_counter = encoder_counter - 1

                while decoder_counter != 0:

                    if layers_encoder == 1 and layers_decoder >= layers_encoder:

                        for layer in range(layers_decoder):
                            self.decoding_layers.append(self.get_layer_object(
                                input_size=dataset_shape[1],
                                hidden_size=dataset_shape[1],
                                num_layers=1,
                                batch_first=True
                            ))
                        break

                    if layers_decoder == 1:
                        self.decoding_layers.insert(0, self.get_layer_object(
                            input_size=hidden_dimension,
                            hidden_size=dataset_shape[1],
                            num_layers=1,
                            batch_first=True
                        ))
                        break

                    layer_step = int((dataset_shape[1] - i) / decoder_counter)  # Make more complex logic
                    last_i = i
                    i = i + layer_step
                    z = z + layer_step

                    if layers_decoder > layers_encoder:

                        self.decoding_layers.append(self.get_layer_object(
                            input_size=hidden_dimension,
                            hidden_size=i,
                            num_layers=1,
                            batch_first=True
                        ))
                        hidden_dimension = i

                    else:
                        self.decoding_layers.append(self.get_layer_object(
                            input_size=hidden_dimension,
                            hidden_size=i,
                            num_layers=1,
                            batch_first=True
                        ))
                        hidden_dimension += layer_step
                        input_dimension += layer_step

                    decoder_counter = decoder_counter - 1

            if len(self.encoding_layers) == 0:
                self.bottleneck_size = 0
            else:
                self.bottleneck_size = int(self.encoding_layers[-1].hidden_size)
                self.encoding_layers.append(nn.Linear(self.bottleneck_size, self.bottleneck_size))
                self.encoding_layers.append(nn.Linear(self.bottleneck_size, self.bottleneck_size))
                self.decoding_layers.append(nn.Linear(dataset_shape[1], self.seq_len))

    def map_optimizer(self, gene):
        gene = np.array([gene])
        bins = np.array([0.0, 0.167, 0.334, 0.50, 0.667, 0.834, 1.01])
        inds = np.digitize(gene, bins)

        """When AE does not have any layers"""
        if len(list(self.parameters())) == 0:
            self.optimizer_name = "Empty"
            return None

        # TODO add weight decay to solution array
        # https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c
        if inds[0] - 1 == 0:
            self.optimizer_name = "Adam"
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        elif inds[0] - 1 == 1:
            self.optimizer_name = "Adagrad"
            return torch.optim.Adagrad(self.parameters(), lr=self.learning_rate)

        elif inds[0] - 1 == 2:
            self.optimizer_name = "SGD"
            return torch.optim.SGD(self.parameters(), lr=self.learning_rate)

        elif inds[0] - 1 == 3:
            self.optimizer_name = "RAdam"
            return torch.optim.RAdam(self.parameters(), lr=self.learning_rate)

        elif inds[0] - 1 == 4:
            self.optimizer_name = "ASGD"
            return torch.optim.ASGD(self.parameters(), lr=self.learning_rate)

        elif inds[0] - 1 == 5:
            self.optimizer_name = "RPROP"
            return torch.optim.Rprop(self.parameters(), lr=self.learning_rate)

        else:
            raise ValueError(f"Value not between boundaries 0.0 and 1.0. Value is: {inds[0] - 1}")

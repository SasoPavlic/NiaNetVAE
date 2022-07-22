from random import random

import numpy as np
import torch
from .base import BaseVAE
from .types_ import *
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions

torch.manual_seed(0)


class LSTMVAE(BaseVAE, nn.Module):
    def __init__(self,
                 seq_len,
                 n_features,
                 embedding_dim,
                 solution=[0.1, 0.220, 0.7, 0.1, 0.1, 0.1, 0.1],
                 dataset_shape=[1, 140],
                 **kwargs) -> None:
        super(LSTMVAE, self).__init__()

        """
        Dimensionality:
            y1: topology shape,
            y2: number of neurons per layer
            y3: number of layers,
            y4: activation function
            y5: number of epochs,
            y6: learning rate
            y7: optimizer algorithm.
        """

        self.encoding_layers = nn.ModuleList()
        self.decoding_layers = nn.ModuleList()

        self.shape = self.get_shape(solution[0])
        self.layer_step = self.get_layer_step(solution[1], dataset_shape)
        self.layers = self.get_layers(solution[2], self.layer_step, dataset_shape)
        # https://ai.stackexchange.com/questions/3156/how-to-select-number-of-hidden-layers-and-number-of-memory-cells-in-an-lstm
        self.activation = self.get_activation(solution[3])
        self.epochs = self.get_epochs(solution[4])
        self.learning_rate = self.get_learning_rate(solution[5])

        self.bottleneck_size = embedding_dim
        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.hidden_dim = 2 * embedding_dim

        # TODO make it work
        self.generate_autoencoder(self.shape,
                                  self.layers,
                                  dataset_shape,
                                  self.layer_step)

        self.optimizer = self.get_optimizer(solution[6])

        # self.encoder_rnn1 = nn.LSTM(
        #     input_size=1,
        #     hidden_size=140,
        #     num_layers=1,
        #     batch_first=True
        # )
        #
        # self.encoder_rnn2 = nn.LSTM(
        #     input_size=140,
        #     hidden_size=70,
        #     num_layers=1,
        #     batch_first=True
        # )
        #
        # self.encoder_rnn3 = nn.LSTM(
        #     input_size=70,
        #     hidden_size=35,
        #     num_layers=1,
        #     batch_first=True
        # )
        #
        # self.fc_mu = nn.Linear(self.bottleneck_size, self.bottleneck_size)
        # self.fc_var = nn.Linear(self.bottleneck_size, self.bottleneck_size)
        #
        # self.decoder_rnn1 = nn.LSTM(
        #     input_size=35,
        #     hidden_size=70,
        #     num_layers=1,
        #     batch_first=True
        # )
        #
        # self.decoder_rnn2 = nn.LSTM(
        #     input_size=70,
        #     hidden_size=140,
        #     num_layers=1,
        #     batch_first=True
        # )
        #
        # self.decoder_rnn3 = nn.LSTM(
        #     input_size=140,
        #     hidden_size=140,
        #     num_layers=1,
        #     batch_first=True
        # )
        #
        # self.decoder_output_layer = nn.Linear(140, self.seq_len)

    def encode(self, x: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """

        # input = Tensor (140, 1)

        # batch_size=1, seq_len=140, n_features=1
        x = x.reshape((1, self.seq_len, self.n_features))
        # x = Tensor(1,140,1)

        x, (hidden_n, cell_n) = x, (None, None)
        for layer in self.encoding_layers[:self.layers]:
            x, (hidden_n, cell_n) = layer(x)

        # # TODO Why hidden state needs to be passed
        # # https://github.com/chrisvdweth/ml-toolkit\
        # # https://discuss.pytorch.org/t/lstm-autoencoders-in-pytorch/139727
        hidden_n = hidden_n.reshape((self.n_features, self.embedding_dim))
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
        # z = Tensor (1, 128)
        # x = z.repeat(self.seq_len, self.n_features)

        x = z.reshape((self.n_features, self.embedding_dim))
        # x = Tensor (1, 128)

        x, (hidden_n, cell_n) = x, (None, None)
        for layer in self.decoding_layers[:self.layers]:
            x, (hidden_n, cell_n) = layer(x)

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
        # input = Tensor(140, 1)

        input = input.reshape(input.shape[1], input.shape[0])
        # input = Tensor(140,1)

        mu, log_var = self.encode(input)
        # mu = Tensor(1,128)
        # log_var = Tensor(1,128)

        z = self.reparameterize(mu, log_var)
        # z = Tensor(1,128)

        input = input.reshape(input.shape[1], input.shape[0])
        # input = Tensor(1, 140)
        reconstructed = self.decode(z)
        # reconstructed = Tensor(1, 140)

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
        z = torch.randn(num_samples, self.embedding_dim)

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

    def get_shape(self, gene):
        gene = np.array([gene])
        bins = np.array([0.0, 0.5])
        inds = np.digitize(gene, bins)

        if inds[0] - 1 == 0:
            return "SYMMETRICAL"

        elif inds[0] - 1 == 1:
            return "A-SYMMETRICAL"

        else:
            raise ValueError(f"Value not between boundaries 0.0 and 1.0. Value is: {inds[0] - 1}")

    def get_layer_step(self, gene, dataset_shape):
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

    def get_layers(self, gene, layer_step, dataset_shape):
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

            return inds[0]

    def get_activation(self, gene):
        gene = np.array([gene])
        bins = np.array([0.0, 0.125, 0.25, 0.375, 0.500, 0.625, 0.750, 0.875, 1.01])
        inds = np.digitize(gene, bins)

        if inds[0] - 1 == 0:
            return F.elu

        elif inds[0] - 1 == 1:
            return F.relu

        elif inds[0] - 1 == 2:
            return F.leaky_relu

        elif inds[0] - 1 == 3:
            return F.rrelu

        elif inds[0] - 1 == 4:
            return F.selu

        elif inds[0] - 1 == 5:
            return F.celu

        elif inds[0] - 1 == 6:
            return F.gelu

        elif inds[0] - 1 == 7:
            return torch.tanh

        else:

            raise ValueError(f"Value not between boundaries 0.0 and 1.0. Value is: {inds[0] - 1}")

    def get_epochs(self, gene):
        gene = np.array([gene])
        bins = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.60, 0.7, 0.8, 0.9, 1.01])
        inds = np.digitize(gene, bins)

        return inds[0] * 10 + 100

    def get_learning_rate(self, gene):
        gene = np.array([gene])
        bins = []
        value = 1 / 1000
        step = value
        for col in range(0, 1000):
            bins.append(step)
            step += value
        bins[-1] = 1.01
        inds = np.digitize(gene, bins)
        lr = np.array(bins)[inds[0]]

        return lr

    def generate_autoencoder(self, shape, layers, dataset_shape, layer_step):
        if shape == "SYMMETRICAL":

            i = dataset_shape[1]
            z = dataset_shape[1] - layer_step
            input = self.n_features
            hidden_dim = self.seq_len
            last_decoder_layer_flag = True

            while layers != 0:
                """Minimum depth reached"""
                if z < 1:
                    self.encoding_layers.append(nn.Linear(in_features=i, out_features=z + 1))
                    self.decoding_layers.insert(0, nn.Linear(in_features=z + 1, out_features=i))
                    self.bottleneck_size = z + 1
                    break

                self.encoding_layers.append(nn.LSTM(
                    input_size=input,
                    hidden_size=hidden_dim,
                    num_layers=1,
                    batch_first=True
                ))

                if last_decoder_layer_flag:
                    """ Last layer needs to have same input and hidden dims"""
                    input = input * hidden_dim
                    last_decoder_layer_flag = False

                self.decoding_layers.insert(0, nn.LSTM(
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
                self.bottleneck_size = self.encoding_layers[-1].hidden_size
                self.encoding_layers.append(nn.Linear(self.embedding_dim, self.embedding_dim))
                self.encoding_layers.append(nn.Linear(self.embedding_dim, self.embedding_dim))
                self.decoding_layers.append(nn.Linear(dataset_shape[1], self.seq_len))

        elif shape == "A-SYMMETRICAL":
            i = dataset_shape[1]
            z = dataset_shape[1] - layer_step

            if layers == 1 or layers == 2:
                self.encoding_layers.append(nn.Linear(in_features=i, out_features=z))
                self.decoding_layers.insert(0, nn.Linear(in_features=z, out_features=i))

            if layers >= 3:
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

                    if z < 1:
                        self.encoding_layers.append(nn.Linear(in_features=i, out_features=z + 1))
                        self.bottleneck_size = z + 1
                        break

                    self.encoding_layers.append(nn.Linear(in_features=i, out_features=z))

                    i = i - layer_step
                    z = z - layer_step
                    encoder_counter = encoder_counter - 1

                while decoder_counter != 0:

                    if layers_decoder == 1:
                        self.decoding_layers.insert(0, nn.Linear(in_features=i, out_features=dataset_shape[1]))
                        break

                    layer_step = int((dataset_shape[1] - i) / decoder_counter)  # Make more complex logic
                    last_i = i
                    i = i + layer_step
                    z = z + layer_step
                    decoder_counter = decoder_counter - 1

                    self.decoding_layers.append(nn.Linear(in_features=last_i, out_features=i))

            if len(self.encoding_layers) == 0:
                self.bottleneck_size = 0
            else:
                self.bottleneck_size = self.encoding_layers[-1].out_features

    def get_optimizer(self, gene):
        gene = np.array([gene])
        bins = np.array([0.0, 0.167, 0.334, 0.50, 0.667, 0.834, 1.01])
        inds = np.digitize(gene, bins)

        """When AE does not have any layers"""
        if len(list(self.parameters())) == 0:
            return None

        # TODO add weight decay to solution array
        if inds[0] - 1 == 0:
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        elif inds[0] - 1 == 1:
            return torch.optim.Adagrad(self.parameters(), lr=self.learning_rate)

        elif inds[0] - 1 == 2:
            return torch.optim.SGD(self.parameters(), lr=self.learning_rate)

        elif inds[0] - 1 == 3:
            return torch.optim.RAdam(self.parameters(), lr=self.learning_rate)

        elif inds[0] - 1 == 4:
            return torch.optim.ASGD(self.parameters(), lr=self.learning_rate)

        elif inds[0] - 1 == 5:
            return torch.optim.Rprop(self.parameters(), lr=self.learning_rate)

        else:
            raise ValueError(f"Value not between boundaries 0.0 and 1.0. Value is: {inds[0] - 1}")

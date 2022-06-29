import torch
from .base import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *

torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np
import matplotlib.pyplot as plt;


class LSTMVAE(BaseVAE):
    def __init__(self,
                 seq_len,
                 n_features,
                 embedding_dim,
                 **kwargs) -> None:
        super(LSTMVAE, self).__init__()

        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

        # TODO decouple data size from algorithm
        # TODO TUPLE TO TENSOR
        # https://stackoverflow.com/questions/53032586/attributeerror-tuple-object-has-no-attribute-dim-when-feeding-input-to-pyt

        self.encoder_rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=n_features,
            batch_first=True
        )

        self.encoder_rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=embedding_dim,
            num_layers=n_features,
            batch_first=True
        )

        self.fc_mu = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.fc_var = nn.Linear(self.embedding_dim, self.embedding_dim)

        self.seq_len, self.input_dim = seq_len, self.embedding_dim
        self.hidden_dim, self.n_features = 2 * self.embedding_dim, self.n_features

        # TODO decouple data size from algorithm

        self.decoder_rnn1 = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.embedding_dim,
            num_layers=self.n_features,
            batch_first=True
        )

        self.decoder_rnn2 = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.n_features,
            batch_first=True
        )

        self.decoder_output_layer = nn.Linear(self.hidden_dim, seq_len)

        # self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        # self.N.scale = self.N.scale.cuda()
        self.kl = 0.0005

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        # out, states = self.encoder_rnn1(input)
        # out, (embedding, _) = self.encoder_rnn2(out)
        # TODO hidden state needs to be passed
        # https://github.com/chrisvdweth/ml-toolkit\
        # https://discuss.pytorch.org/t/lstm-autoencoders-in-pytorch/139727

        # Tensor (140, 64)

        x = input.reshape((1, self.seq_len, self.n_features))
        # Tensor(1,140,64)

        x, (_, _) = self.encoder_rnn1(input)
        # Tensor(140,256)

        x, (hidden_n, _) = self.encoder_rnn2(x)
        # Tensor(140, 128)

        hidden_n = hidden_n.reshape((self.n_features, self.embedding_dim))
        # Tensor(64, 128)

        result = hidden_n

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        # Tensor(64, 128)

        log_var = self.fc_var(result)
        # Tensor(64, 128)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        # Tensor (64, 128)
        # x = z.repeat(self.seq_len, self.n_features)

        x = z.reshape((self.n_features, self.input_dim))
        # Tensor (64, 128)

        out, states = self.decoder_rnn1(x)
        # Tensor (64, 128)

        out, states = self.decoder_rnn2(out)
        # Tensor (64, 256)

        x = out.reshape((self.n_features, self.hidden_dim))
        # Tensor (64, 256)
        result = self.decoder_output_layer(x)
        # Tensor (64, 140)
        return result

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
        # TODO do not use reshape
        # https://discuss.pytorch.org/t/for-beginners-do-not-use-view-or-reshape-to-swap-dimensions-of-tensors/75524
        input = input.reshape(input.shape[1], input.shape[0])
        input_shape = input.shape
        mu, log_var = self.encode(input)
        mu_shape = mu.shape
        log_var_shape = log_var.shape

        z = self.reparameterize(mu, log_var)
        z_shape = z.shape

        input = input.reshape(input.shape[1], input.shape[0])
        result = self.decode(z)
        # input = input.reshape(input.shape[1], input.shape[0])
        result_shape = result.shape

        return [result, input, mu, log_var]

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

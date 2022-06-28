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
                 in_features,
                 latent_dims,
                 out_features,
                 **kwargs) -> None:
        super(LSTMVAE, self).__init__()

        self.latent_dims = latent_dims

        seq_len = 140
        n_features = 64
        embedding_dim = 128
        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

        # TODO decouple data size from algorithm
        # TODO TUPLE TO TENSOR
        # https://stackoverflow.com/questions/53032586/attributeerror-tuple-object-has-no-attribute-dim-when-feeding-input-to-pyt


        self.encoder_rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.encoder_rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
        )

        #
        # self.encoder = nn.Sequential(
        #     nn.LSTM(
        #         input_size=n_features,
        #         hidden_size=self.hidden_dim,
        #         num_layers=1,
        #         batch_first=True
        #     ),
        #     nn.LSTM(
        #         input_size=self.hidden_dim,
        #         hidden_size=self.embedding_dim,
        #         num_layers=1,
        #         batch_first=True
        #     )
        #
        # )

        self.fc_mu = nn.Linear(self.latent_dims, self.latent_dims)
        self.fc_var = nn.Linear(self.latent_dims, self.latent_dims)

        input_dim = 128
        n_features = 64
        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features

        # TODO decouple data size from algorithm
        # self.decoder = nn.Sequential(
        #     nn.LSTM(
        #         input_size=input_dim,
        #         hidden_size=input_dim,
        #         num_layers=1,
        #         batch_first=True
        #     ),
        #     nn.LSTM(
        #         input_size=input_dim,
        #         hidden_size=self.hidden_dim,
        #         num_layers=1,
        #         batch_first=True
        #     ),
        #     nn.Linear(self.hidden_dim, n_features)
        # )

        self.decoder_rnn1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=input_dim,
            num_layers=1,
            batch_first=True
        )



        self.decoder_rnn2 = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.decoder_output_layer = nn.Linear(self.hidden_dim, n_features)

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
        out, states = self.encoder_rnn1(input)
        out, states = self.encoder_rnn2(out)
        result = torch.flatten(out, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """

        #result = self.decoder(z)


        out, states = self.decoder_rnn1(z)
        out, states = self.decoder_rnn2(out)
        result = self.decoder_output_layer(out)

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
        input = input.reshape(input.shape[1], input.shape[0])
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)

        input = input.reshape(input.shape[1], input.shape[0])
        result = self.decode(z)
        input = input.reshape(input.shape[1], input.shape[0])

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
        z = torch.randn(num_samples, self.latent_dims)

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

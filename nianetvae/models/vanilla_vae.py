import torch
from .base import BaseVAE
from .types_ import *
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions

torch.manual_seed(0)

class VanillaVAE(BaseVAE):
    def __init__(self,
                 in_features,
                 latent_dims,
                 out_features,
                 **kwargs) -> None:
        super(VanillaVAE, self).__init__()

        self.latent_dims = latent_dims

        self.encoder = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.Linear(512, 256),
            nn.Linear(256, self.latent_dims),
            nn.LeakyReLU())

        self.fc_mu = nn.Linear(self.latent_dims, self.latent_dims)
        self.fc_var = nn.Linear(self.latent_dims, self.latent_dims)

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dims, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.Linear(512, out_features),
            nn.Sigmoid())

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
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

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
        # result = self.decoder_input(z)
        # result = result.view(-1, 512, 2, 2)
        result = self.decoder(z)
        # result = self.final_layer(result)
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
        input = input.view(input.size(0), -1)
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

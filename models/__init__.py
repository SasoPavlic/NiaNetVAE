from .base import *
from .vanilla_vae import *
from .rnn_vae import *

# Aliases
VAE = VanillaVAE
RNNVAE = RNNVAE

vae_models = {'VanillaVAE':VAE,
              'RNNVAE':RNNVAE}



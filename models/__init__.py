from .base import *
from .vanilla_vae import *
from .lstm_vae import *

# Aliases
VAE = VanillaVAE
LSTMVAE = LSTMVAE

vae_models = {'VanillaVAE':VAE,
              'LSTMVAE':LSTMVAE}



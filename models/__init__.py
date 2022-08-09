from models.base import BaseVAE
from models.rnn_vae import RNNVAE
from models.vanilla_vae import VanillaVAE

# Aliases
VAE = VanillaVAE
RNNVAE = RNNVAE

vae_models = {'VanillaVAE':VAE,
              'RNNVAE':RNNVAE}

__all__ = ["RNNVAE", "VanillaVAE", "BaseVAE", "vae_models"]
__import__("pkg_resources").declare_namespace(__name__)



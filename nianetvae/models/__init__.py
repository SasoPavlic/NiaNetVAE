from nianetvae.models.base import BaseVAE
from nianetvae.models.rnn_vae import RNNVAE
from nianetvae.models.vanilla_vae import VanillaVAE

# Aliases
VAE = VanillaVAE
RNNVAE = RNNVAE

vae_models = {'VanillaVAE':VAE,
              'RNNVAE':RNNVAE}

__all__ = ["RNNVAE", "VanillaVAE", "BaseVAE", "vae_models"]
__import__("pkg_resources").declare_namespace(__name__)



import argparse
import uuid
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from lightning import seed_everything

# Import your data loaders
from nianetvae.dataloaders.ecg_dataloader import ECG5000DataLoader
from nianetvae.dataloaders.kpi_dataloader import KPIDataLoader
from nianetvae.dataloaders.msl_dataloader import MSLDataLoader
from nianetvae.dataloaders.yahoo_dataloader import YahooA1DataLoader

# Set random seed for reproducibility
torch.manual_seed(42)

# Define the VAE class
class VAE(nn.Module):
    def __init__(
        self,
        input_dim,
        latent_dim,
        seq_length,
        rnn_type='LSTM',
        encoder_hidden_dims=None,
        decoder_hidden_dims=None,
        symmetrical=True
    ):
        super(VAE, self).__init__()
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.rnn_type = rnn_type

        # If symmetrical, set decoder_hidden_dims as reversed encoder_hidden_dims
        if symmetrical:
            if encoder_hidden_dims is None:
                raise ValueError("encoder_hidden_dims must be specified when symmetrical is True.")
            self.encoder_hidden_dims = encoder_hidden_dims
            self.decoder_hidden_dims = encoder_hidden_dims[::-1]
        else:
            if encoder_hidden_dims is None or decoder_hidden_dims is None:
                raise ValueError("Both encoder_hidden_dims and decoder_hidden_dims must be specified when symmetrical is False.")
            self.encoder_hidden_dims = encoder_hidden_dims
            self.decoder_hidden_dims = decoder_hidden_dims

        # Build the encoder
        self.encoder_rnn_layers = nn.ModuleList()
        prev_dim = input_dim
        for h_dim in self.encoder_hidden_dims:
            self.encoder_rnn_layers.append(
                self.get_layer_object(
                    input_size=prev_dim,
                    hidden_size=h_dim,
                    num_layers=1,
                    batch_first=True
                )
            )
            prev_dim = h_dim

        # Adjust for the case when encoder_hidden_dims is empty
        if self.encoder_hidden_dims:
            last_encoder_dim = self.encoder_hidden_dims[-1]
        else:
            last_encoder_dim = input_dim  # No hidden layers; use input_dim directly

        # Linear layers for mu and logvar
        self.fc_mu = nn.Linear(last_encoder_dim, latent_dim)
        self.fc_logvar = nn.Linear(last_encoder_dim, latent_dim)

        # Build the decoder
        self.decoder_rnn_layers = nn.ModuleList()

        # Adjust for the case when decoder_hidden_dims is empty
        if self.decoder_hidden_dims:
            first_decoder_dim = self.decoder_hidden_dims[0]
            self.decoder_input = nn.Linear(latent_dim, first_decoder_dim)
        else:
            first_decoder_dim = latent_dim  # No hidden layers; use latent_dim directly
            self.decoder_output = nn.Linear(latent_dim, seq_length * input_dim)

        prev_dim = first_decoder_dim
        for h_dim in self.decoder_hidden_dims[1:] if self.decoder_hidden_dims else []:
            self.decoder_rnn_layers.append(
                self.get_layer_object(
                    input_size=prev_dim,
                    hidden_size=h_dim,
                    num_layers=1,
                    batch_first=True
                )
            )
            prev_dim = h_dim

        if self.decoder_hidden_dims:
            # Final decoder layer to map back to input dimension
            self.final_decoder_rnn = self.get_layer_object(
                input_size=prev_dim,
                hidden_size=input_dim,
                num_layers=1,
                batch_first=True
            )

    def get_layer_object(self, input_size, hidden_size, num_layers, batch_first):
        if self.rnn_type == 'LSTM':
            return nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=batch_first
            )
        elif self.rnn_type == 'GRU':
            return nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=batch_first
            )
        elif self.rnn_type == 'RNN_TANH':
            return nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                nonlinearity='tanh',
                batch_first=batch_first
            )
        else:
            raise ValueError(f"Unsupported rnn_type: {self.rnn_type}")

    def encode(self, x):
        print(f"Input to encoder: {x.shape}")
        if self.encoder_rnn_layers:
            for idx, rnn_layer in enumerate(self.encoder_rnn_layers):
                x, _ = rnn_layer(x)
                print(f"Output of encoder {self.rnn_type} layer {idx}: {x.shape}")
            x_last = x[:, -1, :]
        else:
            # No encoder RNN layers; use the last time step directly
            x_last = x[:, -1, :]
        print(f"Encoder output at last time step: {x_last.shape}")

        mu = self.fc_mu(x_last)
        logvar = self.fc_logvar(x_last)
        print(f"Mu shape: {mu.shape}")
        print(f"LogVar shape: {logvar.shape}")
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        print(f"Sampled latent vector z: {z.shape}")
        return z

    def decode(self, z):
        print(f"Input to decoder (z): {z.shape}")
        if self.decoder_hidden_dims:
            decoder_input = self.decoder_input(z)
            print(f"Decoder input after linear layer: {decoder_input.shape}")
            decoder_input = decoder_input.unsqueeze(1).repeat(1, self.seq_length, 1)
            print(f"Decoder input after unsqueeze and repeat: {decoder_input.shape}")

            x = decoder_input
            for idx, rnn_layer in enumerate(self.decoder_rnn_layers):
                x, _ = rnn_layer(x)
                print(f"Output of decoder {self.rnn_type} layer {idx}: {x.shape}")

            x, _ = self.final_decoder_rnn(x)
            print(f"Output of final decoder {self.rnn_type} layer: {x.shape}")
        else:
            # No decoder RNN layers; use a linear layer to map latent vector to output
            x = self.decoder_output(z)
            print(f"Decoder output after linear layer: {x.shape}")
            x = x.view(-1, self.seq_length, self.input_dim)
            print(f"Decoder output reshaped to: {x.shape}")
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar

# Function to select the appropriate data loader
def select_dataloader(config):
    dataset_type = config["data_params"].get("dataset_type", "")

    dataloader_switch = {
        "YahooA1": YahooA1DataLoader,
        "ECG5000": ECG5000DataLoader,
        "KPI": KPIDataLoader,
        "MSL": MSLDataLoader,
    }

    DataLoaderClass = dataloader_switch.get(dataset_type)

    if DataLoaderClass is None:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    return DataLoaderClass(**config["data_params"])

# Main function
if __name__ == '__main__':
    RUN_UUID = uuid.uuid4().hex
    torch.set_float32_matmul_precision("medium")
    parser = argparse.ArgumentParser(description='Generic runner for VAE models')
    parser.add_argument('--config', '-c',
                        dest="filename",
                        metavar='FILE',
                        help='path to the config file',
                        default='configs/main_config.yaml')

    args = parser.parse_args()

    with open(args.filename, 'r') as file:
        try:
            config = yaml.load(file, Loader=yaml.Loader)
        except yaml.YAMLError as exc:
            print("Error while loading config file")
            print(exc)

    config['logging_params']['save_dir'] += RUN_UUID + '/'

    print(config['logging_params'])
    print(f'Program start: {datetime.now().strftime("%H:%M:%S-%d/%m/%Y")}')
    print(f"RUN UUID: {RUN_UUID}")
    print(f"PyTorch version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"Is CUDA available? {'Yes' if cuda_available else 'No'}")

    print("NiaNetVAE settings")
    print(config)

    seed_everything(config['exp_params']['manual_seed'], True)
    # TODO Check in your previous code how the tensor changes shape through flow
    # Setup data loader
    data_loader = select_dataloader(config)
    data_loader.setup()
    train_loader = data_loader.train_dataloader()

    # Dataset information
    data_shape = train_loader.dataset.data.shape
    if len(data_shape) == 3:
        # Multivariate data
        input_dim = data_shape[2]
        seq_length = data_shape[1]
        is_univariate = False
    elif len(data_shape) == 2:
        # Univariate data
        input_dim = 1
        seq_length = data_shape[1]
        # Reshape data to add an extra dimension
        train_loader.dataset.data = train_loader.dataset.data.unsqueeze(-1)
        is_univariate = True
    else:
        raise ValueError(f"Unsupported data shape: {data_shape}")

    # User-defined parameters or from external functions
    rnn_type = 'LSTM'          # 'LSTM', 'GRU', or 'RNN_TANH'
    symmetrical = False        # True or False
    num_layers = 5             # Number of layers in the encoder
    layer_step = 10            # Difference between hidden sizes

    if is_univariate:
        h_init = 50  # Initial hidden size for univariate data
        # Calculate encoder hidden dimensions
        def calculate_univariate_hidden_dims(h_init, layer_step, num_layers):
            hidden_dims = []
            current_dim = h_init
            for _ in range(num_layers):
                if current_dim <= 0:
                    raise ValueError("layer_step is too large, resulting in non-positive hidden dimensions.")
                hidden_dims.append(current_dim)
                current_dim -= layer_step
            return hidden_dims

        encoder_hidden_dims = calculate_univariate_hidden_dims(h_init, layer_step, num_layers)
        latent_dim = encoder_hidden_dims[-1]
        print(f"Calculated latent_dim: {latent_dim}")
        print("Calculated encoder_hidden_dims:", encoder_hidden_dims)

        # Calculate decoder hidden dimensions
        if symmetrical:
            decoder_hidden_dims = encoder_hidden_dims[::-1]
        else:
            decoder_hidden_dims = encoder_hidden_dims[::-1]
        print("Calculated decoder_hidden_dims:", decoder_hidden_dims)
    else:
        # For multivariate data, calculate as before
        latent_dim = input_dim - layer_step * num_layers
        if latent_dim <= 0:
            raise ValueError("The calculated latent_dim is non-positive. Adjust layer_step or num_layers.")
        print(f"Calculated latent_dim: {latent_dim}")

        # Function to calculate hidden dimensions
        def calculate_hidden_dims(start_dim, end_dim, num_layers):
            if num_layers < 1:
                raise ValueError("Number of layers must be at least 1")
            if num_layers == 1:
                return []
            hidden_dims = []
            step = (start_dim - end_dim) / num_layers
            current_dim = start_dim
            for _ in range(num_layers - 1):
                next_dim = current_dim - step
                if (step > 0 and next_dim <= end_dim) or (step < 0 and next_dim >= end_dim):
                    raise ValueError("Step size is too large, resulting in invalid hidden dimensions.")
                hidden_dims.append(int(round(next_dim)))
                current_dim = next_dim
            return hidden_dims

        encoder_hidden_dims = calculate_hidden_dims(input_dim, latent_dim, num_layers)
        print("Calculated encoder_hidden_dims:", encoder_hidden_dims)

        # Calculate decoder hidden dimensions
        if symmetrical:
            decoder_hidden_dims = encoder_hidden_dims[::-1]
        else:
            num_decoder_layers = num_layers
            decoder_hidden_dims = calculate_hidden_dims(latent_dim, input_dim, num_decoder_layers)
            print("Calculated decoder_hidden_dims:", decoder_hidden_dims)

    # Instantiate the VAE
    model = VAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        seq_length=seq_length,
        rnn_type=rnn_type,
        encoder_hidden_dims=encoder_hidden_dims,
        decoder_hidden_dims=decoder_hidden_dims if not symmetrical else None,
        symmetrical=symmetrical
    )

    # Print the encoder and decoder hidden dimensions
    print("Encoder hidden dimensions:", model.encoder_hidden_dims)
    print("Decoder hidden dimensions:", model.decoder_hidden_dims)

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Loss function
    def loss_function(reconstructed, x, mu, logvar):
        recon_loss = nn.functional.mse_loss(reconstructed, x, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kld_loss

    # Training loop
    num_epochs = 5  # User-defined or external
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            data_batch = batch['signal'].float()
            if data_batch.dim() == 2:
                # Add extra dimension for univariate data
                data_batch = data_batch.unsqueeze(-1)
            optimizer.zero_grad()
            reconstructed_batch, mu, logvar = model(data_batch)
            loss = loss_function(reconstructed_batch, data_batch, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item() / len(data_batch):.4f}")

        total_samples = len(train_loader.dataset) if hasattr(train_loader, 'dataset') else len(train_loader) * train_loader.batch_size
        avg_loss = train_loss / total_samples
        print(f"====> Epoch: {epoch+1} Average loss: {avg_loss:.4f}")

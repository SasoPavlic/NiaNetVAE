import torch
from torch.utils.data import Subset
from models.vanilla_vae import VanillaVAE
import torch.utils
import torch.distributions
import torchvision
import numpy as np
import matplotlib.pyplot as plt;

from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on: {device}")
plt.rcParams['figure.dpi'] = 200
torch.manual_seed(0)


def plot_latent(autoencoder, data, num_batches):
    for i, (x, y) in enumerate(data):
        x = x.view(x.size(0), -1)
        z = autoencoder(x.to(device))
        z = z[1].to('cpu').detach().numpy()
        plt.scatter(z[:, 1], z[:, 0], c=y, cmap='tab10')
        if i > num_batches:
            plt.colorbar()
            break

    plt.show()


def plot_reconstructed(autoencoder, r0=(-5, 10), r1=(-10, 5), n=12):
    w = 28
    img = np.zeros((n * w, n * w))
    for i, y in enumerate(np.linspace(*r1, n)):
        for j, x in enumerate(np.linspace(*r0, n)):
            z = torch.Tensor([[x, y]]).to(device)
            z = z.view(128, 256)
            x_hat = autoencoder.decode(z)
            x_hat = x_hat.reshape(28, 28).to('cpu').detach().numpy()
            img[(n - 1 - i) * w:(n - 1 - i + 1) * w, j * w:(j + 1) * w] = x_hat
    plt.imshow(img, extent=[*r0, *r1])

    plt.show()


def fit(VAE, dataloader):
    VAE.train()
    running_loss = 0.0
    for i, x in tqdm(enumerate(dataloader), total=int(len(dataset) / dataloader.batch_size)):
        x, _ = x
        x = x.to(device)  # GPU
        optimizer.zero_grad()
        # TODO decouple data size from algorithm
        #x = x.view(x.size(0), -1)
        results = VAE(x)
        #loss_2 = ((results[0] - results[1]) ** 2).sum() + VAE.kl
        loss = VAE.loss_function(results[0], results[1], results[2], results[3], M_N=VAE.kl)
        loss = loss['loss']
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    train_loss = running_loss / len(dataloader.dataset)
    return train_loss


if __name__ == '__main__':

    dataset = Subset(torchvision.datasets.MNIST('./data',
                                                transform=torchvision.transforms.ToTensor(),
                                                download=True), list(range(1, 60000)))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    latent_dims = 2
    VAE = VanillaVAE(784, 128, 784).to(device)  # GPU
    optimizer = torch.optim.Adam(VAE.parameters())

    train_loss = []
    val_loss = []
    epochs = 3
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")
        train_epoch_loss = fit(VAE, dataloader)
        print(f"Loss: {train_epoch_loss}\n")

    # vae = train(vae, dataloader)
    image_index = 17
    first_image = dataset.dataset.test_data[image_index]
    first_image = np.array(first_image, dtype='float')
    pixels = first_image.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show()

    batch = dataset.dataset.test_data[:128]
    # https://stackoverflow.com/questions/64635630/pytorch-runtimeerror-expected-scalar-type-float-but-found-byte
    batch = batch.view(batch.size(0), -1) / 255
    batch = batch.to(device)
    first_image = VAE.generate(batch)
    first_image = first_image[image_index, :].to('cpu').detach().numpy()
    first_image = np.array(first_image, dtype='float')
    pixels = first_image.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show()

    # plot_latent(VAE, dataloader, dataloader.batch_size)
    # plot_reconstructed(VAE, r0=(-3, 3), r1=(-3, 3))

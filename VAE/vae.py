import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision
from torchvision.models import vgg16
import cv2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vgg = vgg16(pretrained=True).features[:16].eval().cuda()

transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ]
)

dataset = datasets.ImageFolder("102flowers", transform=transform)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                stride=stride,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)

        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride, bias=False
            )
        else:
            self.shortcut = nn.Identity()

        self.nonLinearity = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.conv1(x))
        out = self.nonLinearity(self.conv2(out) + residual)
        return out


class VAE(nn.Module):
    def encoder_layer(self, in_channels, out_channels, num_blocks):
        strides = [2] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(in_channels, out_channels, stride))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def decoder_layer(self, in_channels, out_channels, num_blocks):
        layers = []
        layers.append(nn.Upsample(scale_factor=2, mode="nearest"))  # Upsample first
        for stride in range(num_blocks):
            layers.append(ResidualBlock(in_channels, out_channels, 1))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 7, 2, 3),  # 32x128x128
            nn.ReLU(),
            self.encoder_layer(32, 64, 1),  # 64x64x64
            self.encoder_layer(64, 128, 1),  # 128x32x32
            self.encoder_layer(128, 256, 1),  # 256x16x16
            self.encoder_layer(256, 512, 1),  # 512x8x8
            # self.encoder_layer(512, 1024, 1),  # 1024x4x4
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(512 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(512 * 4 * 4, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 512 * 4 * 4)
        self.decoder = nn.Sequential(
            # self.decoder_layer(1024, 512, 1),  # 512x8x8
            self.decoder_layer(512, 256, 1),  # 256x16x16
            self.decoder_layer(256, 128, 1),  # 128x32
            self.decoder_layer(128, 64, 1),  # 64x64x64
            self.decoder_layer(64, 32, 1),  # 32x128x128
            nn.Upsample(scale_factor=2, mode="nearest"),  # Upsample to 256x256
            nn.Conv2d(32, 3, 7, 1, 3),  # Output layer: 3x256x256
            nn.Sigmoid(),  # Sigmoid activation for pixel values
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, 512, 4, 4)
        return self.decoder(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


vgg_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
vgg_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()


def normalize_vgg(x):
    return (x - vgg_mean) / vgg_std


def perceptual_loss(x_recon, x):
    x_recon_norm = normalize_vgg(x_recon)
    x_norm = normalize_vgg(x).detach()
    f1 = vgg(x_recon_norm)
    f2 = vgg(x_norm)
    return F.l1_loss(f1, f2)


def loss_function(recon_x, x, mu, logvar, beta):
    recon_loss = F.mse_loss(recon_x, x, reduction="sum")
    perceptual_loss_value = perceptual_loss(recon_x, x)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return 0.5 * recon_loss + beta * kl_div + perceptual_loss_value * 0.5


vae = VAE(latent_dim=128).cuda()
optimizer = Adam(vae.parameters(), lr=1e-4)


def save_checkpoint(model, optimizer, epoch, filename="vae_checkpoint.pth"):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        filename,
    )


def load_checkpoint(model, optimizer, filename="vae_checkpoint.pth"):
    try:
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint["epoch"]
    except FileNotFoundError:
        print(f"Checkpoint file {filename} not found. Starting from scratch.")
        return 0
    except Exception as e:
        print(f"Error loading checkpoint: {e}. Starting from scratch.")
        return 0


ep = load_checkpoint(vae, optimizer)

for epoch in range(ep, ep + 2500):
    vae.train()
    total_loss = 0
    count = 0
    bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    for x, _ in bar:
        x = x.cuda()
        optimizer.zero_grad()
        x_recon, mu, logvar = vae(x)
        loss = loss_function(x_recon, x, mu, logvar, beta=4.0)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        count += x.size(0)

        bar.set_description(f"Epoch {epoch+1}, Loss: {total_loss/count:.4f}")

    save_checkpoint(vae, optimizer, epoch)

    vae.eval()
    with torch.no_grad():
        z = torch.randn(16, 128).cuda()
        samples = vae.decode(z).cpu()

    grid = torchvision.utils.make_grid(samples, nrow=4)
    cv2.imwrite(
        f"vae_samples_epoch_{epoch+1}.png",
        grid.permute(1, 2, 0)[..., [2, 1, 0]].numpy() * 255,
    )
    # plt.figure(figsize=(8,8))
    # plt.imshow(grid.permute(1, 2, 0))
    # plt.title("Generated Flowers from VAE")
    # plt.axis('off')
    # plt.show()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataset):.4f}")

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from tqdm import tqdm
import lpips
from torch.utils.tensorboard import SummaryWriter


# Load the dataset from flowersSquared folder
class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transformInput=None, transformOutput=None):
        self.root_dir = root_dir
        self.transformInput = transformInput
        self.transformOutput = transformOutput
        self.image_files = [
            f
            for f in os.listdir(root_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")

        if self.transformInput:
            imageInput = self.transformInput(image)

        if self.transformOutput:
            imageOutput = self.transformOutput(image)

        return imageInput, imageOutput  # Return both inputs and outputs


def get_dataloader(inputSize=128, outputSize=256, batch_size=32):
    transformInput = transforms.Compose(
        [
            transforms.Resize((inputSize, inputSize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    transformOutput = transforms.Compose(
        [
            transforms.Resize((outputSize, outputSize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    root_dir = os.path.join(
        os.path.dirname(__file__), "../flower_dataset/flowersSquared"
    )

    dataset = CustomImageDataset(
        root_dir=root_dir,
        transformInput=transformInput,
        transformOutput=transformOutput,
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, padding=None):
        """Initialisiert einen ResNet-Block mit zwei Convolutional-Schichten, Batch-Normalisierung und ReLU-Aktivierung.

        Parameters:
        -----------

        in_channels (int):
          Anzahl der Eingabekanäle.

        out_channels (int):
            Anzahl der Ausgabekanäle.

        kernel_size (int, optional):
            Größe des Convolutional-Kernels. Standard ist 9.

        padding (int, optional):
            Padding für die Convolutional-Schichten. Standard ist None. In dem Fall wird das Padding automatisch berechnet, so dass die Ausgabe die gleiche Größe wie die Eingabe hat.
        """
        super(ResNetBlock, self).__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(self.relu(out))
        out = self.conv2(out)
        out = out + residual
        return self.bn2(self.relu(out))


def save_checkpoint(model, optimizer, epoch, filename="checkpoint.pth"):
    """Speichert den aktuellen Zustand des Modells und des Optimierers in einer Datei."""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        filename,
    )


def load_checkpoint(model, optimizer, filename="checkpoint.pth"):
    """Lädt den Zustand des Modells und des Optimierers aus einer Datei."""
    try:
        checkpoint = torch.load(filename, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint["epoch"]
    except Exception as e:
        print(f"Fehler beim Laden des Checkpoints {filename}: {e}")
        print("Starte ohne gespeicherten Zustand.")
        return 0


def log_metrics(
    writer, epoch, total_loss, total_lips, total_mse, total_psnr, total_cnt
):
    avg_loss = total_loss / total_cnt
    avg_lips = total_lips / total_cnt
    avg_mse = total_mse / total_cnt
    avg_psnr = total_psnr / total_cnt
    writer.add_scalar("Loss", 1000.0 * avg_loss, epoch)
    writer.add_scalar("LPIPS", 1000.0 * avg_lips, epoch)
    writer.add_scalar("MSE", 1000.0 * avg_mse, epoch)
    writer.add_scalar("PSNR", avg_psnr, epoch)


def log_images(writer, model, dataloader, epoch):
    model.eval()
    with torch.no_grad():
        stiches = []
        for i, (input, target) in enumerate(dataloader):
            if i >= 3:  # Log only first 3 images
                break
            input = input.cuda()
            target = target.cuda()
            output = model(input)

            # Denormalize images
            def denormalize(tensor):
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).cuda()
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).cuda()
                return tensor * std + mean

            # Rescale to 256x256 and stitch together
            input_resized = torch.nn.functional.interpolate(
                input[0:1], size=(256, 256), mode="bilinear", align_corners=False
            )
            output_resized = torch.nn.functional.interpolate(
                output[0:1], size=(256, 256), mode="bilinear", align_corners=False
            )
            target_resized = torch.nn.functional.interpolate(
                target[0:1], size=(256, 256), mode="bilinear", align_corners=False
            )

            input_norm = denormalize(input_resized[0]).clamp(0, 1)
            output_norm = denormalize(output_resized[0]).clamp(0, 1)
            target_norm = denormalize(target_resized[0]).clamp(0, 1)

            # Stitch images horizontally
            stitched = torch.cat([input_norm, output_norm, target_norm], dim=2)
            stiches.append(stitched)

        # Convert to grid and log
        stitched = torch.cat(stiches, dim=1)
        writer.add_image(f"Images", stitched, epoch)
    model.train()


class PSNR(nn.Module):
    def __init__(self, max_val=1.0):
        super(PSNR, self).__init__()
        self.max_val = max_val

    def forward(self, output, target):
        mse = F.mse_loss(output, target)
        psnr = 20 * torch.log10(self.max_val / torch.sqrt(mse))
        return psnr


# Denormalize images
def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).cuda()
    return tensor * std + mean


def train(prefix, model, dataloader, loss_fn):
    print(f"Training {prefix} model...")

    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    metric = lpips.LPIPS(net="vgg").cuda()  # Using SqueezeNet for perceptual loss
    mseMetric = nn.MSELoss()
    psnrMetric = PSNR(max_val=6.0)

    ep = load_checkpoint(model, optim, filename=f"{prefix}.pt")

    writer = SummaryWriter(f"runs/{prefix}")

    for epoch in range(ep, ep + 30):
        total_loss = 0.0
        total_lips = 0.0
        total_mse = 0.0
        total_psnr = 0.0
        total_cnt = 0
        bar = tqdm(dataloader)
        for batch in bar:
            input, target = batch
            input = input.cuda()
            target = target.cuda()

            optim.zero_grad()
            output = model(input)
            loss = loss_fn(output, target)
            loss.backward()
            optim.step()

            total_loss += loss.item()
            total_cnt += 1  # input.size(0)

            total_lips += (
                metric(2.0 * denormalize(output) - 1.0, 2.0 * denormalize(target) - 1.0)
                .mean()
                .item()
            )
            total_mse += mseMetric(output, target).item()
            total_psnr += psnrMetric(output, target).item()

            bar.set_description(
                f"[{epoch+1}], Loss: {1000.0 * total_loss / total_cnt:.3f}, LPIPS: {total_lips / total_cnt:.3f}, MSE: {total_mse / total_cnt:.3f}, PSNR: {total_psnr / total_cnt:.3f}"
            )

        log_metrics(
            writer, epoch + 1, total_loss, total_lips, total_mse, total_psnr, total_cnt
        )
        log_images(writer, model, dataloader, epoch + 1)
        save_checkpoint(model, optim, epoch + 1, filename=f"{prefix}.pt")

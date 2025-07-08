import torch
from torch import nn
from torchvision.models import vgg16


class VGG16PerceptualLoss(nn.Module):
    def __init__(self):
        """Initialize the VGG16 perceptual loss model.

        It computes the perceptual loss as the mean squared error between the features.

        The model is set to evaluation mode and the parameters are frozen.

        **TODO**:

        - Load the VGG16 model with pretrained weights. Use `torchvision.models.vgg16(pretrained=True)`.

        - Restrict the VGG16 model to the first 16 layers by using `self.vgg = vgg16(pretrained=True).features[:16]`.

        - Set the model to evaluation mode using `.eval()`.

        - Freeze the parameters of the VGG16 model by setting `param.requires_grad = False` for all parameters.
          NOTE: Iterate through all parameters by using the `self.vgg.parameters()`-Iterator.

        - Initialize the L2 loss function using `nn.MSELoss()`.
        """
        super(VGG16PerceptualLoss, self).__init__()
        self.vgg = vgg16(pretrained=True).features[:16].eval().cuda()

        for param in self.vgg.parameters():
            param.requires_grad = False

        self.l1_loss = nn.L1Loss()

    def forward(self, output, target):
        """Compute the perceptual loss between two images.

        Parameters:
        -----------
            output (torch.Tensor):
              The output image tensor from the upscaler network.

            target (torch.Tensor):
              The target image tensor from ground truth.

        Returns:
        --------
            torch.Tensor:
              The computed perceptual loss as the mean squared error between the features of the two images.

        **TODO**:

        - Resize `output` and `target` to 224x224 using `torch.nn.functional.interpolate()`. Use `mode='bilinear'` and `align_corners=False`.

        - Pass `output` through the VGG16 model to get the features `f1`.

        - Pass `target` through the VGG16 model to get the features `f2`. Note: You should use `torch.no_grad()` to avoid computing gradients for the target image.

        - Compute and return the L2 loss between `f1` and `f2` using `self.l2_loss(f1, f2)`.
        """
        f1 = self.vgg(output)

        with torch.no_grad():
            f2 = self.vgg(target)

        return self.l1_loss(f1, f2)


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

        # img_paths = [os.path.join(self.root_dir, self.image_files[idx]) for idx in range(len(self.image_files))]
        # self.images = [Image.open(img_path).convert("RGB") for img_path in img_paths]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        # image = self.images[idx]

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

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, pin_memory=True  # , num_workers=8
    )

    return dataloader


class ResNetBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=9,
        padding=None,
        stride=1,
        norm=True,
    ):
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
            stride=stride,
        )
        self.bn1 = (
            nn.BatchNorm2d(out_channels) if norm else nn.Identity()
        )  # nn.InstanceNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )
        self.bn2 = (
            nn.BatchNorm2d(out_channels) if norm else nn.Identity()
        )  # nn.InstanceNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, bias=False, stride=stride
                ),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.relu(self.bn1(out))
        out = self.conv2(out)
        out = out + residual
        return self.relu(self.bn2(out))


def save_checkpoint(checkpoint_dict, epoch, filename="checkpoint.pth"):
    """Speichert den aktuellen Zustand des Modells und des Optimierers in einer Datei."""

    checkpoint_dict = {
        key: value.state_dict() for key, value in checkpoint_dict.items()
    }

    checkpoint_dict["epoch"] = epoch

    torch.save(
        checkpoint_dict,
        filename,
    )


def load_checkpoint(checkpoint_dict, filename="checkpoint.pth"):
    """Lädt den Zustand des Modells und des Optimierers aus einer Datei."""
    try:
        checkpoint = torch.load(filename, weights_only=True)

        for key in checkpoint_dict:
            if key in checkpoint:
                checkpoint_dict[key].load_state_dict(checkpoint[key])
            else:
                raise KeyError(f"Key {key} not found in checkpoint.")

        return checkpoint["epoch"]
    except Exception as e:
        print(f"Fehler beim Laden des Checkpoints {filename}: {e}")
        print("Starte ohne gespeicherten Zustand.")
        return 0


def log_metrics(writer, metrics):
    for key, value in metrics.items():
        step = value.step
        if value.count <= 20:
            continue

        writer.add_scalar(key, value.compute(), global_step=step)

    writer.flush()


# Denormalize images
def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).cuda()
    return tensor * std + mean


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

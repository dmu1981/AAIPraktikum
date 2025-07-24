import torch
from torch import nn
from torchvision.models import vgg16
import torchvision

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
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
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

        return tuple(transform(image) for transform in self.transforms)

def get_dataloader(batch_size=32):
    sizes = [64]
    tr = []
    for size in sizes:
        transform = transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        tr.append(transform)

    root_dir = os.path.join(
        os.path.dirname(__file__), "../flower_dataset/flowersSquared"
    )

    dataset = CustomImageDataset(
        root_dir=root_dir,
        transforms=tr
    )

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, pin_memory=True  # , num_workers=8
    )

    return dataloader


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, padding=None, stride=1, norm=True):
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
            stride=stride,
            bias=not norm,
        )
        self.bn1 = nn.BatchNorm2d(out_channels) if norm else nn.Identity()
        self.relu = nn.LeakyReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=not norm,
        )
        self.bn2 = nn.BatchNorm2d(out_channels) if norm else nn.Identity()

        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False, stride=stride),
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


class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, img):
        return torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])) + torch.mean(
            torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])
        )


# Denormalize images
def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).cuda()
    return tensor * std + mean


def log_images(writer, model, dataloader, epoch):
    model.eval()
    with torch.no_grad():
        batch = next(iter(dataloader))
        batch = [b.cuda() for b in batch]


        noise = torch.randn((16, 100, 1, 1), device="cuda")

        sizes = [64]
        for index in [0]:
          output = model(noise)

          # Rescale to 256x256 and stitch together
          output_resized = torch.nn.functional.interpolate(
              output[0:16], size=(64, 64), mode="nearest"#, align_corners=False
          )

          real_resized = torch.nn.functional.interpolate(
              batch[index][0:16], size=(64, 64), mode="nearest"#, align_corners=False
          )

          output_norm = denormalize(output_resized[0:16]).clamp(0, 1)
          real_norm = denormalize(real_resized[0:16]).clamp(0, 1)

          # Stitch images horizontally
          stitched = torchvision.utils.make_grid(output_norm, nrow=4, padding=2)
          writer.add_image(f"Fake_{sizes[index]}", stitched, epoch)

          stitched = torchvision.utils.make_grid(real_norm, nrow=4, padding=2)
          writer.add_image(f"Real{sizes[index]}", stitched, epoch)

#          break
    model.train()


class PSNR(nn.Module):
    def __init__(self, max_val=1.0):
        super(PSNR, self).__init__()
        self.max_val = max_val

    def forward(self, output, target):
        mse = F.mse_loss(output, target)
        psnr = 20 * torch.log10(self.max_val / torch.sqrt(mse))
        return psnr


class Metric:
    def __init__(self, factor=1.0, abs=False):
        self.total = 0.0
        self.count = 0
        self.step = 0
        self.factor = factor
        self.abs = abs

    def state_dict(self):
        return {
            "total": self.total,
            "count": self.count,
            "step": self.step,
            "factor": self.factor,
            "abs": self.abs,
        }

    def load_state_dict(self, state_dict):
        self.total = state_dict["total"]
        self.count = state_dict["count"]
        self.step = state_dict["step"]
        self.factor = state_dict["factor"]
        self.abs = state_dict["abs"]

    def update(self, value):
        self.total += value
        self.count += 1
        self.step += 1

    def compute(self, reset=True):
        if self.count == 0:
            avg = 0.0
        else:
            avg = self.total / self.count

        if reset:
            self.total = 0.0
            self.count = 0

        if self.abs:
            avg = abs(avg)

        return self.factor * avg


def train(prefix, trainer, dataloader):
    print(f"Training {prefix} model...")

    metric = lpips.LPIPS(net="vgg").cuda()  # Using SqueezeNet for perceptual loss
    mseMetric = nn.MSELoss()
    psnrMetric = PSNR(max_val=6.0)

    writer = SummaryWriter(f"runs/{prefix}")

    lpips_score = Metric()
    mse_score = Metric()
    psnr_score = Metric()
    content_loss_score = Metric()
    loss_c_score = Metric(abs=True)
    generator_loss_score = Metric()
    adversarial_loss_score = Metric(abs=True)
    gradient_norm_score = Metric()
    generator_gradient_norm_score = Metric()

    scores = {
        "LPIPS": lpips_score,
        "MSE": mse_score,
        "PSNR": psnr_score,
        "Content": content_loss_score,
        "loss_C": loss_c_score,
        "Generator": generator_loss_score,
        "Adversarial": adversarial_loss_score,
        "Critic Gradient Norm": gradient_norm_score,
        "Generator Gradient Norm": generator_gradient_norm_score,
    }

    checkpoint_dict = {
        "generator": trainer.generator,
        "critic": trainer.critic,
        "optimGenerator": trainer.optimGenerator,
        "optimCritic": trainer.optimCritic,
    } | scores

    ep = load_checkpoint(checkpoint_dict, filename=f"{prefix}_checkpoint.pt")

    for epoch in range(ep, ep + 6000):
        bar = tqdm(dataloader)

        for target in bar:
            target = [t.cuda() for t in target]

            result = trainer.train_batch(target, epoch)
            scoresCritic, scoresGenerator = (result["critic"], result["generator"])
            output = scoresGenerator["output"] if scoresGenerator is not None else None

            if scoresCritic is not None:
                gradient_norm_score.update(scoresCritic["gradient_norm"])
                loss_c_score.update(scoresCritic["loss_c"])

            if scoresGenerator is not None:
                generator_gradient_norm_score.update(scoresGenerator["gradient_norm"])
                content_loss_score.update(scoresGenerator["content_loss"])
                adversarial_loss_score.update(scoresGenerator["adversarial_loss"])
                generator_loss_score.update(scoresGenerator["loss"])

                

                bar.set_description(
                    f"[{epoch+1}], Content: {content_loss_score.compute(reset=False):.3f}, loss_C: {loss_c_score.compute(reset=False):.3f}"
                )
                
            log_metrics(writer, scores)

        log_images(writer, trainer.generator, dataloader, epoch + 1)
        save_checkpoint(
            checkpoint_dict,
            epoch + 1,
            filename=f"{prefix}_checkpoint.pt",
        )



class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        # self.model256 = nn.Sequential(
        #     nn.Conv2d(3, 32, kernel_size=9, stride=2, padding=4),
        #     nn.LeakyReLU(inplace=True),  # 32x128x128
        #     nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
        #     nn.LeakyReLU(inplace=True),  # 64x64x64
        #     nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
        #     nn.LeakyReLU(inplace=True),  # 128x32x32
        #     nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
        #     nn.LeakyReLU(inplace=True),  # 256x16x16
        #     nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2),
        #     nn.LeakyReLU(inplace=True),  # 512x8x8
        #     nn.Conv2d(512, 1024, kernel_size=5, stride=2, padding=2),
        #     nn.LeakyReLU(inplace=True),  # 1024x4x4
        #     nn.AvgPool2d(kernel_size=(4, 4)),  # 1024x1x1
        #     nn.Flatten(),
        #     nn.Linear(1024, 1, bias=False),  # Final output layer
        # )

        self.model64 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(inplace=True),  # 32x128x128
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),  # 64x64x64
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),  # 128x32x32
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),  # 256x16x16
            nn.AvgPool2d(kernel_size=(4, 4)),  # 1024x1x1
            nn.Flatten(),
            nn.Linear(256, 1, bias=False),  # Final output layer
        )

        for m in self.modules():
          if isinstance(m, (nn.Conv2d, nn.Linear)):
              nn.init.kaiming_normal_(m.weight, a=0.2)

    def forward(self, x):
       return self.model64(x)


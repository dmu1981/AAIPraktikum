import torch
from torch import nn
from torchvision.models import vgg16
import torch.nn.init as init
from misc import denormalize, get_dataloader, ResNetBlock, VGG16PerceptualLoss, train, TVLoss
import lpips
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class Generator(nn.Module):
    def __init__(self):
        """Initialize the Upscale4x model.

        This model performs 4x upscaling using a series of ResNet blocks and an upsampling layer.

        **TODO**:

        - Call the `__init__` method of the base class `nn.Module`.

        - Define an upsampling layer using `nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True) <https://docs.pytorch.org/docs/stable/generated/torch.nn.Upsample.html>`_.

        - Define a sequential model consisting of:

        - Five `ResNetBlock` layers with 3->16, 16->32, 32->64, 64->128 and 128->256 channels as well as kernel sizes 7.

        - A PixelShuffle layer with an upscale factor of 4.

        - A final convolutional layer with 16 input channels, 3 output channels and kernel size 5 with padding 2.
        """
        super(Generator, self).__init__()

        self.upBilinear = nn.Upsample(
            scale_factor=4, mode="bilinear", align_corners=True
        )

        self.model = nn.Sequential(
            ResNetBlock(3, 16, kernel_size=7),
            ResNetBlock(16, 32, kernel_size=7),
            ResNetBlock(32, 64, kernel_size=7),
            ResNetBlock(64, 128, kernel_size=7),
            ResNetBlock(128, 256, kernel_size=7),
            nn.PixelShuffle(upscale_factor=4),  # First upsample
            nn.Conv2d(16, 3, kernel_size=7, padding=3),  # Final conv to reduce channels
        )

    def forward(self, x):
        """Perform the forward pass of the Upscale2x model.

        Parameters:
        -----------
            x (torch.Tensor):
              The input tensor to be upscaled.

        Returns:
        --------
            torch.Tensor:
              The upscaled output tensor.

        **TODO**:

        - Pass the input tensor through the model.

        - Also, apply the upsampling layer to the input tensor `x`.

        - Add the upsampled tensor to the output of the model.
        """
        x = self.upBilinear(x) + self.model(x)

        return x


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=9, stride=2, padding=4),
            nn.LeakyReLU(inplace=True),  # 16x128x128
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),  # 32x64x64
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),  # 64x32x32
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),  # 128x16x16
            nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),  # 256x8x8
            nn.Conv2d(512, 1024, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),  # 512x4x4
            nn.AvgPool2d(kernel_size=(4, 4)),  # 1024x1x1
            nn.Flatten(),
            nn.Linear(1024, 1, bias=False),  # Final output layer
        )

    def forward(self, x):
        return self.model(x)





class GeneratorLoss(nn.Module):
    def __init__(self, critic):
        super(GeneratorLoss, self).__init__()
        self.perceptualLoss = VGG16PerceptualLoss()
        self.mseLoss = nn.MSELoss()
        self.tvLoss = TVLoss()
        self.critic = critic

    def forward(self, output, target, epoch):
        adversarial_loss = -torch.tanh(0.01 * self.critic(output)).mean()

        if epoch < 5:
            adversarial_lambda = 0.0
        else:
            adversarial_lambda = min(1.0, (epoch - 3) / 5.0)

        content_loss = (
            self.perceptualLoss(output, target)
            #+ 2.0 * self.mseLoss(output, target)
            + 0.1 * self.tvLoss(output)
        )

        return (
            content_loss + adversarial_lambda * adversarial_loss,
            content_loss,
            adversarial_loss,
        )


class CriticLoss(nn.Module):
    def __init__(self):
        super(CriticLoss, self).__init__()

    def compute_gradient_penalty(self, critic, real, fake, lambda_gp=30):
        batch_size = real.size(0)
        epsilon = torch.rand(batch_size, 1, 1, 1, device=real.device)
        interpolated = (epsilon * real + (1 - epsilon) * fake).requires_grad_(True)

        critic_output = critic(interpolated)
        grad_outputs = torch.ones_like(critic_output)

        gradients = torch.autograd.grad(
            outputs=critic_output,
            inputs=interpolated,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1) ** 2).mean() * lambda_gp
        return gradient_penalty, gradient_norm

    def forward(self, critic, real, fake):
        gp, gradient_norm = self.compute_gradient_penalty(critic, real, fake)

        loss_c = -critic(real).mean() + critic(fake).mean()
        return loss_c + gp, gradient_norm, loss_c


class UpscaleTrainer:
    def __init__(self):
        self.criticUpdates = 0
        self.generator = Generator().cuda()
        self.critic = Critic().cuda()

        self.generatorLoss = GeneratorLoss(self.critic).cuda()
        self.criticLoss = CriticLoss().cuda()

        self.optimGenerator = torch.optim.Adam(self.generator.parameters(), lr=0.005)
        self.optimCritic = torch.optim.Adam(self.critic.parameters(), lr=0.001)

        # Count and print parameters
        gen_params = sum(p.numel() for p in self.generator.parameters())
        critic_params = sum(p.numel() for p in self.critic.parameters())
        print(f"Generator parameters: {gen_params:,}")
        print(f"Critic parameters: {critic_params:,}")

    def train_critic(self, input, target):
        output = self.generator(input)

        self.optimCritic.zero_grad()
        critic_loss, gradient_norm, loss_c = self.criticLoss(
            self.critic, target, output
        )
        critic_loss.backward()
        self.optimCritic.step()

        return {
            "gradient_norm": gradient_norm.mean().item(),
            "loss_c": loss_c.item(),
        }

    def train_generator(self, input, target, epoch):
        self.optimGenerator.zero_grad()
        output = self.generator(input)

        loss, content_loss, adversarial_loss = self.generatorLoss(output, target, epoch)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
        gen_norm = torch.nn.utils.clip_grad_norm_(
            self.generator.parameters(), max_norm=1e9
        )

        self.optimGenerator.step()

        return {
            "loss": loss.item(),
            "content_loss": content_loss.item(),
            "adversarial_loss": adversarial_loss.mean().item(),
            "gradient_norm": gen_norm.item(),
        }, output

    def train_batch(self, input, target, epoch):
        # Train Critic every step
        scoresCritic = self.train_critic(input, target)
        self.criticUpdates += 1

        # Train Generator only every 4th step
        if self.criticUpdates == 4 or epoch < 4:
            scoresGenerator, output = self.train_generator(input, target, epoch)
            self.criticUpdates = 0
        else:
            scoresGenerator = None
            output = None

        return scoresCritic, scoresGenerator, output


if __name__ == "__main__":
    prefix = "upscale4x_mse"

    dataloader = get_dataloader(inputSize=64, outputSize=256, batch_size=48)

    trainer = UpscaleTrainer()

    train(prefix, trainer, dataloader)

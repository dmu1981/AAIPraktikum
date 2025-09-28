import torch
from torch import nn
from torchvision.models import vgg16
import torch.nn.init as init
from misc import (
    get_dataloader,
    ResNetBlock,
    VGG16PerceptualLoss,
    train,
    TVLoss,
)
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
        """Initialize the Critic model.

        This model is a convolutional neural network that takes an image as input and outputs a single score indicating the quality of the image.

        **TODO**:

        - Call the `__init__` method of the base class `nn.Module`.

        - Define a sequential model consisting of:
            - A convolutional layer with 3 input channels, 32 output channels, kernel size 9, stride 2, and padding 4.
            - A LeakyReLU activation function with an inplace operation.
            - A convolutional layer with 32 input channels, 64 output channels, kernel size 5, stride 2, and padding 2.
            - A LeakyReLU activation function with an inplace operation.
            - A convolutional layer with 64 input channels, 128 output channels, kernel size 5, stride 2, and padding 2.
            - A LeakyReLU activation function with an inplace operation.
            - A convolutional layer with 128 input channels, 256 output channels, kernel size 5, stride 2, and padding 2.
            - A LeakyReLU activation function with an inplace operation.
            - A convolutional layer with 256 input channels, 512 output channels, kernel size 5, stride 2, and padding 2.
            - A LeakyReLU activation function with an inplace operation.
            - A convolutional layer with 512 input channels, 1024 output channels, kernel size 5, stride 2, and padding 2.
            - A LeakyReLU activation function with an inplace operation.
            - An average pooling layer with kernel size (4, 4) to reduce the spatial dimensions.
            - A flattening layer to convert the output to a 1D tensor.
            - A linear layer with 1024 input features and 1 output feature (no bias).
        """
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=9, stride=2, padding=4),
            nn.LeakyReLU(inplace=True),  # 32x128x128
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),  # 64x64x64
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),  # 128x32x32
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),  # 256x16x16
            nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),  # 512x8x8
            nn.Conv2d(512, 1024, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),  # 1024x4x4
            nn.AvgPool2d(kernel_size=(4, 4)),  # 1024x1x1
            nn.Flatten(),
            nn.Linear(1024, 1, bias=False),  # Final output layer
        )

    def forward(self, x):
        """
        Perform the forward pass of the Critic model.
        Parameters:
        -----------
            x (torch.Tensor):
              The input tensor to be processed by the Critic model.

        Returns:
        --------
            torch.Tensor: The output score from the Critic model.

        **TODO**:

        - Pass the input tensor through the model.

        - Return the output score from the model.
        """
        return self.model(x)


class GeneratorLoss(nn.Module):
    def __init__(self, critic):
        """Initialize the GeneratorLoss module.

        Parameters:
        -----------
            critic (nn.Module):
              The critic model used for adversarial loss computation.

        **TODO**:

        - Call the `__init__` method of the base class `nn.Module`.

        - Initialize the `VGG16PerceptualLoss` for perceptual loss computation.

        - Initialize the `TVLoss` for total variation loss computation.

        - Store the critic model for adversarial loss computation.
        """
        super(GeneratorLoss, self).__init__()
        self.perceptualLoss = VGG16PerceptualLoss()
        self.tvLoss = TVLoss()
        self.critic = critic

    def forward(self, output, target, epoch):
        """Compute the generator loss.

        The generator loss is a combination of perceptual loss, total variation loss, and adversarial loss.

        The sum of the perceptual loss and total variation loss is called content loss as it is used to measure the quality of the generated image in terms
        of content similarity to the target image.

        The adversarial loss is computed using the critic model, which is trained to distinguish between real and generated images.
        The generator aims to maximize the critic's output for generated images, thus it tries to fool the critic. Mathematically, this is achieved by negating
        the critic's output.

        Since the critic is not yet fully trained during the initial epochs, we apply a linear scaling factor to the adversarial loss based on the current epoch.
        This allows the generator to focus more on content loss in the early stages of training and gradually increase the importance of adversarial loss as
        training progresses. In the first epoch, the adversarial loss is not applied at all, and it starts to increase linearly until it reaches its full weight at epoch 5 epoch.

        The generator shall minimize the content loss while maximizing the adversarial loss, which is achieved by negating the critic's output.


        Parameters:
        -----------
            output (torch.Tensor):
              The output tensor from the generator.

            target (torch.Tensor):
              The target tensor for comparison.

            epoch (int):
              The current training epoch.

        Returns:
        --------
            Dictionary with the following keys:

            - "generator_loss": The total generator loss, which includes perceptual loss, TV loss, and adversarial loss.

            - "content_loss": The content loss (perceptual loss).

            - "adversarial_loss": The adversarial loss computed from the critic.

        **TODO**:

        - Compute the adversarial loss by running the generator images through the critic and **taking the mean**. Then scale it by 0.01.

        - Compute the linear scaling factor for the adversarial loss based on the current epoch. The scaling factor should be 0 in the first epoch and increase linearly to 1 by epoch 5.

        - Compute the content loss as the sum of perceptual loss and TV loss. Scale the TV loss by 0.1 to reduce its impact on the total loss.

        - Compute the total generator loss as the sum of content loss and the **negative** adversarial loss scaled by the linear scaling factor.

        - Return a dictionary containing the total generator loss, content loss, and adversarial loss.
        """
        adversarial_loss = 0.01 * self.critic(output).mean()

        adversarial_lambda = min(1.0, epoch / 5.0)

        content_loss = self.perceptualLoss(output, target) + 0.1 * self.tvLoss(output)

        return {
            "generator_loss": content_loss - adversarial_lambda * adversarial_loss,
            "content_loss": content_loss,
            "adversarial_loss": adversarial_loss,
        }


class CriticLoss(nn.Module):
    def __init__(self, critic):
        """Initialize the CriticLoss module.
        This module computes the loss for the critic model, including the gradient penalty to enforce the Lipschitz constraint.
        """
        super(CriticLoss, self).__init__()
        self.critic = critic

    def compute_gradient_penalty(self, real, fake, lambda_gp=30):
        """Compute the gradient penalty for the critic.
        This function calculates the gradient penalty to enforce the Lipschitz constraint on the critic model.
        """

        # Generate random interpolation between real and fake images
        batch_size = real.size(0)
        epsilon = torch.rand(batch_size, 1, 1, 1, device=real.device)
        interpolated = (epsilon * real + (1 - epsilon) * fake).requires_grad_(True)

        # Compute the critic's output for the interpolated images
        critic_output = self.critic(interpolated)

        # Compute the gradients of the critic's output with respect to the interpolated images
        grad_outputs = torch.ones_like(critic_output)

        gradients = torch.autograd.grad(
            outputs=critic_output,
            inputs=interpolated,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # Compute the gradient penalty
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1) ** 2).mean() * lambda_gp

        # Return the gradient penalty and the gradient norm for logging purposes
        return gradient_penalty, gradient_norm

    def forward(self, real, fake):
        """Compute the critic loss, including the gradient penalty.
        Parameters:
        -----------
            real (torch.Tensor):
              The real images from the dataset.

            fake (torch.Tensor):
              The generated images from the generator model.

        Returns:
        --------
            Dictionary with the following keys:

            - "loss_c": The total critic loss, which includes the WGAN loss and the gradient penalty (torch.Tensor).

            - "gradient_norm": The gradient norm for logging purposes (torch.Tensor).

            - "pure_wgan_loss": The pure WGAN loss (without gradient penalty) for logging purposes (torch.Tensor).

        **TODO**:
            - Calculate the WGAN loss as the difference between the **mean** critic score for real images and the **mean** critic score for fake images.

            - Compute the gradient penalty using the `compute_gradient_penalty` method. Note: This method returns both the gradient penalty and the gradient norm.

            - Return the total critic loss, gradient norm, and pure WGAN loss.
        """
        gp, gradient_norm = self.compute_gradient_penalty(real, fake)

        loss_c = -self.critic(real).mean() + self.critic(fake).mean()

        return {
            "loss_c": loss_c + gp,
            "gradient_norm": gradient_norm,
            "pure_wgan_loss": loss_c,
        }


class UpscaleTrainer:
    def __init__(self):
        self.criticUpdates = 0
        self.generator = Generator().cuda()
        self.critic = Critic().cuda()

        self.generatorLoss = GeneratorLoss(self.critic).cuda()
        self.criticLoss = CriticLoss(self.critic).cuda()

        self.optimGenerator = torch.optim.Adam(self.generator.parameters(), lr=0.0005)
        self.optimCritic = torch.optim.Adam(self.critic.parameters(), lr=0.0001)

        # Count and print parameters
        gen_params = sum(p.numel() for p in self.generator.parameters())
        critic_params = sum(p.numel() for p in self.critic.parameters())
        print(f"Generator parameters: {gen_params:,}")
        print(f"Critic parameters: {critic_params:,}")

    def train_critic(self, input, target):
        """Train the critic model on a batch of input and target images.

        Parameters:
        -----------
            input (torch.Tensor):
              The input tensor containing the images to be processed by the generator.

            target (torch.Tensor):
              The target tensor containing the ground truth images for comparison.

        Returns:
        --------
            dict: A dictionary containing the gradient norm and the critic loss with the following keys:.

                "gradient_norm":  The gradient norm computed during the training of the critic (float).
                "loss_c": The critic loss computed during the training (float).

        **TODO**:

        - Pass the input tensor through the generator to obtain the generator output.

        - Zero the gradients of the critic optimizer (self.optimCritic).

        - Compute the critic loss using the `CriticLoss` module, which includes the WGAN loss and the gradient penalty.
          Store the gradient norm and the critic loss for later so you can return it.

        - Backpropagate the critic loss to compute the gradients.

        - Clip the gradients of the generator to prevent exploding gradients (use `torch.nn.utils.clip_grad_norm_ https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html>`_ with `max_norm=5.0`).

        - Step the critic optimizer to update the critic's parameters.

        - Return a dictionary containing the gradient norm and the critic loss.
        """
        output = self.generator(input)

        self.optimCritic.zero_grad()
        result = self.criticLoss(target, output)
        critic_loss, gradient_norm, loss_c = (
            result["loss_c"],
            result["gradient_norm"],
            result["pure_wgan_loss"],
        )
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=5.0) # THIS IS A BUG, IT SHOULD BE self.critic.parameters()
        self.optimCritic.step()

        return {
            "gradient_norm": gradient_norm.mean().item(),
            "loss_c": loss_c.item(),
        }

    def train_generator(self, input, target, epoch):
        """Train the generator model on a batch of input and target images.

        Parameters:
        -----------
            input (torch.Tensor):
                The input tensor containing the images to be processed by the generator.

            target (torch.Tensor):
                The target tensor containing the ground truth images for comparison.

            epoch (int):
            The current training epoch, used to scale the adversarial loss.

        Returns:
        --------
            dict: A dictionary containing the total generator loss, content loss, adversarial loss, and gradient norm with the following keys:
                "loss": The total generator loss (float).
                "content_loss": The content loss (float).
                "adversarial_loss": The adversarial loss (float).
                "gradient_norm": The gradient norm (float).
                "output": The output tensor from the generator (torch.Tensor).

        **TODO**:

        - Zero the gradients of the generator optimizer (self.optimGenerator).

        - Pass the input tensor through the generator to obtain the generated upsample image.

        - Compute the generator loss using the `GeneratorLoss` module, which includes perceptual loss, TV loss, and adversarial loss.
          Store the content loss, adversarial loss, and total generator loss for later so you can return it.

        - Backpropagate the total generator loss to compute the gradients.

        - Clip the gradients of the generator to prevent exploding gradients (use `torch.nn.utils.clip_grad_norm_ https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html>`_ with `max_norm=1.0`).

        - Call `torch.nn.utils.clip_grad_norm_` again with `max_norm=1e9` and store the gradient norm for later so you can return it.

        - Step the generator optimizer to update the generator's parameters.

        - Return a dictionary containing the total generator loss, content loss, adversarial loss, the output and the gradient norm.
        """
        self.optimGenerator.zero_grad()
        output = self.generator(input)

        result = self.generatorLoss(output, target, epoch)
        loss, content_loss, adversarial_loss = (
            result["generator_loss"],
            result["content_loss"],
            result["adversarial_loss"],
        )

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
            "output": output.detach() if output is not None else None,
        }

    def train_batch(self, input, target, epoch):
        """Train a batch of images using the critic and generator models.

        Parameters:
        -----------
            input (torch.Tensor):
              The input tensor containing the images to be processed by the generator.

            target (torch.Tensor):
              The target tensor containing the ground truth images for comparison.

            epoch (int):
              The current training epoch, used to scale the adversarial loss.

        Returns:
        --------
            A dictionary containing the scores from the critic and generator models with the following keys:
            - "critic": A dictionary containing the critic scores with keys "gradient_norm" and "loss_c".
            - "generator": A dictionary containing the generator scores with keys "loss", "content_loss", "adversarial_loss", "gradient_norm", and "output".

        **TODO**:

        - Train the critic model using the `train_critic` method with the input and target tensors.

        - Increment the critic updates counter (self.criticUpdates).

        - If the critic updates counter is 5 or the epoch is less than 1,
          train the generator model using the `train_generator` method with the input and target tensors, and the current epoch. Also reset the critic updates counter to 0.

        - If the critic updates counter is not 5 and the epoch is greater than or equal to 1, skip training the generator and set the generator scores to None.

        - Return the critic scores and generator scores (if available)
        """
        # Train Critic every step
        scoresCritic = self.train_critic(input, target)
        self.criticUpdates += 1

        # Train Generator only every 4th step
        if self.criticUpdates == 5 or epoch < 1:
            scoresGenerator = self.train_generator(input, target, epoch)
            self.criticUpdates = 0
        else:
            scoresGenerator = None

        return {"critic": scoresCritic, "generator": scoresGenerator}


if __name__ == "__main__":
    prefix = "upscale4x_adversarialloss"

    dataloader = get_dataloader(inputSize=64, outputSize=256, batch_size=48)

    trainer = UpscaleTrainer()

    train(prefix, trainer, dataloader)

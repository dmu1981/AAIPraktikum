import torch
from torch import nn
from torchvision.models import vgg16
from misc import get_dataloader, train, ResNetBlock
from perceptual import VGG16PerceptualLoss, TVLoss


class Upscale4x(nn.Module):
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
        pass

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
        pass

class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        self.perceptualLoss = VGG16PerceptualLoss()
        self.mseLoss = nn.MSELoss()
        self.tvLoss = TVLoss()

    def forward(self, output, target):
          return self.perceptualLoss(output, target) + 0.1 * self.tvLoss(output)
    
if __name__ == "__main__":
    prefix = "upscale4x_perceptual"

    upscaler = Upscale4x().cuda()
    dataloader = get_dataloader(inputSize=64, outputSize=256, batch_size=16)
    loss = GeneratorLoss().cuda()

    # TODO Aufgabe 3: Use mseLoss instead of perceptualLoss for training
    train(prefix, upscaler, dataloader, loss)

import torch
from torch import nn
from torchvision.models import vgg16
from misc import get_dataloader, train, ResNetBlock
from perceptual import VGG16PerceptualLoss

class Upscale2x(nn.Module):
    def __init__(self):
        """Initialize the Upscale2x model.
        
        This model performs 2x upscaling using a series of ResNet blocks and an upsampling layer.
        
        **TODO**:

        - Call the `__init__` method of the base class `nn.Module`.

        - Define an upsampling layer using `nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True) <https://docs.pytorch.org/docs/stable/generated/torch.nn.Upsample.html>`_.

        - Define a sequential model consisting of:

        - Three `ResNetBlock` layers with 3->64, 64->32 and 32->16 channels as well as kernel sizes 9, 7, and 5 respectively.

        - A final convolutional layer with 16 input channels, 3 output channels and kernel size 5 with padding 2.
        """
        super(Upscale2x, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.model = nn.Sequential(
            ResNetBlock(3, 64, kernel_size=9),
            ResNetBlock(64, 32, kernel_size=7),
            ResNetBlock(32, 16, kernel_size=5),
            nn.Conv2d(16, 3, kernel_size=5, padding=2),
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

        - Apply the upsampling layer to the input tensor `x`.

        - Pass the upsampled tensor through the model.

        - Add the upsampled tensor to the output of the model.  
        """
        up = self.upsample(x)
        x = up + self.model(up)
        return x



if __name__ == "__main__":    
  prefix = "upscale2x_mse"

  upscaler = Upscale2x().cuda()
  dataloader = get_dataloader(inputSize=128, outputSize=256, batch_size=64)
  perceptualLoss = VGG16PerceptualLoss().cuda()
  mseLoss = nn.MSELoss().cuda()

  # TODO Aufgabe 3: Use mseLoss instead of perceptualLoss for training
  train(prefix, upscaler, dataloader, mseLoss)

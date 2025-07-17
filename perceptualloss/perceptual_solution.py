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
        - Pass `output` through the VGG16 model to get the features `f1`.

        - Pass `target` through the VGG16 model to get the features `f2`. Note: You should use `torch.no_grad()` to avoid computing gradients for the target image.

        - Compute and return the L2 loss between `f1` and `f2` using `self.l2_loss(f1, f2)`.
        """
        # output = torch.nn.functional.interpolate(
        #     output, size=(224, 224), mode="bilinear", align_corners=False
        # )
        # target = torch.nn.functional.interpolate(
        #     target, size=(224, 224), mode="bilinear", align_corners=False
        # )

        f1 = self.vgg(output)

        with torch.no_grad():
            f2 = self.vgg(target)

        return self.l1_loss(f1, f2)

class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, img):
        return (
            torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))
            + torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]))
        )
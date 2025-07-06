import torch
from upscale2x import Upscale2x
from upscale4x import Upscale4x
from misc import load_checkpoint, get_dataloader
import os

upscale_mse = Upscale4x().cuda()
load_checkpoint(upscale_mse, None, filename="upscale4x_mse.pt")

upscale_perceptual = Upscale4x().cuda()
load_checkpoint(upscale_perceptual, None, filename="upscale4x_perceptual.pt")

dataloader = get_dataloader(inputSize=64, batch_size=4)

def stitch_images(models, input, target):
    for model in models:
        model.eval()

    with torch.no_grad():
        input = input.cuda()
        target = target.cuda()

        outputs = [model(input) for model in models]

        # Denormalize images
        def denormalize(tensor):
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).cuda()
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).cuda()
            return tensor * std + mean
        # Rescale to 256x256 and stitch together
        input_resized = torch.nn.functional.interpolate(input[0:1], size=(256, 256), mode='bilinear', align_corners=False)
        outputs_resized = [torch.nn.functional.interpolate(output[0:1], size=(256, 256), mode='bilinear', align_corners=False) for output in outputs]
        target_resized = torch.nn.functional.interpolate(target[0:1], size=(256, 256), mode='bilinear', align_corners=False)

        input_norm = denormalize(input_resized[0]).clamp(0, 1)
        outputs_norm = [denormalize(output)[0].clamp(0, 1) for output in outputs_resized]
        target_norm = denormalize(target_resized[0]).clamp(0, 1)

        # Stitch images horizontally
        stitched = torch.cat([input_norm, *outputs_norm, target_norm], dim=2)

        return stitched

for i, (input, target) in enumerate(dataloader):
    if i >= 32:  # Log only first 3 images
                break
    stitched = stitch_images([upscale_mse, upscale_perceptual], input, target)

    # Save or display the stitched images as needed
    # For example, you can convert to PIL and save:
    from torchvision.utils import save_image
    os.makedirs('images', exist_ok=True)
    save_image(stitched, f'images/upscale4x_{i}.png')
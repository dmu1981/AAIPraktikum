
import torch
from torch import nn
from torchvision.models import vgg16
from misc import get_dataloader, ResNetBlock, VGG16PerceptualLoss, PSNR, log_metrics, log_images, save_checkpoint, load_checkpoint
import lpips
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

class Upscale4x(nn.Module):
    def __init__(self):
        super(Upscale4x, self).__init__()
        
        self.upBilinear = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        
        self.model = nn.Sequential(
            ResNetBlock(3, 64, kernel_size=7),
            ResNetBlock(64, 32, kernel_size=7),
            ResNetBlock(32, 32, kernel_size=7),
            nn.Conv2d(32, 3, kernel_size=7, padding=3),  # Final conv to reduce channels
        )

        self.upsample = nn.Sequential(
            nn.Conv2d(3, 16*3, kernel_size=7, padding=3),
            nn.PixelShuffle(upscale_factor=4),  # First upsample
            # Note: This is supposed to be a 4x upsample, so no non-linearity here
        )

    def forward(self, x):
        up = self.upsample(x)
        x = up + self.model(up)
        
        return x
    
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=9, stride=2, padding=4), 
            nn.LeakyReLU(inplace=True), # 16x128x128
      
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2), 
            nn.LeakyReLU(inplace=True), # 32x64x64

            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2), 
            nn.LeakyReLU(inplace=True), # 64x32x32

            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2), 
            nn.LeakyReLU(inplace=True), # 128x16x16

            nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2), 
            nn.LeakyReLU(inplace=True), # 256x8x8

            nn.Conv2d(512, 1024, kernel_size=5, stride=2, padding=2), 
            nn.LeakyReLU(inplace=True), # 512x4x4

            nn.AvgPool2d(kernel_size=(4,4)), # 512x1x1
            nn.LeakyReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, 1, bias=False),  # Final output layer
            #nn.Tanh()
        )

    def _make_layer(self, in_channels, out_channels, kernel_size, num_blocks=1, stride=1):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(ResNetBlock(in_channels, out_channels, kernel_size=kernel_size, stride=s, norm=False))
            in_channels = out_channels
        
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def tv_loss(img):
    return torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])) + \
           torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]))


class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        self.perceptualLoss = VGG16PerceptualLoss()
        self.mseLoss = nn.MSELoss()

    def forward(self, output, target):
        return self.perceptualLoss(output, target) + self.mseLoss(output, target) + 1e-3 * tv_loss(output)

def compute_gradient_penalty(critic, real, fake, lambda_gp=10):
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
        only_inputs=True
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1) ** 2).mean() * lambda_gp
    return gradient_penalty


def train(prefix, generator, critic, dataloader, generator_loss_fn):
  print(f"Training {prefix} model...")

  optimGenerator = torch.optim.Adam(generator.parameters(), lr=0.005)
  optimCritic = torch.optim.Adam(critic.parameters(), lr=0.001)

  metric = lpips.LPIPS(net='vgg').cuda()  # Using SqueezeNet for perceptual loss
  mseMetric = nn.MSELoss()
  psnrMetric = PSNR()

  ep = load_checkpoint(generator, optimGenerator, filename=f"{prefix}_generator.pt")
  ep = max(ep, load_checkpoint(critic, optimCritic, filename=f"{prefix}_critic.pt"))

  writer = SummaryWriter(f"runs/{prefix}")

  # weight_clip = 0.01
  # for p in critic.parameters():
  #       p.data.clamp_(-weight_clip, weight_clip)

  for epoch in range(ep, ep + 300):
      total_loss_gen = 0.0
      total_lips = 0.0
      total_mse = 0.0
      total_psnr = 0.0
      total_cnt = 0
      total_loss_critic_real = 0.0
      total_loss_critic_fake = 0.0
      total_content_loss = 0.0
      total_generator_loss = 0.0
      total_adversarial_loss = 0.0
      bar = tqdm(dataloader)

      for index, (input, target) in enumerate(bar):
          input = input.cuda()
          target = target.cuda()

          

          

          input = input.cuda()
          target = target.cuda()
          with torch.no_grad():
            output = generator(input)

          optimCritic.zero_grad()

          critic_real = critic(target).mean()
          critic_fake = critic(output.detach()).mean()

          
          gp = compute_gradient_penalty(critic, target, output.detach())
          
          critic_output = -critic_real + critic_fake + gp
          total_loss_critic_real += critic_real.item()
          total_loss_critic_fake += critic_fake.item()
          critic_output.backward()
          optimCritic.step()

          # for p in critic.parameters():
          #   p.data.clamp_(-weight_clip, weight_clip)

          # Train Critic on real images, output shall be maximized
          n_critic = 5
          # Train Generator only every 5th step
          if (index+1) % n_critic == 0:
            #Train Generator on its loss
            optimGenerator.zero_grad()
            output = generator(input)
            adversarial_loss = -torch.tanh(0.001 * critic(output))
            
            adversarial_loss = adversarial_loss.mean()
              
            total_adversarial_loss += adversarial_loss.item() * n_critic

            gen_loss = generator_loss_fn(output, target)
            total_content_loss += gen_loss.item()
            #loss = 0.1 * gen_loss + adversarial_loss

            adversarial_lambda = min(1.0, epoch / 10.0)
            loss = gen_loss + adversarial_lambda * adversarial_loss
            total_generator_loss += loss.item() * n_critic
            loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            optimGenerator.step()

            def denormalize(tensor):
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).cuda()
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).cuda()
                return tensor * std + mean
            total_lips += metric(2.0 * denormalize(output) - 1.0, 2.0 * denormalize(target) - 1.0).mean().item()

            total_mse += mseMetric(output, target).item()
            total_psnr += psnrMetric(output, target).item()

            total_cnt += input.size(0)

            
            critic_bias = (total_loss_critic_real + total_loss_critic_fake) / 2.0
            c_real = (total_loss_critic_real - critic_bias) / total_cnt
            c_fake = (total_loss_critic_fake - critic_bias) / total_cnt
            bar.set_description(f"[{epoch+1}], Content: {1000.0 * total_content_loss / total_cnt:.3f}, Adversarial: {1000.0 * total_adversarial_loss / total_cnt:.3f}, C(real): {c_real:.3f}, C(fake): {c_fake:.3f}")

      log_metrics(writer, epoch + 1, total_lips, total_mse, total_psnr, total_loss_critic_real, total_loss_critic_fake, total_generator_loss, total_content_loss, total_cnt)
      log_images(writer, generator, dataloader, epoch + 1)
      save_checkpoint(generator, optimGenerator, epoch + 1, filename=f"{prefix}_generator.pt")
      save_checkpoint(critic, optimCritic, epoch + 1, filename=f"{prefix}_critic.pt")

if __name__ == "__main__":    
  prefix = "upscale4x_mse"

  upscaler = Upscale4x().cuda()
  critic = Critic().cuda()
  dataloader = get_dataloader(inputSize=64, outputSize=256, batch_size=64)
  generatorLoss = GeneratorLoss().cuda()
  
  train(prefix, upscaler, critic, dataloader, generatorLoss)

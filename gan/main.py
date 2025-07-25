import torch
from torch import nn
from torchvision.models import vgg16
import torch.nn.init as init
import torch.nn.functional as F
from misc import (
    get_dataloader,
    ResNetBlock,
    train,
    TVLoss,
    #Critic
)
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter



class Critic4(nn.Module):
    def __init__(self, input_channels=3, mid_channels=1024):
        super(Critic4, self).__init__()

        self.midChannels = mid_channels
        self.down = nn.Sequential(
          nn.Conv2d(input_channels, self.midChannels, kernel_size=4, stride=1, padding=0),
          nn.LeakyReLU(inplace=True)
        )

        self.fc = nn.Sequential(
          nn.Linear(self.midChannels, self.midChannels // 2),
          nn.LeakyReLU(inplace=True),
          nn.Linear(self.midChannels // 2, 1),
        )


    def forward(self, x):
        x = self.down(x)
        x = self.fc(x.view(-1, self.midChannels))

        return x
    
class Critic(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Critic, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(channels_img, features_d, kernel_size=7, stride=1, padding=3),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features_d, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d * 2, 3, 2, 1),
            self._block(features_d * 2, features_d * 4, 3, 2, 1),
            self._block(features_d * 4, features_d * 8, 3, 2, 1),
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return ResNetBlock(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding, 
            stride=stride,
            norm=False
        )
    def forward(self, x):
        return self.disc(x)

def initialize_weights(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.002)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)

# class UpsampleBlock(nn.Module):
#     def __init__(self, mid_channels):
#         super(UpsampleBlock, self).__init__()

#         self.up = nn.Sequential(
#             nn.Conv2d(3, mid_channels, kernel_size=5, padding=2),
#             #nn.BatchNorm2d(mid_channels * 4),
#             nn.LeakyReLU(inplace=True),
#             nn.Conv2d(mid_channels, mid_channels * 4, kernel_size=5, padding=2),
#             #nn.BatchNorm2d(out_channels * 4),
#             nn.LeakyReLU(inplace=True),
#             nn.PixelShuffle(upscale_factor=2),
#             nn.Conv2d(mid_channels, 3, kernel_size=5, padding=2),
#             )

#     def forward(self, x):
#         x = self.up(x)
#         return x

# class LinearNetwork(nn.Module):
#     def __init__(self, input_size, output_size, n_layers=4):
#         super(LinearNetwork, self).__init__()
#         self.input_size = input_size
        
#         layers = [self._layer(input_size, input_size) for _ in range(n_layers - 1)]
#         layers.append(self._layer(input_size, output_size))
#         self.inputNetwork = nn.Sequential(
#             *layers
#         )

#     def _layer(self, input_size, output_size):
#         return nn.Sequential(
#             nn.Linear(input_size, output_size, bias=False),
#             nn.BatchNorm1d(output_size),
#             nn.LeakyReLU(inplace=True)
#         )

#     def forward(self, x):
#         x = x.reshape(-1, self.input_size)
#         x = self.inputNetwork(x)
#         return x 

class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)
    
class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, features_g * 16 * 4 * 4),
            View((-1, features_g * 16, 4, 4)),
            nn.ReLU(),
            self._block(features_g * 16, features_g * 8, 5, 2, 2),
            self._block(features_g * 8, features_g * 4, 5, 2, 2),
            self._block(features_g * 4, features_g * 2, 5, 2, 2),
            self._block(features_g * 2, features_g, 5, 2, 2),
            nn.Conv2d(features_g, channels_img, kernel_size=5, stride=1, padding=2),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        if stride == 1:
            return nn.Sequential(
              ResNetBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=1, norm=True),
              nn.ReLU()
            )
        
        return nn.Sequential(
            ResNetBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=1, norm=True),
            nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, padding=0),
            nn.PixelShuffle(upscale_factor=2),
            nn.ReLU()
        )

    def forward(self, x):
        res = self.net(x)
        return res
    
class GeneratorLoss(nn.Module):
    def __init__(self, critic):
        super(GeneratorLoss, self).__init__()
        self.tvLoss = TVLoss()
        self.critic = critic

    def forward(self, img, epoch):
        #self.critic.eval()
        adversarial_loss = torch.tanh(0.001 * self.critic(img)).mean()

        adversarial_lambda = 1.0# min(1.0, epoch / 5.0)

        content_loss = 0.01 * self.tvLoss(img)

        return {
            "generator_loss": content_loss - adversarial_lambda * adversarial_loss,
            #"generator_loss": -adversarial_lambda * adversarial_loss,
            "content_loss": content_loss,
            "adversarial_loss": adversarial_loss,
        }



class CriticLoss(nn.Module):
    def __init__(self, critic):
        super(CriticLoss, self).__init__()
        self.critic = critic

    def compute_gradient_penalty(self, realTuple, fakeTuple, lambda_gp=10):
        # Generate random interpolation between real and fake images

        batch_size = realTuple[0].size(0)
        epsilon = torch.rand(batch_size, 1, 1, 1, device=realTuple[0].device)

        realTuple = (realTuple[0],)
        fakeTuple = (fakeTuple[0],)
        interpolated = [(epsilon * real + (1 - epsilon) * fake).requires_grad_(True) for real, fake in zip(realTuple, fakeTuple)]

        # Compute the critic's output for the interpolated images
        critic_output = self.critic(interpolated[0])

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
        gp, gradient_norm = self.compute_gradient_penalty(real, fake)
        #self.critic.train()

        loss_c = -self.critic(real[0]).mean() + self.critic(fake).mean()

        return {
            "loss_c": loss_c + gp,
            "gradient_norm": gradient_norm,
            "pure_wgan_loss": loss_c,
        }


class UpscaleTrainer:
    def __init__(self):
        self.criticUpdates = 0
        self.generator = Generator(100, 3, 64).cuda()
        self.critic = Critic(3, 160).cuda()

        initialize_weights(self.generator)
        initialize_weights(self.critic)

        self.generatorLoss = GeneratorLoss(self.critic).cuda()
        self.criticLoss = CriticLoss(self.critic).cuda()

        self.optimGenerator = torch.optim.Adam(self.generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
        self.optimCritic = torch.optim.Adam(self.critic.parameters(), lr=1e-4, betas=(0.5, 0.999))

        # Count and print parameters
        gen_params = sum(p.numel() for p in self.generator.parameters())
        critic_params = sum(p.numel() for p in self.critic.parameters())
        print(f"Generator parameters: {gen_params:,}")
        print(f"Critic parameters: {critic_params:,}")

    def generate(self, batch_size):
        noise = torch.randn((batch_size, 100), device="cuda")
        return self.generator(noise)

    def train_critic(self, fake, target):
        self.optimCritic.zero_grad()
        result = self.criticLoss(target, fake)
        critic_loss, gradient_norm, loss_c = (
            result["loss_c"],
            result["gradient_norm"],
            result["pure_wgan_loss"],
        )
        critic_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=150.0)
        self.optimCritic.step()

        return {
            "gradient_norm": gradient_norm.mean().item(),
            "loss_c": loss_c.item(),
        }

    def train_generator(self, batch_size, epoch):
        self.optimGenerator.zero_grad()
        
        fake = self.generate(batch_size)

        result = self.generatorLoss(fake, epoch)
        loss, content_loss, adversarial_loss = (
            result["generator_loss"],
            result["content_loss"],
            result["adversarial_loss"],
        )

        loss.backward()

        #torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=150.0)
        gen_norm = torch.nn.utils.clip_grad_norm_(
            self.generator.parameters(), max_norm=1e9
        )

        self.optimGenerator.step()

        return {
            "loss": loss.item(),
            "content_loss": content_loss.item(),
            "adversarial_loss": adversarial_loss.mean().item(),
            "gradient_norm": gen_norm.item(),
            "output": fake.detach() if fake is not None else None,
        }

    def train_batch(self, target, epoch):
        # Train Critic every step
        batch_size = target[0].size(0)
        
        
        for _ in range(5):
          # Train Critic
          fake = self.generate(batch_size)
          scoresCritic = self.train_critic(fake, target)
  
        self.criticUpdates += 1

        # Train Generator only every 4th step
        if self.criticUpdates == 5 or True:
            scoresGenerator = self.train_generator(batch_size, epoch)
            self.criticUpdates = 0
        else:
            scoresGenerator = None

        return {"critic": scoresCritic, "generator": scoresGenerator}


if __name__ == "__main__":
    prefix = "gan64"

    dataloader = get_dataloader(batch_size=256)

    trainer = UpscaleTrainer()

    train(prefix, trainer, dataloader)

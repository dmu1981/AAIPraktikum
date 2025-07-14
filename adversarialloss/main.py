import torch
from torch import nn
from torchvision.models import vgg16
import torch.nn.init as init
from misc import (
    denormalize,
    get_dataloader,
    ResNetBlock,
    VGG16PerceptualLoss,
    # VGG16PerceptualLossGPT,
    PSNR,
    Metric,
    log_metrics,
    log_images,
    save_checkpoint,
    load_checkpoint,
)
import lpips
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class Upscale4x(nn.Module):
    def __init__(self):
        super(Upscale4x, self).__init__()

        self.upBilinear = nn.Upsample(
            scale_factor=4, mode="bilinear", align_corners=True
        )

        self.model = nn.Sequential(
            #ResNetBlock(3, 16*3, kernel_size=7),#
            nn.Conv2d(3, 16 * 3, kernel_size=7, padding=3),
            nn.PixelShuffle(upscale_factor=4),  # First upsample
            nn.PReLU(),
            ResNetBlock(3, 64, kernel_size=7),
            #ResNetBlock(64, 64, kernel_size=3),#
            ResNetBlock(64, 32, kernel_size=7),
            #ResNetBlock(32, 32, kernel_size=3),#
            ResNetBlock(32, 32, kernel_size=7),
            #ResNetBlock(32, 3, kernel_size=7),#
            nn.Conv2d(32, 3, kernel_size=7, padding=3),  # Final conv to reduce channels
        )

    def forward(self, x):
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
            nn.AvgPool2d(kernel_size=(4, 4)),  # 512x1x1
            nn.LeakyReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(1024, 1, bias=False),  # Final output layer
        )
        # self.model = nn.Sequential(
        #     nn.Conv2d(3, 16, kernel_size=9, stride=2, padding=4),
        #     nn.LeakyReLU(inplace=True),  # 16x128x128
        #     nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
        #     nn.LeakyReLU(inplace=True),  # 32x64x64
        #     nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
        #     nn.LeakyReLU(inplace=True),  # 64x32x32
        #     nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
        #     nn.LeakyReLU(inplace=True),  # 128x16x16
        #     nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
        #     nn.LeakyReLU(inplace=True),  # 256x8x8
        #     nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
        #     nn.LeakyReLU(inplace=True),  # 512x4x4
        #     nn.AvgPool2d(kernel_size=(4, 4)),  # 512x1x1
        #     #nn.LeakyReLU(inplace=True),
        #     nn.Flatten(),
        #     nn.Linear(512, 1, bias=False),  # Final output layer
        # )

    def forward(self, x):
        return self.model(x)
    
class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, img):
        return 0.1 * (
              torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))
            + torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]))
        )    

class GeneratorLoss(nn.Module):
    def __init__(self, critic):
        super(GeneratorLoss, self).__init__()
        self.perceptualLoss = VGG16PerceptualLoss()
        self.mseLoss = nn.MSELoss()
        self.tvLoss = TVLoss()
        self.critic = critic

    def forward(self, output, target, epoch):
        adversarial_loss = -torch.tanh(0.01 * self.critic(output)).mean()
        adversarial_lambda = min(1.0, epoch / 10.0)

        content_loss = (
            self.perceptualLoss(output, target)
            + 2.0 * self.mseLoss(output, target)
            + self.tvLoss(output)
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

class UpscaleTrainer():
    def __init__(self):
        self.generator = Upscale4x().cuda()
        self.critic = Critic().cuda()

        # Count and print parameters
        gen_params = sum(p.numel() for p in self.generator.parameters())
        critic_params = sum(p.numel() for p in self.critic.parameters())
        print(f"Generator parameters: {gen_params:,}")
        print(f"Critic parameters: {critic_params:,}")

        self.generatorLoss = GeneratorLoss(self.critic).cuda()
        self.criticLoss = CriticLoss().cuda()

        self.optimGenerator = torch.optim.Adam(self.generator.parameters(), lr=0.005)
        self.optimCritic = torch.optim.Adam(self.critic.parameters(), lr=0.001)

    def train_critic(self, input, target):
        output = self.generator(input)

        self.optimCritic.zero_grad()
        critic_loss, gradient_norm, loss_c = self.criticLoss(self.critic, target, output)
        critic_loss.backward()
        self.optimCritic.step()

        return {
            "gradient_norm": gradient_norm.mean().item(),   
            "loss_c": loss_c.item(),
        }

    def train_generator(self, input, target, epoch):
        self.optimGenerator.zero_grad()
        output = self.generator(input)

        loss, content_loss, adversarial_loss = self.generatorLoss(
            output, target, epoch
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
        }, output
   
    def train_batch(self, input, target, index, epoch):
        # Train Critic every step
        scoresCritic = self.train_critic(input, target)

        # Train Generator only every 4th step
        n_critic = 4
        if (index + 1) % n_critic == 0:
            scoresGenerator, output = self.train_generator(input, target, epoch)
        else:
            scoresGenerator = None
            output = None

        return scoresCritic, scoresGenerator, output





def train(prefix, dataloader):
    print(f"Training {prefix} model...")

    metric = lpips.LPIPS(net="vgg").cuda()  # Using SqueezeNet for perceptual loss
    mseMetric = nn.MSELoss()
    psnrMetric = PSNR(max_val=6.0)

    writer = SummaryWriter(f"runs/{prefix}")

    trainer = UpscaleTrainer()

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

    

    for epoch in range(ep, ep + 1000):
        bar = tqdm(dataloader)

        for index, (input, target) in enumerate(bar):
            input = input.cuda()
            target = target.cuda()

            scoresCritic, scoresGenerator, output = trainer.train_batch(input, target, index, epoch)

            if scoresGenerator is not None and output is not None:
                generator_gradient_norm_score.update(scoresGenerator["gradient_norm"])
                content_loss_score.update(scoresGenerator["content_loss"])
                adversarial_loss_score.update(scoresGenerator["adversarial_loss"])
                generator_loss_score.update(scoresGenerator["loss"])

                lpips_score.update(
                    metric(
                        2.0 * denormalize(output) - 1.0, 2.0 * denormalize(target) - 1.0
                    )
                    .mean()
                    .item()
                )

                mse_score.update(mseMetric(output, target).item())
                psnr_score.update(psnrMetric(output, target).item())

                bar.set_description(
                    f"[{epoch+1}], Content: {content_loss_score.compute(reset=False):.3f}, loss_C: {loss_c_score.compute(reset=False):.3f}"
                )

            gradient_norm_score.update(scoresCritic["gradient_norm"])
            loss_c_score.update(scoresCritic["loss_c"])

            log_metrics(writer, scores)

        log_images(writer, trainer.generator, dataloader, epoch + 1)
        save_checkpoint(
            checkpoint_dict,
            epoch + 1,
            filename=f"{prefix}_checkpoint.pt",
        )


if __name__ == "__main__":
    prefix = "upscale4x_mse"

    dataloader = get_dataloader(inputSize=64, outputSize=256, batch_size=48)
    
    train(prefix, dataloader)

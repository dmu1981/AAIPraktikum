import os
import torch
import torch.nn as nn
from sklearn.manifold import TSNE

from scipy.linalg import orthogonal_procrustes


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import PIL.Image
import numpy as np
import umap
import cv2 

from tqdm import tqdm 
from misc import DEVICE, load_data, epoch, load_checkpoint, TensorBoardLogger, save_checkpoint, LR

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out) + residual)
        return out
    
class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()

        # Residual blocks
        self.layer1 = self._make_layer(3, 16, 6, 2)
        self.layer2 = self._make_layer(16, 32, 8, 2)
        self.layer3 = self._make_layer(32, 64, 6, 2)
        self.layer4 = self._make_layer(64, 128, 6, 1)

        # Aveerage pooling and fully connected layer
        self.avgpool = nn.AvgPool2d((4,4))
        self.fc = nn.Linear(128, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(ResidualBlock(in_channels, out_channels, s))
            in_channels = out_channels

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        emb = torch.flatten(x, 1)
        x = self.fc(emb)
        return x, emb
    
class EmbeddingLogger(TensorBoardLogger):
    def __init__(self, validation_set):
        super().__init__()
        self.validation_set = validation_set
        self.previous_embeddings_2d = None

        self.frames = []

    def log_embeddings(self, model, step):
        embeddings, labels = self.calculate_all_embeddings(model)
        embeddings_2d = self.calculate_umap_model(embeddings)
        if self.previous_embeddings_2d is not None:
            R, _ = orthogonal_procrustes(embeddings_2d, self.previous_embeddings_2d)
            embeddings_2d = embeddings_2d @ R

        self.previous_embeddings_2d = embeddings_2d
        image = self.log_embeddings_to_tensorboard(embeddings_2d, labels, step)

        self.frames.append(image)
        self.frames[0].save("embeddings/animation.gif", save_all=True, append_images=self.frames[1:], duration=300, loop=0)

        image.save(f"embeddings/embeddings_{step}.png")

        
    def calculate_all_embeddings(self, model):
        """Berechnet alle Embeddings für die Daten im Dataloader."""
        model.eval()
        embeddings = []
        labels = []
        bar = tqdm(self.validation_set, desc="Berechne Embeddings")
        with torch.no_grad():
            for inputs, l in bar:
                inputs = inputs.to(DEVICE)
                _, emb = model(inputs)
                embeddings.append(emb.cpu())
                labels.append(l.cpu())

        bar.close()

        return torch.cat(embeddings, dim=0).numpy(), torch.cat(labels, dim=0).numpy()

    def calculate_umap_model(self, embeddings):
        """Berechnet das UMAP-Modell für die Embeddings."""
        self.umap_model = umap.UMAP(n_components=2)
        embeddings_2d = self.umap_model.fit_transform(embeddings, verbose=True)

        embeddings_2d = np.array(embeddings_2d, dtype=np.float32)
        m = np.mean(embeddings_2d, axis=0, keepdims=True)  # Normalize to zero mean
        s = np.std(embeddings_2d, axis=0, keepdims=True)  # Normalize to unit variance
        embeddings_2d = (embeddings_2d - m) / s  # Normalize the embeddings

        return embeddings_2d

    def log_embeddings_to_tensorboard(self, embeddings_2d, labels, step):
        df = pd.DataFrame({
            'x': embeddings_2d[:, 0],
            'y': embeddings_2d[:, 1],
            'label': labels
        })

        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df, x='x', y='y', hue='label', palette='muted')
        plt.title(f"UMAP Embedding Projection - Step {step}")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()

        image = PIL.Image.open(buf)        
        self.writer.add_image(f'embeddings', np.array(image), global_step=step, dataformats='HWC')
        return image

        


if __name__ == "__main__":
    training_set, validation_set = load_data()

    # Initialisierung des Modells, Loss-Kriteriums und Optimierers
    model = ResNet().to(DEVICE)
    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR
    )  # Checkpoint laden, falls vorhanden

    # Checkpoint laden, falls vorhanden
    dirname = os.path.dirname(os.path.abspath(__file__))
    chkpt_path = os.path.join(dirname, "checkpoint.pth")

    ep = load_checkpoint(model, optimizer, chkpt_path)
    if ep > 0:
        print(f"Checkpoint geladen, fortsetzen bei Epoche {ep}.")

    # Das Modell trainieren
    logger = EmbeddingLogger(validation_set)

    # Logge den Graphen des Modells
    input_tensor = torch.randn(1, 3, 32, 32).to(DEVICE)  # Beispiel-Eingabetensor
    logger.log_graph(model, input_tensor)

    umap_model = None
    for n in range(ep, ep + 200):
        epoch(model, n, True, training_set, criterion, optimizer, logger=logger, log_after_n_samples=10000)
        epoch(model, n, False, validation_set, criterion, optimizer, logger=logger)

        # Checkpoint nach jeder Epoche speichern
        save_checkpoint(model, optimizer, n + 1, chkpt_path)

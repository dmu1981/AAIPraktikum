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

from tqdm import tqdm
from misc import (
    DEVICE,
    load_data,
    epoch,
    load_checkpoint,
    TensorBoardLogger,
    save_checkpoint,
    LR,
    ResNet
)

class EmbeddingLogger(TensorBoardLogger):
    def __init__(self, validation_set):
        super().__init__()
        self.validation_set = validation_set
        self.previous_embeddings_2d = None

        self.frames = []
        self.step = 1

    

        
    def calculate_embeddings(self, model):
        """Berechnet alle Embeddings für die Daten im Dataloader.
        
        Parameters:
        ----------- 
        model (nn.Module):
            Das Modell, das die Embeddings berechnet. 

        Returns:
        --------  
        embeddings (np.ndarray):
            Die berechneten Embeddings als NumPy-Array. 

        labels (np.ndarray):
            Die zugehörigen Labels als NumPy-Array.

        **TODO**:

        -  Setzen Sie das Modell in den Evaluationsmodus (`model.eval() <https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.eval>`_), um sicherzustellen, dass die Batch-Normalisierung deaktiviert ist.

        -  Erstellen Sie leere Listen für `embeddings` und `labels`, um die Ergebnisse zu speichern.

        -  Verwenden Sie `torch.no_grad() <https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html>`_, um den Gradientenfluss zu deaktivieren, da wir nur die Embeddings berechnen und nicht trainieren.

        -  Iterieren Sie über `self.validation_set` und berechnen Sie die Embeddings für jedes Batch indem Sie die Eingaben auf das Gerät (`DEVICE`) verschieben und das Modell aufrufen.

        -  Das Modell liefert ein Tupel zurück, wobei der zweite Wert die Embeddings sind. 

        -  Verschieben Sie die Embeddings und Labels auf die CPU (rufen Sie `tensor.cpu() <https://docs.pytorch.org/docs/stable/generated/torch.Tensor.cpu.html>`_ auf ) und speichern Sie sie in den Listen `embeddings` und `labels`.

        -  Konvertieren Sie die Listen `embeddings` und `labels` in NumPy-Arrays, indem Sie `torch.cat(embeddings, dim=0) <https://docs.pytorch.org/docs/stable/generated/torch.cat.html>`_ `.numpy() <https://docs.pytorch.org/docs/stable/generated/torch.Tensor.numpy.html>`_ und `torch.cat(labels, dim=0) <https://docs.pytorch.org/docs/stable/generated/torch.cat.html>`_ `.numpy() <https://docs.pytorch.org/docs/stable/generated/torch.Tensor.numpy.html>`_ verwenden.

        -  Setzen Sie das Modell wieder in den Trainingsmodus (`model.train() <https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.train>`_), um sicherzustellen, dass es für zukünftige Trainingsschritte bereit ist.

        -  Geben Sie die berechneten Embeddings und Labels zurück.
        """
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
        
        model.train()

        return torch.cat(embeddings, dim=0).numpy(), torch.cat(labels, dim=0).numpy()

    def calculate_tsne(self, embeddings, previous_embeddings_2d=None):
        """Berechnet das t-SNE-Modell für die Embeddings.
        
        Parameters:
        -----------
        embeddings (np.ndarray):
            Die Embeddings, die in 2D projiziert werden sollen.

        Returns:
        --------
        embeddings_2d (np.ndarray):
            Die 2D-Projektion der Embeddings. 

        **TODO**:

        -  Verwenden Sie `sklearn.manifold.TSNE <https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html>`_ um die Embeddings in 2D zu projizieren. 
           Setzen Sie `n_components=2` und verwenden Sie `init="pca"` für die Initialisierung. 
           Wenn `self.previous_embeddings_2d` nicht `None` ist, verwenden Sie stattdessen diese als Initialisierung.

        -  Konvertieren Sie die 2D-Embeddings in ein `NumPy-Array <https://numpy.org/doc/stable/reference/generated/numpy.array.html>`_ mit `dtype=np.float32`.
        
        -  Normalisieren Sie die 2D-Embeddings, indem Sie den Mittelwert und die Standardabweichung berechnen und 
           die Embeddings so transformieren, dass sie einen Mittelwert von 0 und eine Standardabweichung von 1 haben.

        -  Verwenden Sie `np.mean(embeddings_2d, axis=0, keepdims=True) <https://numpy.org/doc/2.2/reference/generated/numpy.mean.html>`_ für den Mittelwert und `np.std(embeddings_2d, axis=0, keepdims=True) <https://numpy.org/doc/stable/reference/generated/numpy.std.html>`_ für die Standardabweichung.  
        
        -  Normalisieren Sie die Embeddings mit `(embeddings_2d - m) / s`, wobei `m` der Mittelwert und `s` die Standardabweichung ist.
        
        -  Geben Sie die normalisierten 2D-Embeddings zurück.
        """
        if previous_embeddings_2d is not None:
            tsne_model = TSNE(n_components=2, init=previous_embeddings_2d)
        else:
            tsne_model = TSNE(n_components=2, init="pca")

        embeddings_2d = tsne_model.fit_transform(embeddings)

        embeddings_2d = np.array(embeddings_2d, dtype=np.float32)
        m = np.mean(embeddings_2d, axis=0, keepdims=True)  # Normalize to zero mean
        s = np.std(embeddings_2d, axis=0, keepdims=True)  # Normalize to unit variance
        embeddings_2d = (embeddings_2d - m) / s  # Normalize the embeddings

        return embeddings_2d

    def register_embeddings_2d(self, embeddings_2d, previous_embeddings_2d=None):
        """Registriert die 2D-Embeddings, um sie mit den vorherigen Embeddings zu vergleichen.
        
        Parameters:
        -----------
        embeddings_2d (np.ndarray):
            Die 2D-Embeddings, die registriert werden sollen.

        previous_embeddings_2d (np.ndarray, optional):  
              Die vorherigen 2D-Embeddings, die für die Registrierung verwendet werden sollen. Standardmäßig None.
              
        Returns:    
        --------
        embeddings_2d (np.ndarray):   
            Die registrierten 2D-Embeddings.  

        **TODO**:

        - Wenn `previous_embeddings_2d` nicht `None` ist, verwenden Sie `scipy.linalg.orthogonal_procrustes <https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.orthogonal_procrustes.html>`_ um die 2D-Embeddings zu registrieren.
          Dies hilft, die Embeddings so zu transformieren, dass sie mit den vorherigen bestmöglich Embeddings übereinstimmen.

        - Die Funktion liefert die orthogonale Rotationsmatrix `R` und die Skala `s`, aber wir verwenden nur `R`, um die 2D-Embeddings zu transformieren.    

        - Transformieren Sie die 2D-Embeddings mit `embeddings_2d @ R`, um sie an die vorherigen Embeddings anzupassen.

        - Geben Sie die transformierten 2D-Embeddings zurück.
        """
        if previous_embeddings_2d is not None:
            R, _ = orthogonal_procrustes(embeddings_2d, previous_embeddings_2d)
            embeddings_2d = embeddings_2d @ R

        return embeddings_2d

    def visualize_embeddings(self, embeddings_2d, labels, step, axs):
        """Visualisiert die 2D-Embeddings mit t-SNE und speichert das Bild.
        
        Parameters:
        -----------
        embeddings_2d (np.ndarray): 
            Die 2D-Embeddings, die visualisiert werden sollen.

        labels (np.ndarray):
            Die zugehörigen Labels für die Embeddings.  

        step (int):
            Der aktuelle Schritt oder die Epoche, die für den Titel des Plots verwendet wird.

        axs (matplotlib.axes.Axes):
            Die Achsen, auf denen die Embeddings visualisiert werden sollen.

        **TODO**:

        - Erstellen Sie mit Pandas ein DataFrame mit den 2D-Embeddings und den zugehörigen Labels, 
          um die Daten für die Visualisierung vorzubereiten.
        
        - Verwenden Sie `seaborn.scatterplot <https://seaborn.pydata.org/generated/seaborn.scatterplot.html>`_ um die 2D-Embeddings zu visualisieren.
        
        - Setzen Sie die Achsenlimits auf (-3.0, 3.0) für beide Achsen, um eine konsistente Darstellung zu gewährleisten.
        
        - Entfernen Sie die Legende (`axs.get_legend().remove()`), um den Plot übersichtlicher zu gestalten.
        
        - Setzen Sie den Titel des Plots sinnvoll.            
        """
        df = pd.DataFrame(
            {"x": embeddings_2d[:, 0], "y": embeddings_2d[:, 1], "label": labels}
        )

        sns.scatterplot(data=df, x="x", y="y", hue="label", palette="muted", ax=axs)
        axs.set_xlim(-3.0, 3.0)
        axs.set_ylim(-3.0, 3.0)
        axs.get_legend().remove()
        axs.set_title(f"t-SNE Embedding Projection - Step {step}")

        

    def append_frame(self, image):
        """Fügt ein Bild zu den Frames hinzu, die später als GIF gespeichert werden."""
        self.writer.add_image(
            f"embeddings", np.array(image), global_step=self.step, dataformats="HWC"
        )

        self.frames.append(image)

        dirname = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(dirname, "images")
        os.makedirs(image_path, exist_ok=True)

        self.frames[0].save(
            os.path.join(image_path, "animation.gif"),
            save_all=True,
            append_images=self.frames[1:],
            duration=300,
            loop=0,
        )

        image.save(os.path.join(image_path, f"embeddings_{self.step}.png"))

    def log_embeddings(self, model):
        embeddings, labels = self.calculate_embeddings(model)
        embeddings_2d = self.calculate_tsne(embeddings, self.previous_embeddings_2d)
        embeddings_2d = self.register_embeddings_2d(embeddings_2d, self.previous_embeddings_2d)
        self.previous_embeddings_2d = embeddings_2d

        fig = plt.figure(figsize=(8, 6))
        axs = fig.add_subplot(1, 1, 1)
        image = self.visualize_embeddings(embeddings_2d, labels, self.step, axs)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)

        image = PIL.Image.open(buf)

        self.append_frame(image)
        self.step += 1


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
        log_after = 100000
        if n == 0:
            log_after = 5000
        if n == 1:
            log_after = 10000
        if n == 2:
            log_after = 50000  

        epoch(
            model,
            n,
            True,
            training_set,
            criterion,
            optimizer,
            logger=logger,
            log_after_n_samples=log_after,
        )
        epoch(model, n, False, validation_set, criterion, optimizer, logger=logger)

        # Checkpoint nach jeder Epoche speichern
        save_checkpoint(model, optimizer, n + 1, chkpt_path)

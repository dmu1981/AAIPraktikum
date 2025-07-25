import os
import torch
import torch.nn as nn
from misc import (
    DEVICE,
    load_data,
    epoch,
    load_checkpoint,
    TensorBoardLogger,
    save_checkpoint,
    LR,
)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        """Initialisiert einen Residual Block.

        Parameters:
        -----------
        in_channels (int):
          Anzahl der Eingabekanäle.

        out_channels (int):
          Anzahl der Ausgabekanäle.

        stride (int):
          Schrittweite für die Faltung. Standard ist 1.

        **TODO**:

        - Rufen Sie die `__init__` Methode der Basisklasse `nn.Module` auf.

        - Initialisieren Sie dann die Schichten des Residual Blocks.

        - Verwenden Sie `nn.Conv2d <https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html>`_ für die Faltungsschichten. Setzen Sie `kernel_size=3`, `padding=1` und `bias=False`.

        - Die erste Faltungsschicht sollte `in_channels` zu `out_channels` transformieren, die zweite Faltungsschicht sollte `out_channels` zu `out_channels` transformieren.

        - Die ersten Faltungsschicht sollte `stride` als Schrittweite verwenden.

        - Fügen Sie `nn.BatchNorm2d <https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html>`_ nach jeder Faltungsschicht hinzu. Achten Sie darauf, dass die Batch-Normalisierung die gleiche Anzahl an Ausgabekanälen wie die Faltungsschicht hat.

        - Verwenden Sie `nn.ReLU <https://docs.pytorch.org/docs/stable/generated/torch.nn.ReLU.html>`_ als Aktivierungsfunktion.

        - Implementieren Sie die Shortcut-Verbindung. Wenn `stride` nicht 1 ist oder `in_channels` nicht gleich `out_channels`, verwenden Sie eine 1x1 Faltung, um die Dimensionen anzupassen. Andernfalls verwenden Sie `nn.Identity() <https://pytorch.org/docs/stable/generated/torch.nn.Identity.html>`_.
        """
        pass

    def forward(self, x):
        """Führt den Vorwärtsdurchlauf des Residual Blocks aus.

        Parameters:
        -----------
        x (torch.Tensor):
          Eingabetensor.

        **TODO**:
        Implementieren Sie den Vorwärtsdurchlauf des Residual Blocks.
        Orientieren Sie sich an der in der Aufgabenstellung gegebenen Beschreibung sowie der Grafik.
        """
        pass


class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        """Initialisiert das ResNet Modell.

        Parameters:
        -----------
        num_classes (int):
          Anzahl der Klassen für die Klassifikation.

        **TODO**:

        - Rufen Sie die `__init__` Methode der Basisklasse `nn.Module` auf.

        - Definieren Sie dann die Schichten des ResNet Modells.

        - Verwenden Sie `nn.Conv2d <https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html>`_ für die erste Faltungsschichten um von 3 auf 32 Kanäle zu transformieren. Setzen Sie `kernel_size=7`, `padding=3` und `stride=2` für diese Schicht.

        - Fügen Sie `nn.BatchNorm2d <https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html>`_ und `nn.ReLU <https://docs.pytorch.org/docs/stable/generated/torch.nn.ReLU.html>`_ nach der ersten Faltungsschicht hinzu.

        - Hinweis: Sie können die `nn.Sequential` Klasse verwenden, um mehrere Schichten zu kombinieren.

        - Erstellen Sie dann drei Ebenen mit der Methode `make_layer`.

        - Die erste Ebene sollte 6 Residual Blocks mit `in_channels=32`, `out_channels=32` und `stride=1` enthalten.

        - Die zweite Ebene sollte 6 Residual Blocks mit `in_channels=32`, `out_channels=64` und `stride=2` enthalten.

        - Die dritte Ebene sollte 12 Residual Blocks mit `in_channels=64`, `out_channels=128` und `stride=2` enthalten.

        - Fügen Sie eine `nn.AvgPool2d <https://docs.pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html>`_ Schicht mit einem Kernel von (4, 4) hinzu, um die räumliche Dimension der Feature-Maps zu reduzieren.

        - Fügen Sie eine voll verbundene Schicht `nn.Linear <https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html>`_ hinzu, die die Ausgabe der Durchschnittspooling-Schicht auf `num_classes` transformiert.

        - Die Eingabegröße für die voll verbundene Schicht sollte 128 sein, da die letzte Residual Block Schicht 128 Kanäle hat.

        - Verwenden Sie `torch.flatten <https://pytorch.org/docs/stable/generated/torch.flatten.html>`_ um die Ausgabe der Durchschnittspooling-Schicht in einen Vektor umzuwandeln, bevor Sie sie an die voll verbundene Schicht weitergeben.
        """
        pass

    def make_layer(self, in_channels, out_channels, num_blocks, stride):
        """Erstellt eine Sequenz von Residual Blocks.

        Parameters:
        -----------
        in_channels (int):
          Anzahl der Eingabekanäle.

        out_channels (int):
          Anzahl der Ausgabekanäle.

        num_blocks (int):
          Anzahl der Residual Blocks in dieser Schicht.

        stride (int):
          Schrittweite für die erste Faltungsschicht des ersten Blocks.

        Returns:
        --------
        nn.Sequential:
          Eine Sequenz von Residual Blocks.

        **TODO**:

        - Erstellen Sie eine Liste von Schichten, die die Residual Blocks enthalten.

        - Die erste Schicht sollte einen Residual Block mit `in_channels`, `out_channels` und `stride` sein.

        - Die folgenden Schichten sollten Residual Blocks mit gleichbleibender Kanalanzahl sein. Verwenden Sie `out_channels` sowohl für die Eingabe- als auch für die Ausgabekanäle.

        - Verwenden Sie `nn.Sequential` um die Schichten zu kombinieren und zurückzugeben.

        **Hinweis**:

        - Die erste Schicht sollte die Schrittweite `stride` verwenden, während die anderen Schichten eine Schrittweite von 1 haben.

        - Sie können die gewünschten Layer mit `nn.Sequential <https://docs.pytorch.org/docs/stable/generated/torch.nn.Sequential.html>`_ kombinieren.

        - Dazu können Sie die Blöcke zunächst in einer Liste (z.B. `layers`) sammeln und dann `nn.Sequential(*layers)` verwenden, um sie zu kombinieren.
        """
        pass

    def forward(self, x):
        """Führt den Vorwärtsdurchlauf des ResNet Modells aus.

        Parameters:
        -----------
        x (torch.Tensor):
          Eingabetensor.

        **TODO**:
        Implementieren Sie den Vorwärtsdurchlauf des ResNet Modells.
        Orientieren Sie sich an der in der Aufgabenstellung gegebenen Beschreibung sowie der Grafik.
        """
        pass


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
    logger = TensorBoardLogger()

    # Logge den Graphen des Modells
    input_tensor = torch.randn(1, 3, 32, 32).to(DEVICE)  # Beispiel-Eingabetensor
    logger.log_graph(model, input_tensor)

    umap_model = None
    for n in range(ep, ep + 200):
        epoch(
            model,
            n,
            True,
            training_set,
            criterion,
            optimizer,
            logger=logger,
            log_after_n_samples=10000,
        )
        epoch(model, n, False, validation_set, criterion, optimizer, logger=logger)

        # Checkpoint nach jeder Epoche speichern
        save_checkpoint(model, optimizer, n + 1, chkpt_path)

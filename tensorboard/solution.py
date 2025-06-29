import torch
from torch import nn
import torchvision
import os
from misc import (
    CNNNetwork,
    load_data,
    epoch,
    save_checkpoint,
    load_checkpoint,
    DEVICE,
    LR,
)

from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    def __init__(self):
        """
        Initialisiert den TensorBoard-Logger.
        """
        self.create_writer()
        self._reset_samples_statistics()
        self._reset_metrics()

    def create_writer(self):
        """
        Erstellt einen TensorBoard-SummaryWriter, der die Logs in einem Verzeichnis speichert.

        **TODO**:
        Erstellen Sie einen `SummaryWriter <https://docs.pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter>`_, der die Logs in einem Verzeichnis namens "runs"
        speichert. Das Verzeichnis sollte im gleichen Verzeichnis wie dieses Skript liegen.
        Verwenden Sie `os.path.dirname <https://www.tutorialspoint.com/python/os_path_dirname.htm#:~:text=The%20Python%20os.,the%20specified%20file%20or%20directory.>`_ `(os.path.abspath <https://www.geeksforgeeks.org/python/python-os-path-abspath-method-with-example/>`_ `(__file__))`, um den Pfad zum aktuellen Verzeichnis zu erhalten,
        und `os.path.join() <https://www.geeksforgeeks.org/python/python-os-path-join-method/>`_, um den Pfad zum "runs"-Verzeichnis zu erstellen.
        """
        dirname = os.path.dirname(os.path.abspath(__file__))
        board_path = os.path.join(dirname, "runs")

        self.writer = SummaryWriter(log_dir=board_path)

    def _reset_metrics(self):
        """Setzt die Metriken zurück."""
        self.metrics = {"total_loss": 0.0, "total_correct": 0.0, "total_samples": 0}

    def _reset_samples_statistics(self):
        """Setzt die Statistik der Samples zurück."""
        self.sample_statistics = {}
        for i in range(10):
            self.sample_statistics[i] = {
                "samples": torch.tensor([], device=DEVICE),
                "loss": torch.tensor([], device=DEVICE),
            }

    def log_graph(self, model, input_tensor):
        """Loggt den Graphen des Modells in TensorBoard.

        Parameters:
        -----------
        model (nn.Module):
          Das Modell, dessen Graph geloggt werden soll.

        input_tensor (torch.Tensor):
          Ein Beispiel-Eingabetensor, der die Form des Eingabedaten repräsentiert.

        **TODO**:
        Verwenden Sie `writer.add_graph() <https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_graph>`_, um den Graphen
        des Modells zu loggen.
        """
        self.writer.add_graph(model, input_tensor)

    def update_metrics(self, logits, labels):
        """Aktualisiert die Metriken für Trainings- oder Validierungsdaten.

        Parameters:
        -----------
        logits (torch.Tensor):
          Die zum aktuell verarbeiteten Batch gehörenden Logits (Vorhersagen) des Modells.

        labels (torch.Tensor):
          Die zugehörigen Labels für den Batch.

        Updates:
        --------
        self.metrics (dict):
          Diese Variable speichert die Metriken `total_loss`, `total_correct` und `total_samples`.

        self.metrics["total_loss"] (float):
          Der kumulierte summarische Verlust über alle bisherigen Batches.

        self.metrics["total_correct"] (int):
          Die Anzahl der korrekten Vorhersagen über alle bisherigen Batches.

        self.metrics["total_samples"] (int):
          Die Gesamtzahl der verarbeiteten Samples über alle bisherigen Batches.

        **TODO**:
        Aktualisiere die Metriken `total_loss`, `total_correct` und `total_samples` für den aktuellen Batch.
        - Berechne den Verlust für den Batch mit `nn.CrossEntropyLoss()`.
        - Zähle die Anzahl der korrekten Vorhersagen im Batch. Hinweis: Verwende `torch.argmax(logits, 1) <https://docs.pytorch.org/docs/stable/generated/torch.argmax.html>`_ um die Vorhersagen zu erhalten und vergleiche sie mit den Labels.
        - Aktualisiere die Metriken in `self.metrics["total_loss"]`, `self.metrics["total_correct"]` und `self.metrics["total_samples"]` entsprechend.
        """
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, labels)

        # Berechne die Anzahl der korrekten Vorhersagen
        predicted = torch.argmax(logits, 1)
        correct = (predicted == labels).sum().item()

        # Aktualisiere die Metriken
        self.metrics["total_loss"] += loss.item()
        self.metrics["total_correct"] += correct
        self.metrics["total_samples"] += labels.size(0)

    def log_metrics(self, step, train=True):
        """Loggt Metriken in TensorBoard.

        Parameters:
        -----------
        step (int):
          Der aktuelle Schritt, der für das Logging verwendet wird.

        train (bool):
          Gibt an, ob die Metriken aus dem Trainings- oder Validierungsdatensatz stammen.

        **TODO**:
        Logge die skalaren Metriken `loss` und `accuracy` über den `SummaryWriter` ins TensorBoard.
        Die Metriken sollten in zwei verschiedenen Tags gespeichert werden: "train" für Trainingsmetriken und "validation" für
        Validierungsmetriken. Überprüfe, ob `train` wahr ist, um zu entscheiden, ob es sich um Trainings- oder Validierungsmetriken handelt.

        Die Metriken sollten mit dem aktuellen Schritt `step` geloggt werden.
        - Berechnen Sie den Verlust als `self.metrics["total_loss"] / self.metrics["total_samples"]`.
        - Berechnen Sie die Genauigkeit als `self.metrics["total_correct"] / self.metrics["total_samples"]`.
        - Verwenden Sie `self.writer.add_scalar() <https://docs.pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_scalar>`_ um die Metriken zu loggen.
        - Rufen Sie zum Schluß `self._reset_metrics()` auf, um die Metriken zurückzusetzen.
        """
        loss = self.metrics["total_loss"] / self.metrics["total_samples"]
        accuracy = self.metrics["total_correct"] / self.metrics["total_samples"]

        tag = "train" if train else "validation"
        self.writer.add_scalar(f"{tag}/loss", loss, step)
        self.writer.add_scalar(f"{tag}/accuracy", accuracy, step)

        self._reset_metrics()

    def log_sample_statistics(self, train, step):
        """Loggt die am schlechtesten klassifizierten Samples in TensorBoard.

        Parameters:
        -----------
        train (bool):
          Gibt an, ob die Samples aus dem Trainings- oder Validierungsdatensatz stammen.

        step (int):
          Der aktuelle Schritt, der für das Logging verwendet wird.

        **TODO**:
        Logge die am schlechtesten klassifizierten Samples für jede Klasse in TensorBoard.
        Die Samples sollten in einem Grid-Format geloggt werden, wobei jede Klasse in einem eigenen Tag gespeichert wird.

        - Iteriere über die Klassen-IDs (0-9) und logge die Samples für jede Klasse.
        - Verwende `self.sample_statistics[cls_id]["samples"]` um die Samples für die Klasse `cls_id` zu erhalten.
        - Verwende `torchvision.utils.make_grid() <https://docs.pytorch.org/vision/stable/generated/torchvision.utils.make_grid.html>`_ um die Samples in einem Grid zu formatieren
        - Übergeben Sie `normalize=True` um die Samples zu normalisieren.
        - Logge die Samples mit `self.writer.add_image() <https://docs.pytorch.org/docs/stable//tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_image>`_ unter dem Tag `f"{tag}/worst_samples/class_{cls_id}"`, wobei `tag` entweder "train" oder "validation" ist.
        - Rufen Sie ganz zum Schluß `self._reset_samples_statistics()` auf, um die Statistik der Samples zurückzusetzen, nachdem die Samples geloggt wurden.
        """
        # Logge die schlechtesten Samples, wenn die Epoche abgeschlossen ist
        tag = "train" if train else "validation"

        # Iteriere über die Klassen-IDs (0-9) und logge die Samples für jede Klasse
        for cls_id in range(10):
            # Erstelle ein Grid aus den Samples der Klasse
            grid = torchvision.utils.make_grid(
                self.sample_statistics[cls_id]["samples"],
                normalize=True,
            )

            # Logge das Grid der Samples in TensorBoard
            self.writer.add_image(
                f"{tag}/worst_samples/class_{cls_id}",
                grid,
                global_step=step,
            )

        # Setze die Statistik der Samples zurück
        self._reset_samples_statistics()

    def update_sample_statistics(self, batch, labels, loss):
        """Aggregiere die am schlechtesten klassifizierten Samples für jede Klasse.

        Parameters:
        -----------
        batch (torch.Tensor):
          Der Batch von Eingabedaten.

        labels (torch.Tensor):
          Die zugehörigen Labels für den Batch.

        loss (torch.Tensor):
          Der Verlust für den Batch, berechnet mit `nn.CrossEntropyLoss(reduction="none")`.

        Updates:
        --------
        self.sample_statistics (dict):
          Diese Variable speichert die Samples und deren zugehörigen Verlusten für jede Klasse.

        self.sample_statistics[cls_id]["samples"] (torch.Tensor):
          Enthält die bisher schwierigsten Samples für die Klasse `cls_id`.

        self.sample_statistics[cls_id]["loss"] (torch.Tensor):
          Enthält die bisher größten Loss-Werte für die Klasse `cls_id`.

        Diese Methode aggregiert die am schlechtesten klassifizierten Samples für jede Klasse.
        Die Samples werden in `self.sample_statistics` gespeichert, die für jede Klasse eine Liste von Samples
        und deren zugehörigen Verlusten enthält.
        Die Methode iteriert über die Klassen-IDs (0-9) und speichert die 64 Samples mit dem höchsten Verlust
        für jede Klasse.

        **TODO**:
        - Iterieren Sie über die Klassen-IDs (0-9) und speichern Sie die 64 Samples mit dem höchsten Verlust für jede Klasse.
        - Verwenden Sie `torch.cat() <https://docs.pytorch.org/docs/stable/generated/torch.cat.html>`_ um die Samples und Verluste für jede Klasse zu aggregieren.
          Verwenden Sie `torch.clone() <https://docs.pytorch.org/docs/stable/generated/torch.clone.html>`_ `.detach() <https://docs.pytorch.org/docs/stable/generated/torch.Tensor.detach.html>`_ um sicherzustellen, dass die Samples und Verluste nicht mehr mit der
           Gradientenberechnung von AutoGrad verbunden sind.
        - Sortieren Sie die Samples nach Verlust in absteigender Reihenfolge und behalten Sie nur die 64 schlechtesten Samples.
          Verwenden Sie `torch.argsort() <https://docs.pytorch.org/docs/stable/generated/torch.argsort.html>`_ um die Indizes der Samples nach Verlust zu sortieren.
        - Aktualisieren Sie `self.sample_statistics` für jede Klasse mit den aggregierten Samples und Verlusten.
        """
        # Iteriert über die Klassen-IDs (0-9) und speichert die schlechtesten Samples
        for cls_id in range(10):
            # Filtere die Samples für die aktuelle Klasse
            ids = labels == cls_id

            # Konkatenieren der Samples und Verluste für die aktuelle Klasse
            self.sample_statistics[cls_id]["samples"] = torch.cat(
                [
                    self.sample_statistics[cls_id]["samples"],
                    batch[ids].clone().detach(),
                ]
            )
            self.sample_statistics[cls_id]["loss"] = torch.cat(
                [
                    self.sample_statistics[cls_id]["loss"],
                    loss[ids].clone().detach(),
                ]
            )

            # Sortiere die Samples nach Verlust in absteigender Reihenfolge
            sorted_indices = torch.argsort(
                self.sample_statistics[cls_id]["loss"], descending=True
            )

            # Behalte nur die 64 schlechtesten Samples
            sorted_indices = sorted_indices[:64]

            # Aktualisiere die Samples und Verluste für die aktuelle Klasse
            self.sample_statistics[cls_id]["samples"] = self.sample_statistics[cls_id][
                "samples"
            ][sorted_indices]

            self.sample_statistics[cls_id]["loss"] = self.sample_statistics[cls_id][
                "loss"
            ][sorted_indices]


if __name__ == "__main__":
    training_set, validation_set = load_data()

    # Initialisierung des Modells, Loss-Kriteriums und Optimierers
    model = CNNNetwork().to(DEVICE)
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

    for n in range(ep, ep + 30):
        epoch(model, n, True, training_set, criterion, optimizer, logger=logger)
        epoch(model, n, False, validation_set, criterion, optimizer, logger=logger)

        # Checkpoint nach jeder Epoche speichern
        save_checkpoint(model, optimizer, n + 1, chkpt_path)

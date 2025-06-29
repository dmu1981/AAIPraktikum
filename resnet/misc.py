import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os

LR = 0.001  # Lernrate für den Optimierer
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO: Data Augmentation Pipeline
# Hier definieren wir die Transformationspipeline für die Trainings- und Validierungsdaten.
# Die Trainingsdaten werden mit verschiedenen Transformationen augmentiert, um die Robustheit des Modells
# zu erhöhen. Folgend Sie den Anweisungen in der Aufgabenstellung, um die Pipeline zu vervollständigen.
training_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomCrop(size=(32, 32), padding=4),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

# TODO: Validation Pipeline
# Definieren Sie hier eine zweite Transformationspipeline für die Validierungsdaten.
# Folgend Sie den Anweisungen in der Aufgabenstellung, um die Pipeline zu vervollständigen.
validation_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)


class TensorBoardLogger:
    def __init__(self):
        """
        Initialisiert den TensorBoard-Logger.
        """
        self.create_writer()
        self._reset_samples_statistics()
        self._reset_metrics()

    def create_writer(self):
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
        self.writer.add_graph(model, input_tensor)

    def update_metrics(self, logits, labels):
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
        loss = self.metrics["total_loss"] / self.metrics["total_samples"]
        accuracy = self.metrics["total_correct"] / self.metrics["total_samples"]

        tag = "train" if train else "validation"
        self.writer.add_scalar(f"{tag}/loss", loss, step)
        self.writer.add_scalar(f"{tag}/accuracy", accuracy, step)

        self._reset_metrics()

    def log_sample_statistics(self, train, step):
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


def load_data():
    # TODO: Laden der CIFAR-10-Daten
    training_data = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=training_transform
    )
    validation_data = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=validation_transform
    )

    training_set = torch.utils.data.DataLoader(
        training_data, batch_size=256, shuffle=True
    )
    validation_set = torch.utils.data.DataLoader(
        validation_data, batch_size=256, shuffle=False
    )

    return training_set, validation_set


def epoch(
    model,
    n,
    train,
    dataloader,
    criterion,
    optimizer,
    log_after_n_samples=5000,
    logger=None,
):
    """
    Führt eine einzelne Trainings- oder Evaluations-Epoche für das Modell aus.
    """
    # Vorbereiten des Modells für Training oder Evaluation
    if train:
        model.train()
    else:
        model.eval()

    # Training des Modells
    total_samples = 0
    counter = 0
    bar = tqdm(dataloader)
    for data, labels in bar:
        # Daten und Labels auf das Gerät verschieben
        data, labels = data.to(DEVICE), labels.to(DEVICE)

        # Gradienten zurücksetzen
        if train:
            optimizer.zero_grad()

        # Vorwärtsdurchlauf
        with torch.set_grad_enabled(train):
            outputs, _ = model(data)

        # Verlust berechnen und Rückwärtsdurchlauf
        loss = criterion(outputs, labels)

        # Wenn eine Callback-Funktion für Samples definiert ist, rufen Sie sie auf
        if logger is not None:
            logger.update_sample_statistics(data, labels, loss)
            logger.update_metrics(outputs, labels)

        loss = loss.mean()  # Durchschnittlichen Verlust über die Batch-Größe berechnen

        # Gradienten berechnen
        if train:
            loss.backward()
            optimizer.step()

        # Aktualisieren der Metriken
        total_samples += data.size(0)

        bar.set_description(
            f"Epoch {n} ({'T' if train else 'V'})"  # , Loss: {total_loss / total_samples:.4f}, Accuracy: {total_correct / total_samples:.2%}"
        )

        # Loggen der Metriken nach einer bestimmten Anzahl von Samples (nur fürs Trainingsset)
        if total_samples > log_after_n_samples and logger is not None and train:
            counter += total_samples
            logger.log_metrics(
                n * len(dataloader.dataset) + counter,
                train,
            )

            total_samples = 0

    # Always output smth. at the end of the epoch
    if logger is not None:
        logger.log_metrics(
            n * len(dataloader.dataset) + counter,
            train,
        )
        logger.log_sample_statistics(train, n * len(dataloader.dataset) + counter)


def save_checkpoint(model, optimizer, epoch, filename="checkpoint.pth"):
    """Speichert den aktuellen Zustand des Modells und des Optimierers in einer Datei."""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        filename,
    )


def load_checkpoint(model, optimizer, filename="checkpoint.pth"):
    """Lädt den Zustand des Modells und des Optimierers aus einer Datei."""
    try:
        checkpoint = torch.load(filename, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint["epoch"]
    except Exception as e:
        print(f"Fehler beim Laden des Checkpoints {filename}: {e}")
        print("Starte ohne gespeicherten Zustand.")
        return 0

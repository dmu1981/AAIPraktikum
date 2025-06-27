import torch
import torch.nn as nn
from torchvision import transforms, datasets
from tqdm import tqdm

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


class CNNNetwork(nn.Module):
    """Ein einfaches neuronales Netzwerk mit einer versteckten Schicht."""

    def __init__(self):
        """Initialisiert das Netzwerk mit mehreren Convolutional-Schichten und voll verbundenen Schichten."""
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding="same")
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding="same")
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding="same")

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.flatten = nn.Flatten()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """Führt den Vorwärtsdurchlauf des Netzwerks aus.eben Sie die Ausgabe der letzten Schicht `fc2` zurück."""
        x = self.pool1(self.bn1(torch.relu(self.conv1(x))))
        x = self.pool2(self.bn2(torch.relu(self.conv2(x))))
        x = self.pool3(self.bn3(torch.relu(self.conv3(x))))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


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
            outputs = model(data)

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
            f"Epoch {n} ({'T' if train else 'V'})"#, Loss: {total_loss / total_samples:.4f}, Accuracy: {total_correct / total_samples:.2%}"
        )

        # Loggen der Metriken nach einer bestimmten Anzahl von Samples (nur fürs Trainingsset)
        if total_samples > log_after_n_samples and logger is not None and train:
            counter += total_samples
            logger.log_metrics(
                n * len(dataloader.dataset) + counter,
                train,
            )

            total_loss = 0.0
            total_samples = 0
            total_correct = 0

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

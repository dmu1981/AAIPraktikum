import torch
import torch.nn as nn
from torchvision import transforms, datasets
from tqdm import tqdm

LR = 0.01  # Lernrate
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO: Data Augmentation Pipeline
# Hier definieren wir die Transformationspipeline für die Trainings- und Validierungsdaten.
# Die Trainingsdaten werden mit verschiedenen Transformationen augmentiert, um die Robustheit des Modells
# zu erhöhen. Folgend Sie den Anweisungen in der Aufgabenstellung, um die Pipeline zu vervollständigen.
training_transform = None

# TODO: Validation Pipeline
# Definieren Sie hier eine zweite Transformationspipeline für die Validierungsdaten.
# Folgend Sie den Anweisungen in der Aufgabenstellung, um die Pipeline zu vervollständigen.
validation_transform = None

if __name__ == "__main__":
    # TODO: Laden der CIFAR-100-Daten
    training_data = None
    validation_data = None

    training_set = None
    validation_set = None


class CNNNetwork(nn.Module):
    """Ein einfaches neuronales Netzwerk mit einer versteckten Schicht."""

    def __init__(self):
        """Initialisiert das Netzwerk mit mehreren Convolutional-Schichten und voll verbundenen Schichten.

        **TODO**:

        - Rufen Sie die Methode `super().__init__()` auf, um die Basisklasse zu initialisieren.

        - Definieren Sie die Faltungs-Schichten `conv1`, `conv2`, `conv3` mit den entsprechenden Eingangs- und Ausgangskanälen. Verwenden Sie `nn.Conv2d(...) <https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html>`_. Setzen Sie `kernel_size=3` und `padding="same"` für alle Schichten.
          Verwenden Sie jeweils 16, 32 und 64 Ausgänge für `conv1`, `conv2` und `conv3`.

        - Definieren Sie die voll verbundenen Schichten `fc1` und `fc2` mit den entsprechenden Eingangs- und Ausgangsgrößen. Verwenden Sie `nn.Linear(...) <https://pytorch.org/docs/stable/generated/torch.nn.Linear.html>`_. Setzen Sie `fc1` auf 512 Ausgänge und `fc2` auf 100 Ausgänge.

        - Fügen Sie eine `Flatten-Schicht <https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html>`_ hinzu, um die Ausgabe der Convolutional-Schichten in einen Vektor umzuwandeln.

        - Fügen Sie eine `Max-Pooling-Schicht <https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html>`_ `pool` mit `kernel_size=2` und `stride=2` hinzu, um die räumliche Dimension der Feature-Maps zu reduzieren.

        - Verwenden Sie `torch.relu <https://pytorch.org/docs/stable/generated/torch.relu.html>`_ für die Aktivierung.
        """
        pass

    def forward(self, x):
        """Führt den Vorwärtsdurchlauf des Netzwerks aus.

        **TODO**:

        - Wenden Sie abwechselnd immer die Faltungs-Schichten `conv1`, `conv2`, `conv3` auf die Eingabe `x` an, gefolgt von einer ReLU-Aktivierung und einem Pooling-Layer.

        - Flatten Sie die Ausgabe der letzten Faltungs-Schicht mit .`self.flatten(x)`

        - Wenden Sie die voll verbundenen Schichten `fc1` und `fc2` auf die flachgelegte Ausgabe an, wobei Sie ReLU-Aktivierung auf die Ausgabe von `fc1` anwenden.

        - Geben Sie die Ausgabe der letzten Schicht `fc2` zurück.
        """
        pass


def epoch(model, n, train, dataloader, criterion, optimizer):
    """
    Führt eine einzelne Trainings- oder Evaluations-Epoche für das Modell aus.

    Parameters:
    -----------

    model (nn.Module):
        Das zu trainierende oder evaluierende Modell.

    n (int):
        Die aktuelle Epoche.

    train (bool):
        Gibt an, ob die Epoche im Trainingsmodus oder im Evaluationsmodus durchgeführt wird.

    dataloader (DataLoader):
        Der DataLoader, der die Daten für die Epoche bereitstellt.

    criterion (nn.Module):
        Das Loss-Kriterium, das zur Berechnung des Verlusts verwendet wird.

    optimizer (torch.optim.Optimizer):
        Der Optimierer, der zur Aktualisierung der Modellparameter verwendet wird.

    **TODO**:

    - Setzen Sie das Modell in den Trainingsmodus, wenn `train=True` ist, und in den Evaluationsmodus, wenn `train=False` ist. Rufen Sie dazu `model.train()` bzw. `model.eval()` auf.

    - Initialisieren Sie `total_loss`, `total_samples` und `total_correct` auf 0.0, 0 und 0.

    - Verwenden Sie `tqdm` für den Fortschrittsbalken, um den Fortschritt der Epoche anzuzeigen.
      Speichern Sie den Iterator in einer eigenen Variable damit er innerhalb der Schleife verwendet werden kann.

    - Iterieren Sie über den `dataloader` und führen Sie die folgenden Schritte aus:

    - Verschieben Sie die Daten und Labels auf das Gerät (`DEVICE`).

    - Setzen Sie die Gradienten des Optimierers zurück, wenn `train=True` ist, indem Sie `optimizer.zero_grad()` aufrufen.

    - Führen Sie den Vorwärtsdurchlauf des Modells aus, indem Sie `model(data)` aufrufen. Verwenden Sie `torch.set_grad_enabled(train)`, um den Gradientenfluss nur im Trainingsmodus zu aktivieren.

    - Berechnen Sie den Verlust mit `criterion(outputs, labels)`.

    - Wenn `train=True` ist, führen Sie den Rückwärtsdurchlauf aus, indem Sie `loss.backward()` aufrufen und die Parameter mit `optimizer.step()` aktualisieren.

    - Aktualisieren Sie `total_loss`, `total_samples` und `total_correct` mit den entsprechenden Werten aus dem aktuellen Batch.
      Der `total_loss` sollte den Verlust des aktuellen Batches aufsummieren, `total_samples` die Anzahl der Samples im aktuellen Batch und
      `total_correct` die Anzahl der korrekt klassifizierten Samples. Die Anzahl der korrekt klassifizierten Samples kann mit `(outputs.argmax(dim=1) == labels).sum()` berechnet werden.

    - Aktualisieren Sie den Fortschrittsbalken mit dem aktuellen Verlust und der Genauigkeit. Zeigen Sie auch an ob das Netz im Trainings- oder Validationsmodus betrieben wird.
      Rufen Sie dazu tqdm.set_description() auf und formatieren Sie die Ausgabe entsprechend.
    """
    pass


if __name__ == "__main__":
    # Initialisierung des Modells, Loss-Kriteriums und Optimierers
    model = CNNNetwork().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Das Modell trainieren
    model.train()
    for n in range(1, 30):
        epoch(model, n, True, training_set, criterion, optimizer)
        epoch(model, n, False, validation_set, criterion, optimizer)

import torch
import torch.nn as nn
from torchvision import transforms, datasets
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO: Data Augmentation Pipeline
# Hier definieren wir die Transformationspipeline für die Trainings- und Validierungsdaten.
# Die Trainingsdaten werden mit verschiedenen Transformationen augmentiert, um die Robustheit des Modells   
# zu erhöhen. Folgend Sie den Anweisungen in der Aufgabenstellung, um die Pipeline zu vervollständigen.
training_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.RandomCrop(size=(32, 32), padding=4),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# TODO: Validation Pipeline
# Definieren Sie hier eine zweite Transformationspipeline für die Validierungsdaten.
# Folgend Sie den Anweisungen in der Aufgabenstellung, um die Pipeline zu vervollständigen.
validation_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def load_data():
    # TODO: Laden der CIFAR-10-Daten
    training_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=training_transform)
    validation_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=validation_transform)

    training_set = torch.utils.data.DataLoader(training_data, batch_size=256, shuffle=True)
    validation_set = torch.utils.data.DataLoader(validation_data, batch_size=256, shuffle=False)

    return training_set, validation_set

class CNNNetwork(nn.Module):
    """Ein einfaches neuronales Netzwerk mit einer versteckten Schicht."""

    def __init__(self):
        """Initialisiert das Netzwerk mit mehreren Convolutional-Schichten und voll verbundenen Schichten.
        """
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
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """Führt den Vorwärtsdurchlauf des Netzwerks aus.eben Sie die Ausgabe der letzten Schicht `fc2` zurück.
        """
        x = self.pool(self.bn1(torch.relu(self.conv1(x))))
        x = self.pool(self.bn2(torch.relu(self.conv2(x))))
        x = self.pool(self.bn3(torch.relu(self.conv3(x))))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def epoch(model, n, train, dataloader, criterion, optimizer):
    """
    Führt eine einzelne Trainings- oder Evaluations-Epoche für das Modell aus.
    """
    # Vorbereiten des Modells für Training oder Evaluation
    if train:
        model.train()
    else:
        model.eval()

    # Training des Modells
    total_loss = 0.0
    total_samples = 0
    total_correct = 0
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

        # Gradienten berechnen
        if train:
            loss.backward()
            optimizer.step()

        # Aktualisieren der Metriken
        total_loss += loss.item()
        total_samples += data.size(0)
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        # print(labels)
        # print(outputs.argmax(dim=1) == labels)
        # exit()

        bar.set_description(f"Epoch {n} ({'T' if train else 'V'}), Loss: {total_loss / total_samples:.4f}, Accuracy: {total_correct / total_samples:.2%}")

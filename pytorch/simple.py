import torch
import torch.nn as nn

LR = 0.04 # Lernrate
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO: 
# Legen Sie die Trainingsdaten und Labels fest.
# Die Trainingsdaten repräsentieren die Eingaben für ein XOR-Problem.
# Die Labels repräsentieren die erwarteten Ausgaben für diese Eingaben.
# Die Eingaben sind 2D-Punkte, und die Labels sind die erwarteten Klassifikationen.
# Die Daten und Labels sollten auf das Gerät `DEVICE` verschoben werden.
# Achten Sie darauf, dass die Daten als `torch.float32` und die Labels als `torch.long` definiert sind.
training_data = None
labels = None

class SimpleNetwork(nn.Module):
    """Ein einfaches neuronales Netzwerk mit einer versteckten Schicht."""

    def __init__(self):
        """Initialisiert das Netzwerk mit einer versteckten Schicht.
        
        **TODO**:
        
        - Rufen Sie die Methode `super().__init__()` auf, um die Basisklasse zu initialisieren.

        - Definieren Sie die erste voll verbundene Schicht `fc1` mit 2 Eingängen und 8 Ausgängen.

        - Definieren Sie die zweite voll verbundene Schicht `fc2` mit 8 Eingängen und 2 Ausgängen.
        """
        pass

    def forward(self, x):
        """Führt den Vorwärtsdurchlauf des Netzwerks aus.

        **TODO**:

        - Wenden Sie die erste voll verbundene Schicht `fc1` auf die Eingabe `x` an.

        - Wenden Sie die ReLU-Aktivierungsfunktion (`torch.relu <https://docs.pytorch.org/docs/stable/generated/torch.nn.ReLU.html>`_) auf die Ausgabe der ersten Schicht `fc1` an.

        - Wenden Sie die zweite voll verbundene Schicht `fc2` auf die Ausgabe der ReLU-Aktivierung an.

        - Geben Sie die Ausgabe der zweiten Schicht `fc2` zurück.
        """
        pass

def train_model(model, data, labels, criterion, optimizer, epochs=8000):
    """Trainiert das Modell mit den gegebenen Daten und Labels.
    
    Diese Funktion führt das Training des Modells durch, indem sie die Eingabedaten und Labels verwendet,
    um die Gewichte des Modells zu aktualisieren. Der Verlust wird in jeder 1000. Epoche ausgegeben.

    Parameter:
    ----------

    model : nn.Module
        Das zu trainierende neuronale Netzwerk.

    data : torch.Tensor
        Die Eingabedaten für das Training.

    labels : torch.Tensor
        Die zugehörigen Labels für die Eingabedaten.

    criterion : nn.Module
        Das Kriterium zur Berechnung des Verlusts (z.B. CrossEntropyLoss).

    optimizer : torch.optim.Optimizer
        Der Optimierer, der verwendet wird, um die Gewichte des Modells zu aktualisieren

    epochs : int, optional
        Die Anzahl der Epochen, die das Modell trainiert werden soll (Standard: 8000).

    **TODO**:
    
    Iterieren Sie über die Anzahl der Epochen und führen Sie in jeder Epoche die folgenden Schritte aus:

    - Setzen Sie die Gradienten des Optimierers zurück. (`optimizer.zero_grad() <https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html>`_)

    - Führen Sie einen Vorwärtsdurchlauf des Modells mit den Eingabedaten `data` durch.
    
    - Berechnen Sie den Verlust zwischen den Modell-Ausgaben und den Labels mit dem Kriterium `criterion`.

    - Führen Sie den Rückwärtsdurchlauf durch, um die Gradienten zu berechnen. (`loss.backward() <https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html>`_)

    - Geben Sie den Verlust alle 1000 Epochen aus.

    - Führen Sie den Optimierungsschritt durch, um die Gewichte des Modells zu aktualisieren. (`optimizer.step() <https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.step.html>`_)
    """
    pass

if __name__ == "__main__":
    # Initialisierung des Modells, Loss-Kriteriums und Optimierers
    model = SimpleNetwork().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)

    # Das Modell trainieren
    model.train()
    train_model(model, training_data, labels, criterion, optimizer)

    # Nach dem Training das Modell verwenden
    model.eval()
    with torch.no_grad():
      outputs = model(training_data)

    print("Training complete.")
    print(outputs)
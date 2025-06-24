import os
import torch
from torch import nn
from misc import DEVICE, CNNNetwork, load_data, epoch

LR = 0.001 # Lernrate

def save_checkpoint(model, optimizer, epoch, filename='checkpoint.pth'):
    """Speichert den aktuellen Zustand des Modells und des Optimierers in einer Datei.

    Parameters:
    -----------
    model (nn.Module): 
        Das zu speichernde Modell.

    optimizer (torch.optim.Optimizer): 
        Der Optimierer, dessen Zustand gespeichert werden soll.

    epoch (int): 
        Die aktuelle Epoche, die im Checkpoint gespeichert wird.

    filename (str): 
        Der Name der Datei, in der der Checkpoint gespeichert wird.

    **TODO**:
    Erzeuge ein Dictionary, das den Zustand des Modells, des Optimierers und die aktuelle Epoche enthält.
    Den Zustand der Modells und des Optimierers kannst du mit `model.state_dict()` und `optimizer.state_dict()` erhalten.
    Speichere dieses Dictionary mit `torch.save()` unter dem angegebenen Dateinamen.
    """
    pass

def load_checkpoint(model, optimizer, filename='checkpoint.pth'):
    """Lädt den Zustand des Modells und des Optimierers aus einer Datei.

    Parameters:
    -----------
    model (nn.Module): 
        Das Modell, in das die gespeicherten Zustände geladen werden.

    optimizer (torch.optim.Optimizer): 
        Der Optimierer, dessen Zustand geladen wird.

    filename (str): 
        Der Name der Datei, aus der der Checkpoint geladen wird.

    **TODO**:
    Versuche, den Checkpoint mit `torch.load()` zu laden.
    Wenn die Datei nicht gefunden wird, gib eine entsprechende Fehlermeldung aus und starte ohne gespeicherten Zustand.
    Wenn der Checkpoint geladen wird, versuche, den Zustand des Modells und des Optimizers zu laden.
    Du kannst `model.load_state_dict()` und `optimizer.load_state_dict()` verwenden um die Zustände ins Modell zu laden.
    Wenn ein Fehler beim Laden auftritt, gib eine Fehlermeldung aus und starte ohne gespeicherten Zustand.
    Gibt die aktuelle Epoche zurück, die im Checkpoint gespeichert ist.
    """
    pass
    
if __name__ == "__main__":
    training_set, validation_set = load_data()

    # Initialisierung des Modells, Loss-Kriteriums und Optimierers
    model = CNNNetwork().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Checkpoint laden, falls vorhanden
    dirname = os.path.dirname(os.path.abspath(__file__))
    chkpt_path = os.path.join(dirname, 'checkpoint.pth')

    ep = load_checkpoint(model, optimizer, chkpt_path)
    if ep > 0:
        print(f"Checkpoint geladen, fortsetzen bei Epoche {ep}.")

    # Das Modell trainieren
    for n in range(ep, ep + 30):
        epoch(model, n, True, training_set, criterion, optimizer)
        epoch(model, n, False, validation_set, criterion, optimizer)

        # Checkpoint nach jeder Epoche speichern
        save_checkpoint(model, optimizer, n + 1, chkpt_path)

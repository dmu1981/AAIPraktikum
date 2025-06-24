import torch 
from torch import nn
import os
from misc import CNNNetwork, load_data, epoch, save_checkpoint, load_checkpoint, DEVICE, LR

from torch.utils.tensorboard import SummaryWriter

def create_writer():
  dirname = os.path.dirname(os.path.abspath(__file__))
  board_path = os.path.join(dirname, 'runs')
  
  return SummaryWriter(log_dir=board_path)

def log_metrics(writer, metrics):
    """Loggt Metriken in TensorBoard."""
    tag = 'train' if metrics['train'] else 'validation'
    writer.add_scalar(f'{tag}/loss', metrics['loss'], metrics['step'])
    writer.add_scalar(f'{tag}/accuracy', metrics['accuracy'], metrics['step'])

if __name__ == "__main__":  
    training_set, validation_set = load_data()

    # Initialisierung des Modells, Loss-Kriteriums und Optimierers
    model = CNNNetwork().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)    # Checkpoint laden, falls vorhanden
    
    # Checkpoint laden, falls vorhanden
    dirname = os.path.dirname(os.path.abspath(__file__))
    chkpt_path = os.path.join(dirname, 'checkpoint.pth')

    ep = load_checkpoint(model, optimizer, chkpt_path)
    if ep > 0:
        print(f"Checkpoint geladen, fortsetzen bei Epoche {ep}.")

    # Das Modell trainieren
    writer = create_writer()

    for n in range(ep, ep + 30):
        epoch(model, n, True, training_set, criterion, optimizer, log_clbk=lambda metrics: log_metrics(writer, metrics))
        epoch(model, n, False, validation_set, criterion, optimizer, log_clbk=lambda metrics: log_metrics(writer, metrics))

        # Checkpoint nach jeder Epoche speichern
        save_checkpoint(model, optimizer, n + 1, chkpt_path)

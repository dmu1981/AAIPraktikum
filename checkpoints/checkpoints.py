import torch
from torch import nn
from misc import DEVICE, CNNNetwork, load_data, epoch

LR = 0.01 # Lernrate

if __name__ == "__main__":
    # Initialisierung des Modells, Loss-Kriteriums und Optimierers
    model = CNNNetwork().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    training_set, validation_set = load_data()

    # Das Modell trainieren
    model.train()
    for n in range(1, 30):
        epoch(model, n, True, training_set, criterion, optimizer)
        epoch(model, n, False, validation_set, criterion, optimizer)

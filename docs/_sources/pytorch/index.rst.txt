PyTorch - Grundlagen
====================

Während diesem Semester werden wir uns intensiv mit PyTorch beschäftigen. 
Sie benötigen eine funktionierende Installation von PyTorch, um die Aufgaben zu bearbeiten.
Sie sollen die offizielle Installationsanleitung von PyTorch befolgen, die Sie `hier <https://pytorch.org/get-started/locally/>`_ finden.

Installieren Sie PyTorch mit der passenden Konfiguration für Ihre Hardware und Ihr Betriebssystem. 
Achten Sie darauf, dass Sie die richtige Version von CUDA auswählen, wenn Sie eine NVIDIA-GPU verwenden möchten.

Überprüfen Sie anschließend ihre Installation, indem Sie die folgenden Befehle in einem Python-Interpreter ausführen:

.. code:: python
  
    import torch
    print(torch.__version__)
    print(torch.cuda.is_available())

Wenn die Installation erfolgreich war, sollten Sie die PyTorch-Version und den Status der CUDA-Verfügbarkeit sehen.

**Aufgabe 1**: Ein einfaches neuronales Netzwerk erstellen
----------------------------------------------------------

In dieser Aufgabe werden Sie ein einfaches neuronales Netzwerk mit PyTorch erstellen. 
Folgen Sie den Schritten in der offiziellen Dokumentation, um ein neuronales Netzwerk zu definieren, 
es zu trainieren und die Leistung zu bewerten.

Sie arbeiten in der Datei :file:`pytorch/simple.py`.

Wir wollen das s.g. XOR-Problem lösen, bei dem das Netzwerk lernen soll, die logische XOR-Funktion zu approximieren.
Sie sollen ein Netzwerk mit mindestens einem versteckten Layer erstellen, das in der Lage ist, diese Aufgabe zu bewältigen.

Definieren Sie zunächst die Trainingsdaten für das XOR-Problem:

.. math::

    \begin{align*}
    \text{Input} & : \begin{pmatrix}
    0 & 0 \\
    0 & 1 \\
    1 & 0 \\
    1 & 1
    \end{pmatrix} \\
    \text{Output} & : \begin{pmatrix}
    0 \\
    1 \\
    1 \\
    0
    \end{pmatrix}
    \end{align*}

Legen Sie dazu PyTorch-Tensoren für die Eingaben und Ausgaben an. Achten Sie darauf die Tensoren direkt
im richtigen Datentyp zu erstellen, z.B. `torch.float32` für die Eingaben und `torch.long` für die Zielklassen. 
Legen Sie diese Tensoren auf das richtige Gerät (CPU oder GPU) ab, abhängig von Ihrer Hardware. Schauen Sie
in die `offizielle Dokumentation <https://docs.pytorch.org/docs/stable/tensors.html>`_, wie Sie dies tun können.


.. admonition:: Lösung anzeigen
  :class: toggle

  .. code-block:: python 

    training_data = torch.tensor(
      [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float32, device=DEVICE
    )
    labels = torch.tensor([0, 1, 1, 0], dtype=torch.long, device=DEVICE)

**Aufgabe 2**: Das Netzwerk definieren
--------------------------------------

Implementieren Sie nun die Klasse :class:`SimpleNetwork`, die ein einfaches neuronales Netzwerk mit einem versteckten Layer definiert.
PyTorch Module sind Klassen, die von `torch.nn.Module` erben. Sie managen das registrieren von trainierbaren Parametern
und bieten eine einfache Schnittstelle für das Training und die Vorhersage. 

Es ist zwingend notwendig, dass Sie die `forward`-Methode implementieren,
die die Vorwärtsausbreitung des Netzwerks definiert. Da unser Netzwerk trainierbare Parameter hat müssen Sie auch die 
`__init__`-Methode implementieren, die die Struktur des Netzwerks definiert. 
Rufen Sie **immer** in der `__init__`-Methode die `super().__init__()`-Methode auf, um die Basisklasse zu initialisieren.  
Verwenden Sie dann `torch.nn.Linear <https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html>`_ für die Definition der Schichten des Netzwerks.

.. autoclass:: pytorch.simple.SimpleNetwork
   :members:
   :special-members: __init__

.. admonition:: Lösung für den Konstruktor __init__ anzeigen
  :class: toggle

  .. code-block:: python 
      
      def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 2)

.. admonition:: Lösung für die Forward-Methode anzeigen
  :class: toggle

  .. code-block:: python

      def forward(self, x):
          x = self.fc1(x)
          x = torch.relu(x)
          x = self.fc2(x)
          return x

**Aufgabe 3**: Das Netzwerk trainieren
--------------------------------------

Implementieren Sie dann die Methode

.. autofunction:: pytorch.simple.train_model

um das Netzwerk zu trainieren.

.. admonition:: Lösung für die Forward-Methode anzeigen
  :class: toggle

  .. code-block:: python

      def train_model(model, data, labels, criterion, optimizer, epochs=8000):
          for epoch in range(epochs):
              optimizer.zero_grad()
              outputs = model(data)
              loss = criterion(outputs, labels)
              loss.backward()
              if epoch % 1000 == 999:
                  print(f"Epoch {epoch+1}, Loss: {loss.item()}")
              optimizer.step()

**Aufgabe 4**: Das Netzwerk testen
-----------------------------------

Starten Sie das Training des Netzwerks, indem Sie das Skript :file:`pytorch/simple.py` ausführen.

.. code-block:: bash

    python pytorch/simple.py
    
    Epoch 1000, Loss: 0.03407074883580208
    Epoch 2000, Loss: 0.009740689769387245
    Epoch 3000, Loss: 0.005211123265326023
    Epoch 4000, Loss: 0.0034439843147993088
    Epoch 5000, Loss: 0.0025294627994298935
    Epoch 6000, Loss: 0.001979016000404954
    Epoch 7000, Loss: 0.0016146576963365078
    Epoch 8000, Loss: 0.0013573128962889314
    Training complete.
    tensor([[ 3.0257, -3.4460],
            [-3.0238,  3.6015],
            [-2.5722,  4.1114],
            [ 3.9732, -2.6664]], device='cuda:0')

**Musterlösung**:

:doc:`simple_source`
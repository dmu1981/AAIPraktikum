PyTorch - Convolutional Neural Networks
=======================================

Im Jahr 1998 veröffentlichte
`Yann LeCun <https://de.wikipedia.org/wiki/Yann_LeCun>`_ das erste Convolutional Neural Network (CNN) mit dem Namen `LeNet-5 <http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf>`_.

Er war der erste, der die Architektur eines CNNs definierte und es erfolgreich auf die Erkennung von handgeschriebenen Ziffern 
anwandte.

.. image:: ../pytorch/YannLeCun.jpg
   :width: 300px
   :align: center
   :alt: Yann LeCun (Quelle: `https://de.wikipedia.org/wiki/Yann_LeCun`_)

In dieser Aufgabe werden Sie ein Convolutional Neural Network (CNN) mit PyTorch erstellen, das auf dem CIFAR-100-Datensatz trainiert wird.

`CIFAR-100 <https://www.kaggle.com/datasets/fedesoriano/cifar100>`_ ist ein Datensatz, der 100 verschiedene Klassen von Bildern enthält, darunter Tiere, Fahrzeuge und alltägliche Objekte.

.. image:: ../pytorch/cifar100.png
   :width: 600px
   :align: center
   :alt: CIFAR-100 

In dieser Aufgabe arbeiten Sie in der Datei :file:`pytorch/cifar100.py`.

**Aufgabe 1**: Data Augmentation Pipeline
-----------------------------------------

In dieser Aufgabe werden Sie eine Data Augmentation Pipeline für den CIFAR-100-Datensatz erstellen.
Data Augmentation ist eine Technik, die verwendet wird, um die Vielfalt der Trainingsdaten zu erhöhen, indem verschiedene Transformationen auf die Bilder angewendet werden.
Dies kann helfen, die Generalisierungsfähigkeit des Modells zu verbessern und Overfitting zu reduzieren.
Sie können verschiedene Transformationen wie zufällige Drehungen, Skalierungen, Spiegelungen und Farbänderungen anwenden.
PyTorch bietet eine einfache Möglichkeit, Data Augmentation mit der `torchvision.transforms <https://docs.pytorch.org/vision/0.9/transforms.html>`_-Bibliothek zu implementieren.

Dabei verwendet man in der Regel die Klasse `torchvision.transforms.Compose <https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.Compose.html>`_, um mehrere Transformationen zu kombinieren.
Sie sollten eine Pipeline erstellen, die mindestens folgende Transformationen enthält:

- **Konvertierung in Tensor**: Die `torchvision.transforms.ToTensor() <https://docs.pytorch.org/vision/main/generated/torchvision.transforms.ToTensor.html>`_-Klasse konvertiert die Bilder in PyTorch-Tensoren, die für das Training verwendet werden können.
- **Zufällige horizontale Spiegelung**: Die `torchvision.transforms.RandomHorizontalFlip() <https://docs.pytorch.org/vision/main/generated/torchvision.transforms.RandomHorizontalFlip.html>`_-Klasse spiegelt die Bilder zufällig horizontal, was bei vielen Objekten sinnvoll ist. Verwenden Sie `p=0.5`.
- **Zufällige Drehung**: Die Klasse `torchvision.transforms.RandomRotation() <https://docs.pytorch.org/vision/main/generated/torchvision.transforms.RandomRotation.html#torchvision.transforms.RandomRotation>`_ dreht die Bilder um einen zufälligen Winkel, um die Robustheit des Modells gegenüber verschiedenen Orientierungen zu erhöhen. Verwenden Sie `degrees=15`, um die Bilder um bis zu 15 Grad (plus oder minus) zu drehen.
- **Zufälliger Zuschnitt**: Die `torchvision.transforms.RandomCrop() <https://docs.pytorch.org/vision/main/generated/torchvision.transforms.RandomCrop.html>`_-Klasse schneidet die Bilder zufällig aus, um die Robustheit des Modells gegenüber verschiedenen Bildausschnitten zu erhöhen. Verwenden Sie `size=(32, 32)` und `padding=4`, um die Bilder auf die Größe 32x32 zu beschneiden und einen Rand von 4 Pixeln hinzuzufügen.
- **Normalisierung**: Die `torchvision.transforms.Normalize() <https://docs.pytorch.org/vision/main/generated/torchvision.transforms.Normalize.html>`_-Klasse normalisiert die Bilder, um die Pixelwerte in einen bestimmten Bereich zu bringen. Verwenden Sie `mean = (0.5, )` und `std = (0.5, )`, um die Bilder zu normalisieren. 

**Achtung**: Erstellen Sie auch eine zweite Pipeline für die Validierung, die nur die Konvertierung in Tensor und die Normalisierung enthält, ohne Data Augmentation.

.. admonition:: Lösung anzeigen
  :class: toggle

  .. code-block:: python 

    training_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomCrop(size=(32, 32), padding=4),
        transforms.Normalize((0.5,), (0.5,))
    ])

    validation_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

**Aufgabe 2**: Der Datensatz laden
----------------------------------

Nun müssen Sie den CIFAR-100-Datensatz laden. PyTorch bietet eine einfache Möglichkeit, diesen Datensatz zu laden und in Trainings- und Validierungssets zu unterteilen.
Sie können den Datensatz mit der Klasse `torchvision.datasets.CIFAR100` laden. 

Instantieren Sie zwei `torchvision.datasets.CIFAR100 <https://docs.pytorch.org/vision/main/generated/torchvision.datasets.CIFAR100.html>`_-Objekte: eines für das Training und eines für die Validierung.
Verwenden Sie die `root`-Option, um den Speicherort des Datensatzes anzugeben, und die `download`-Option, um den Datensatz herunterzuladen, falls er nicht vorhanden ist.
Verwenden Sie die `train`-Option, um anzugeben, ob es sich um das Trainings- oder Validierungsset handelt.

.. admonition:: Lösung anzeigen
  :class: toggle

  .. code-block:: python 

    training_data = datasets.CIFAR100(
        root="data/cifar100",
        train=True,
        download=True,
        transform=training_transform
    )

    validation_data = datasets.CIFAR100(
        root="data/cifar100",
        train=False,
        download=True,
        transform=validation_transform
    )

Wrappen Sie die Datensätze in `torch.utils.data.DataLoader <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`_-Objekte, um sie in Batches laden zu können.
Ein Batch ist dabei eine Gruppe von Bildern, die gleichzeitig verarbeitet werden. 
Verwenden Sie die `batch_size`-Option, um die Größe der Batches festzulegen, und die `shuffle`-Option, um die Daten zufällig zu mischen.
Wählen Sie eine Batch-Größe zwischen 32 und 256, abhängig von Ihrer Hardware und den verfügbaren Ressourcen.


.. admonition:: Lösung anzeigen
  :class: toggle

  .. code-block:: python 

    training_set = torch.utils.data.DataLoader(training_data, batch_size=256, shuffle=True)
    validation_set = torch.utils.data.DataLoader(validation_data, batch_size=256, shuffle=False)


**Aufgabe 3**: Das Netzwerk definieren
--------------------------------------

Implementieren Sie nun die Klasse :class:`CNNNetwork`, die ein einfaches Convolutional Neural Network (CNN) mit mehreren Convolutional-Schichten und voll verbundenen Schichten definiert.    

.. autoclass:: pytorch.cifar100.CNNNetwork
   :members:
   :special-members: __init__
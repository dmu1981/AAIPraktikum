Perceptual Loss
===============

Im Jahr 2016 zeigen Justin Johnson, Alexandre Alahi und Li Fei-Fei in ihrem Paper 
"`Perceptual Losses for Real-Time Style Transfer and Super-Resolution <https://arxiv.org/pdf/1603.08155.pdf>`_" 
die Vorteile von Perceptual Loss für die Bildverarbeitung. 
Sie verwenden vortrainierte Convolutional Neural Networks (CNNs) wie VGG, um Merkmale aus Bildern zu extrahieren und diese Merkmale als Verlustfunktion zu nutzen.
Perceptual Loss misst die Differenz zwischen den extrahierten Merkmalen von zwei Bildern und ermöglicht so eine bessere Wahrnehmung von Bildinhalten im Vergleich zu 
traditionellen Verlustfunktionen wie dem pixelweisen L1-Verlust (Betragsdifferenz).

VGG16
-----
VGG16 ist ein bekanntes Convolutional Neural Network, das ursprünglich für die Bildklassifikation entwickelt wurde.
Es besteht aus 16 Schichten, die Convolutional-Layer, ReLU-Aktivierungen und Max-Pooling-Schichten enthalten.
VGG16 wurde auf dem ImageNet-Datensatz trainiert und hat sich als leistungsstark in verschiedenen Computer Vision-Aufgaben erwiesen.
In diesem Abschnitt werden wir VGG16 verwenden, um Perceptual Loss zu berechnen.  

.. image:: vgg16.png
   :width: 600px
   :align: center

Das obige Bild zeigt die Architektur von VGG16. 
Wir werden die Aktivierungen aus den Convolutional-Layern verwenden, um Merkmale zu extrahieren,
die dann für die Berechnung des Perceptual Loss verwendet werden.

Perceptual Loss mit VGG16
-------------------------

Der Perceptual Loss vergleicht nicht die Pixelwerte direkt, sondern Merkmale (Features), 
die durch ein vortrainiertes neuronales Netzwerk (z. B. VGG16) extrahiert werden. 
Dadurch lässt sich die *wahrgenommene* Ähnlichkeit zwischen Bildern besser bewerten 
als mit klassischen Fehlermaßen wie MSE.

**Berechnungsschritte**

1. Zwei Bilder werden als Eingabe verwendet: ein generiertes Bild (Output) und das Zielbild (Ground Truth).
2. Beide Bilder werden durch ein vortrainiertes VGG16-Netzwerk geleitet.
3. Die Aktivierungen aus einer oder mehreren Zwischenebenen (z. B. ``relu2_2`` oder ``relu3_3``) werden extrahiert.
4. Der **L1-Abstand** (Betragsdifferenz) zwischen den entsprechenden Feature-Maps wird berechnet.
5. Der Durchschnitt dieser Abstände ergibt den Perceptual Loss.

**Hinweise zur Implementierung**

* Das VGG16-Netzwerk wird meist ohne den Klassifikationskopf verwendet (nur bis zu einem bestimmten Layer).
* Die Gewichte des VGG-Netzes bleiben *eingefroren* (nicht trainierbar).

.. image:: perloss.png
   :width: 600px
   :align: center

Die obige Abbildung zeigt den Prozess der Berechnung des Perceptual Loss.
Die beiden zu vergleichenden Bilder (:math:`x` und :math:`G(z)` für Ground Truth und Output) werden durch das 
VGG16-Netzwerk geleitet. Die Aktivierungen aus den Convolutional-Layern werden extrahiert und der L1-Abstand zwischen den
entsprechenden Feature-Maps wird berechnet.   

Total Variation Loss
----------------------
Der *Total Variation Loss* (TV Loss) ist eine Regularisierungstechnik, die häufig in der Bildverarbeitung eingesetzt wird, um die Glätte und Konsistenz von Bildern zu fördern.
Er wird oft in Kombination mit Perceptual Loss verwendet, um die Qualität der generierten Bilder weiter zu verbessern.  
Der TV Loss misst die Variation der Pixelwerte in einem Bild und bestraft große Änderungen zwischen benachbarten Pixeln.
Dies führt zu glatteren Übergängen und reduziert Rauschen in den Bildern.
Der TV Loss wird wie folgt berechnet:

.. math::

   \text{TV}(x) = \sum_{i,j} \left( |x_{i+1,j} - x_{i,j}| + |x_{i,j+1} - x_{i,j}| \right)

Die Summe wird über alle Pixel im Bild gebildet, wobei :math:`x_{i,j}` den Pixelwert an der Position :math:`(i,j)` darstellt.
Die obige Formel berechnet die absolute Differenz zwischen benachbarten Pixeln in horizontaler und vertikaler Richtung.
Der TV Loss wird oft mit einem Gewicht multipliziert, um seinen Einfluss auf den Gesamtverlust zu steuern.

.. image:: tvloss.png
   :width: 600px
   :align: center


Image Upscaling
---------------

Die Bildvergrößerung, auch *Image Super-Resolution* genannt, ist ein zentrales Problem 
in der Computer Vision. Dabei soll aus einem kleinen, niedrig aufgelösten Bild 
eine hochaufgelöste Version rekonstruiert werden – z. B. von 64×64 auf 256×256 Pixel. 
Solche Verfahren sind in vielen Bereichen relevant: von der medizinischen Bildgebung 
über Überwachungskameras bis hin zur Restaurierung alter Fotos.

Eine der größten Herausforderungen dabei ist, dass die Aufgabe **hochgradig unterbestimmt** ist:
Aus einem kleinen Bild lassen sich unendlich viele "mögliche" große Bilder rekonstruieren – 
aber welches ist das *richtige*?

Klassische Verfahren nutzen meist den **MSE (Mean Squared Error) Loss**, der versucht, 
die Pixelwerte des rekonstruierten Bildes möglichst genau an das Original anzupassen. 
Doch dieser Ansatz hat einen entscheidenden Nachteil: Er neigt zu **verwaschenen, 
detailarmen Bildern**, da er im Zweifel lieber "mittelt", um den Fehler zu minimieren.

Genau hier setzt der **Perceptual Loss** an.

Statt einzelne Pixel zu vergleichen, bewertet er die **visuelle Ähnlichkeit** auf Basis 
von Merkmalen (Features), die ein vortrainiertes neuronales Netzwerk gelernt hat – also so, 
wie ein Mensch Unterschiede wahrnimmt.

In dieser Aufgabe wollen wir den Unterschied zwischen MSE und Perceptual Loss bei der Bildvergrößerung herausarbeiten 
und zeigen, wie Perceptual Loss zu deutlich natürlicheren, detailreicheren Ergebnissen führt als klassische 
pixelbasierte Fehlermaße. Am praktischen Beispiel wird dabei deutlich, dass nicht immer der "niedrigste Fehler" 
der menschlich überzeugendste ist – sondern die *wahrgenommene Qualität* zählt.
  

LPIPS: Ein Maß für wahrgenommene Bildqualität
---------------------------------------------

Der *Learned Perceptual Image Patch Similarity* (LPIPS) Score ist ein modernes 
Fehlermodell zur Bewertung der Ähnlichkeit zwischen zwei Bildern – basierend 
auf menschlicher Wahrnehmung.

Im Gegensatz zu klassischen Metriken wie PSNR oder MSE vergleicht LPIPS keine 
einzelnen Pixel, sondern Merkmale (*Features*), die durch ein vortrainiertes 
neuronales Netzwerk (z. B. VGG oder SqueezeNet) extrahiert werden. Dadurch kann LPIPS 
deutlich besser abschätzen, wie "ähnlich" zwei Bilder für das menschliche Auge wirken.

Der LPIPS-Wert liegt typischerweise zwischen ``0`` (identisch) und ``1`` (stark verschieden). 
Ein **niedriger LPIPS-Score** bedeutet also eine hohe visuelle Ähnlichkeit.

Gerade bei Aufgaben wie Bildvergrößerung, Stiltransfer oder GAN-Generierung 
ist LPIPS ein wertvolles Werkzeug, um die *qualitative* Leistung von Modellen 
objektiv zu bewerten – auch wenn klassische Metriken versagen.

PSNR: Klassisches Maß zur Bildqualität
--------------------------------------

Der *Peak Signal-to-Noise Ratio* (PSNR) ist ein weit verbreitetes Maß zur Bewertung 
der Qualität von rekonstruierten Bildern, insbesondere in der Bildkompression 
oder Super-Resolution. Er quantifiziert den Unterschied zwischen einem Originalbild 
und dessen rekonstruiertem Gegenstück, basierend auf dem mittleren quadratischen Fehler (MSE).

Berechnet wird PSNR folgendermaßen:

.. math::

   \text{PSNR} = 20 \cdot \log_{10} \left( \frac{\text{MAX}}{\sqrt{\text{MSE}}} \right)

Dabei ist ``MAX`` der maximale darstellbare Pixelwert (z. B. 1.0 bei normalisierten Bildern oder 255 bei 8-Bit-Graustufenbildern). Ein höherer PSNR-Wert deutet auf eine geringere Abweichung vom Originalbild hin – und damit auf eine bessere Qualität.

**Einschränkungen**

Trotz seiner weiten Verbreitung hat PSNR entscheidende Schwächen: 
Er korreliert **nicht gut mit der menschlichen Wahrnehmung**. Zwei Bilder mit hohem 
PSNR können dennoch visuell sehr unterschiedlich wirken, insbesondere bei Texturen oder 
feinen Details. Deshalb wird PSNR oft durch wahrnehmungsorientierte Metriken wie 
LPIPS oder SSIM ergänzt.

In dieser Arbeit wird PSNR als Referenzmaß genutzt, um klassische Fehlermaße mit 
perzeptuell motivierten Alternativen zu vergleichen.


PixelShuffle in PyTorch
-----------------------

Die Klasse `torch.nn.PixelShuffle <https://docs.pytorch.org/docs/stable/generated/torch.nn.PixelShuffle.html>`_ wird in neuronalen Netzen verwendet, um die räumliche Auflösung von Tensoren zu erhöhen. Sie ist besonders nützlich in Super-Resolution-Netzen, bei denen ein niedrig aufgelöstes Bild in ein hochaufgelöstes umgewandelt werden soll.

**Funktionsweise**

``PixelShuffle`` nimmt einen Eingabetensor der Form ``(N, C * r^2, H, W)`` und reorganisiert ihn in einen Tensor der Form ``(N, C, H * r, W * r)``, wobei:

- ``N``: Batchgröße
- ``C``: Anzahl der Kanäle nach dem Shuffle
- ``H, W``: Höhe und Breite
- ``r``: Upscale-Faktor

Dabei werden die zusätzlichen Kanäle genutzt, um die räumliche Auflösung zu vergrößern. Intern wird dies durch eine Umordnung (Rearrangement) der Daten erreicht, nicht durch Interpolation.

.. image:: pixelshuffle.png
   :width: 600px
   :align: center

**Beispiel**

.. code-block:: python

    import torch
    import torch.nn as nn

    # Upscale-Faktor
    r = 2
    pixel_shuffle = nn.PixelShuffle(upscale_factor=r)

    # Beispiel-Input: (1, 4, 2, 2) → 4 Kanäle = 1 Kanal × 2^2
    input = torch.randn(1, 4, 2, 2)
    output = pixel_shuffle(input)

    print(output.shape)  # Ausgabe: torch.Size([1, 1, 4, 4])

**Verwendung**

PixelShuffle wird oft in Decoder-Architekturen oder Autoencodern eingesetzt, um räumliche Auflösung effizient zu erhöhen, ohne auf kostenintensive Upsampling-Operationen wie ``ConvTranspose2d`` oder ``Bilinear Upsampling`` zurückzugreifen.



**Aufgabe 1**: Perceptual Loss und Total Variation Loss implementieren
----------------------------------------------------------------------

In dieser Aufgabe implementieren Sie den Perceptual Loss unter Verwendung des VGG16-Netzwerks sowie den Total Variation Loss.

Sie arbeiten in der Datei :file:`perceptualloss/perceptual.py`.

Zunächst müssen Sie das VGG16-Netzwerk laden und die erforderlichen Layer extrahieren.
Verwenden Sie dazu die `torchvision.models`-Bibliothek, um das vortrainierte VGG16-Modell zu laden.
Wir werden nur bestimmte Layer des VGG16-Netzwerks verwenden, um die Merkmale zu extrahieren,

Das VGG16-Netzwerk ist in PyTorch bereits vortrainiert und kann direkt verwendet werden.
Sie können es mit `torchvision.models.vgg16(pretrained=True)` laden. 
Eine Liste mit allen Features können Sie mit `model.features` abrufen.

Deaktiveren Sie die Gradientenberechnung für das VGG16-Netzwerk, da wir es nicht trainieren wollen.
Setzen Sie dazu `requires_grad` auf `False` für alle Parameter des Modells.

Implementieren Sie nun die Klasse `VGG16PerceptualLoss` in der Datei `perceptualloss/perceptual.py`.

.. autoclass:: perceptual.VGG16PerceptualLoss
   :members:
   :special-members: __init__, forward
   :undoc-members:
   :show-inheritance:

.. admonition:: Musterlösung für den Konstruktur __init__ anzeigen
   :class: toggle

   .. code-block:: python

       class VGG16PerceptualLoss(nn.Module):
         def __init__(self):
           super(VGG16PerceptualLoss, self).__init__()
           self.vgg = vgg16(pretrained=True).features[:16].eval()

           for param in self.vgg.parameters():
               param.requires_grad = False

           self.l1_loss = nn.L1Loss()

.. admonition:: Musterlösung für den Forward-Pass anzeigen
   :class: toggle

   .. code-block:: python

       def forward(self, output, target):
          output = torch.nn.functional.interpolate(output, size=(224, 224), mode='bilinear', align_corners=False)
          target = torch.nn.functional.interpolate(target, size=(224, 224), mode='bilinear', align_corners=False)

          f1 = self.vgg(output)

          with torch.no_grad():
           f2 = self.vgg(target)

          return self.l1_loss(f1, f2)

Implementieren Sie nun die Klasse `TVLoss`, ebenfalls in der Datei `perceptualloss/perceptual.py`.          

.. autoclass:: perceptual.TVLoss
   :members:
   :special-members: __init__, forward
   :undoc-members:
   :show-inheritance:

.. admonition:: Musterlösung anzeigen
   :class: toggle

   .. code-block:: python

         class TVLoss(nn.Module):
            def __init__(self):
               super(TVLoss, self).__init__()

            def forward(self, img):
               return (
                     torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))
                     + torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]))
               )

**Aufgabe 2**: Super-Resolution CNN definieren
----------------------------------------------
In dieser Aufgabe implementieren Sie ein einfaches Super-Resolution CNN welches wir später mit dem Perceptual Loss trainieren werden.

Wir verwenden eine Abwandlung der ResNet-Blöcke aus den vorherigen Aufgaben, um ein Super-Resolution-Netzwerk zu erstellen.

Ein einzelner ResNet-Block wird dabei wie folgt aussehen:

.. image:: resblock.png
   :width: 600px
   :align: center

Der ResNet-Block ist bereits implementiert und kann verwendet werden.

.. automethod:: perceptualloss.misc.ResNetBlock.__init__

Die Größe der Faltungsmasken ist dabei konstant 7x7 mit einem Padding von 3, so dass die räumliche 
Dimension der Eingabe gleich bleibt. Die Batch-Normalization-Schichten wurden hinter die nicht-linearität geschoben,
um normalisierte Aktivierungen zu erhalten. Die Shortcut-Verbindung passt die Dimension der Eingabe an die des Outputs an.

Wir verwenden vier aufeinanderfolgende ResNet-Blöcke mit zunächst 3 auf 16, dann 16 auf 32, dann 32 auf 64 und schließlich 64 auf 128 Kanäle.
Anschließend verwenden wir ein PixelShuffle-Layer mit einem Upscale-Faktor von 2, um die räumliche Auflösung des Bildes zu verdoppeln.
Dabei wird die Zahl der Kanäle auf 32 reduziert. Zum Schluß verwenden wir eine klassische Faltung mit einer weiteren 7x7 Maske, welche die 32 Kanäle auf 3 reduziert.

Damit das Netzwerk nicht zunächst die Identitätsfunktion lernen muß addieren wir die mit `torch.nn.Upsample <https://pytorch.org/docs/stable/generated/torch.nn.Upsample.html>`_ 
hochskalierte Eingabe zum Output des Netzwerks hinzu. Der Faltungsteil muß also nur lernen die Details zu rekonstruieren, die in der hochskalierten Version fehlen.
Entsprechend wichtig ist es auch das die letzte Faltung keine nicht-linearität enthält, damit die Addition mit der hochskalierten Eingabe funktioniert.


.. image:: srcnn2.png
   :width: 600px
   :align: center 

Implementieren Sie nun die Klasse `SRCNN` in der Datei `perceptualloss/upscale2x.py`, die das Super-Resolution-Netzwerk definiert.   

.. autoclass:: upscale2x.Upscale2x
   :members:
   :special-members: __init__, forward
   :undoc-members:
   :show-inheritance:

.. admonition:: Musterlösung für den Konstruktur __init__ anzeigen
   :class: toggle

   .. code-block:: python

       class Upscale2x(nn.Module):
         def __init__(self):
            super(Upscale2x, self).__init__()
            self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.model = nn.Sequential(
                  ResNetBlock(3, 16, kernel_size=7),
                  ResNetBlock(16, 32, kernel_size=7),
                  ResNetBlock(32, 64, kernel_size=7),
                  ResNetBlock(64, 128, kernel_size=7),
                  nn.PixelShuffle(upscale_factor=2),  # First upsample
                  nn.Conv2d(32, 3, kernel_size=7, padding=3),  # Final conv to reduce channels
            )

.. admonition:: Musterlösung für den Forward-Pass anzeigen
   :class: toggle

   .. code-block:: python

         def forward(self, x):
            p = self.upsample(x)
            x = p + self.model(x)
            return x

**Aufgabe 3:** Super-Resolution mit Perceptual Loss trainieren
--------------------------------------------------------------

Trainieren Sie nun das Super-Resolution-Modell mit dem Perceptual Loss. 
Sie brauchen nur das Skript `perceptualloss/upscale2x.py` auszuführen. Öffnen Sie 
ebenfalls das TensorBoard mit dem Befehl `tensorboard --logdir runs` um das Training zu überwachen.

Nach nur wenigen Epochen sollten Sie eine deutliche Verbesserung der Bildqualität sehen. In den gezeigten Bildern
ist links stets das (hoch-skalierte) Ausgangsbild zu sehen, was an den fehlenden Details sowie dem weichgezeichneten Aussehen zu erkennen ist.
Rechts sehen Sie den Ground Truth, also das nicht-skalierte Bild in Orginalauflösung.

Das mittlere Bild zeigt das Ergebnis des Super-Resolution-Netzwerks, welches deutlich mehr Details und Strukturen enthält.
      

**Aufgabe 4**: Super-Resolution mit MSE-Loss trainieren
-------------------------------------------------------

Um den Unterschied zwischen MSE-Loss und Perceptual Loss zu verdeutlichen,
trainieren Sie das Super-Resolution-Modell nochmal, diesmal allerdings mit dem MSE-Loss zwischen den 
hochskalierten und den Ground Truth Bildern.

Passen Sie das die Hauptmethode in der Datei `perceptualloss/upscale2x.py` an, um den MSE-Loss zu verwenden.

Ihre Trainingskurven sollten ähnlich aussehen wie die folgende:

.. image:: lpips_2x.png
    :width: 600px
    :align: center

.. image:: mse_2x.png
    :width: 600px
    :align: center

.. image:: psnr_2x.png
    :width: 600px
    :align: center    

Der LPIPS-Score sollte für den Perceptual Loss deutlich niedriger sein als für den MSE-Loss, 
was auf eine bessere wahrgenommene Bildqualität hinweist während der PSNR-Score und der MSE-Loss höher ist.
Dies ist typisch, da der MSE-Loss versucht, die Pixelwerte direkt zu minimieren, während der Perceptual Loss 
auf die Wahrnehmung des Bildinhalts abzielt.

In den unten stehenden Bildern sehen Sie die Ergebnisse des Trainings.
Das ganze linke Bild zeigt das (hochskalierte) Eingabebild in niedriger Auflösung.
Daneben finden Sie die Ausgabe des Upscalers, welcher mit MSE-Loss trainiert wurde gefolgt von der Ausgabe
des Upscalers, welcher mit Perceptual Loss trainiert wurde. Ganz rechts sehen Sie das Ground Truth Bild in hoher Auflösung.

Achten Sie auf feine Details und Strukturen in den Bildern.

.. image:: images/upscale2x_0.png
   :width: 600px
   :align: center

.. image:: images/upscale2x_1.png
   :width: 600px
   :align: center

.. image:: images/upscale2x_2.png
   :width: 600px
   :align: center

.. image:: images/upscale2x_3.png
   :width: 600px
   :align: center   

.. image:: images/upscale2x_4.png
   :width: 600px
   :align: center   

.. image:: images/upscale2x_5.png
   :width: 600px
   :align: center   

.. image:: images/upscale2x_6.png
   :width: 600px
   :align: center

.. image:: images/upscale2x_7.png
   :width: 600px
   :align: center   

.. image:: images/upscale2x_8.png
   :width: 600px
   :align: center   

.. image:: images/upscale2x_9.png
   :width: 600px
   :align: center   

.. image:: images/upscale2x_10.png
   :width: 600px
   :align: center   

.. image:: images/upscale2x_11.png
   :width: 600px
   :align: center

.. image:: images/upscale2x_12.png
   :width: 600px
   :align: center   

.. image:: images/upscale2x_13.png
   :width: 600px
   :align: center   

.. image:: images/upscale2x_14.png
   :width: 600px
   :align: center   

.. image:: images/upscale2x_15.png
   :width: 600px
   :align: center   

.. image:: images/upscale2x_16.png
   :width: 600px
   :align: center   

.. image:: images/upscale2x_17.png
   :width: 600px
   :align: center   

.. image:: images/upscale2x_18.png
   :width: 600px
   :align: center   

.. image:: images/upscale2x_19.png
   :width: 600px
   :align: center   

.. image:: images/upscale2x_20.png
   :width: 600px
   :align: center      

.. image:: images/upscale2x_21.png
   :width: 600px
   :align: center      

.. image:: images/upscale2x_22.png
   :width: 600px
   :align: center   

.. image:: images/upscale2x_23.png
   :width: 600px
   :align: center      

.. image:: images/upscale2x_24.png
   :width: 600px
   :align: center      

.. image:: images/upscale2x_25.png
   :width: 600px
   :align: center      

.. image:: images/upscale2x_26.png
   :width: 600px
   :align: center      

.. image:: images/upscale2x_27.png
   :width: 600px
   :align: center      

.. image:: images/upscale2x_28.png
   :width: 600px
   :align: center      

.. image:: images/upscale2x_29.png
   :width: 600px
   :align: center      

.. image:: images/upscale2x_30.png
   :width: 600px
   :align: center   

.. image:: images/upscale2x_31.png
   :width: 600px
   :align: center      

   
**Aufgabe 5**: Super-Resolution x4 
-----------------------------------

Zum Abschluß erweitern Sie das Super-Resolution-Netzwerk, um Bilder von 64x64 auf 256x256 zu skalieren.
Arbeiten Sie in der Datei `perceptualloss/upscale4x.py` und passen Sie die Architektur von vorher an, um die Eingabe von 64x64 auf 256x256 zu skalieren.
Sie können dazu die `Upscale2x`-Klasse als Basis verwenden und diese entsprechend erweitern.

.. autoclass:: upscale4x.Upscale4x
   :members:
   :special-members: __init__, forward
   :undoc-members:
   :show-inheritance:

.. admonition:: Beispiel für einen Upscaler x4 anzeigen
   :class: toggle

   .. code-block:: python

       class Upscale4x(nn.Module):
            def __init__(self):
               super(Upscale4x, self).__init__()
               self.upsample = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
               self.model = nn.Sequential(
                     ResNetBlock(3, 16, kernel_size=7),
                     ResNetBlock(16, 32, kernel_size=7),
                     ResNetBlock(32, 64, kernel_size=7),
                     ResNetBlock(64, 128, kernel_size=7),
                     ResNetBlock(128, 256, kernel_size=7),
                     nn.PixelShuffle(upscale_factor=4),  # First upsample
                     nn.Conv2d(16, 3, kernel_size=7, padding=3),  # Final conv to reduce channels
               )

            def forward(self, x):
               up = self.upsample(x)
               x = up + self.model(up)
               return x

Trainieren Sie dann ihren Upscaler sowohl mit dem Perceptual Loss als auch mit dem MSE-Loss. Vergleichen Sie wieder.

.. image:: lpips_4x.png
    :width: 600px
    :align: center

.. image:: mse_4x.png
    :width: 600px
    :align: center

.. image:: psnr_4x.png
    :width: 600px
    :align: center    

Wieder sollte der LPIPS-Score für den Perceptual Loss deutlich niedriger sein als für den MSE-Loss.

.. image:: images/upscale4x_0.png
   :width: 600px
   :align: center

.. image:: images/upscale4x_1.png
   :width: 600px
   :align: center

.. image:: images/upscale4x_2.png
   :width: 600px    
    :align: center

.. image:: images/upscale4x_3.png
   :width: 600px
   :align: center

.. image:: images/upscale4x_4.png
   :width: 600px
   :align: center

.. image:: images/upscale4x_5.png
   :width: 600px    
   :align: center    

.. image:: images/upscale4x_6.png
   :width: 600px
   :align: center

.. image:: images/upscale4x_7.png
   :width: 600px
   :align: center

.. image:: images/upscale4x_8.png
   :width: 600px    
    :align: center

.. image:: images/upscale4x_9.png
   :width: 600px
   :align: center

.. image:: images/upscale4x_10.png
   :width: 600px
   :align: center

.. image:: images/upscale4x_11.png
   :width: 600px    
   :align: center    

.. image:: images/upscale4x_12.png
   :width: 600px
   :align: center

.. image:: images/upscale4x_13.png
   :width: 600px
   :align: center

.. image:: images/upscale4x_14.png
   :width: 600px
   :align: center

.. image:: images/upscale4x_15.png
   :width: 600px
   :align: center

.. image:: images/upscale4x_16.png
   :width: 600px
   :align: center

.. image:: images/upscale4x_17.png
   :width: 600px
   :align: center

.. image:: images/upscale4x_18.png
   :width: 600px
   :align: center

.. image:: images/upscale4x_19.png
   :width: 600px
   :align: center

.. image:: images/upscale4x_20.png
   :width: 600px
   :align: center

.. image:: images/upscale4x_21.png
   :width: 600px
   :align: center

.. image:: images/upscale4x_22.png
   :width: 600px
   :align: center

.. image:: images/upscale4x_23.png
   :width: 600px
   :align: center

.. image:: images/upscale4x_24.png
   :width: 600px
   :align: center

.. image:: images/upscale4x_25.png
   :width: 600px
   :align: center

.. image:: images/upscale4x_26.png
   :width: 600px
   :align: center

.. image:: images/upscale4x_27.png
   :width: 600px
   :align: center

.. image:: images/upscale4x_28.png
   :width: 600px
   :align: center

.. image:: images/upscale4x_29.png
   :width: 600px
   :align: center

.. image:: images/upscale4x_30.png
   :width: 600px
   :align: center

.. image:: images/upscale4x_31.png
   :width: 600px
   :align: center

**Musterlösung**
----------------

:doc:`source`

















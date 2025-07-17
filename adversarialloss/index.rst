Adversarial Loss
================

In vorherigen Abschnitten haben wir gesehen, wie sich die Qualität hochskalierter Bilder durch klassische Verlustfunktionen wie den Total Variation Loss und den Perceptual Loss (z. B. mit VGG16-Features) verbessern lässt. Diese Methoden optimieren bestimmte Bildmerkmale – z. B. Glattheit oder semantische Ähnlichkeit – und führen zu visuell ansprechenderen Resultaten als einfache Pixel-basierte Fehlermaße wie MSE oder MAE.

Allerdings stößt diese Optimierung an ihre Grenzen: Auch wenn das rekonstruierte Bild laut Perceptual-Loss „ähnlich“ zum Original ist, fehlt oft der visuelle Realismus, den unser menschliches Auge bei natürlichen Bildern erwartet. Diese Diskrepanz entsteht, weil die verwendeten Verlustfunktionen nicht explizit modellieren, wie ein *realistisches Bild* aussehen sollte – sie bestrafen lediglich Abweichungen vom Original in bestimmten Merkmalen.

Hier kommt der *Adversarial Loss* ins Spiel. Anstatt das hochskalierte Bild nur auf Basis vordefinierter Fehlermaße zu optimieren, wird das Problem als ein Spiel zwischen zwei Netzwerken formuliert: einem Generator, der Bilder erzeugt, und einem Diskriminator, der versucht zu erkennen, ob ein Bild real oder künstlich ist. Diese GAN-ähnliche Architektur führt dazu, dass der Generator lernt, Bilder zu erzeugen, die nicht nur „ähnlich genug“ sind, sondern statistisch realistisch wirken – so wie echte Bilder aus der Trainingsverteilung.

Kurz gesagt: Während TV- und Perceptual Loss eher lokale oder semantische Fehler reduzieren, ermöglicht der Adversarial Loss eine globale Verbesserung des Bildrealismus. Er verschiebt den Fokus von „rekonstruiere das Original“ hin zu „täusche den Betrachter“. Für das Upscaling bedeutet das oft: schärfere Kanten, realistischere Texturen und weniger künstlich wirkende Artefakte – insbesondere in fein strukturierten Bildbereichen wie Haaren, Gras oder Texturen.

Im nächsten Abschnitt implementieren wir diesen Ansatz mithilfe eines einfachen GAN-Setups.

Das neue Setup
--------------

Im bisherigen Setup (vgl. frühere Aufgabe zum Perceptual Loss) haben wir ein einzelnes Netzwerk trainiert, um hochskalierte Bilder zu erzeugen. Dieses Netzwerk wurde mit dem Perceptual Loss optimiert, um die Qualität der Ergebnisse zu verbessern.

.. image:: srcnn_prevloss.png
   :width: 600px
   :align: center

Der Adversarial Loss hingegen erfordert zwei Netzwerke: einen Generator und einen Kritiker. Die Aufgabe des Kritikers (Diskriminator) ist es, 
zwischen echten Bildern und den vom Generator erzeugten Bildern zu unterscheiden. Er tut dies jedoch nicht in Form eine binären Klassifikation, sondern bewertet die Realitätsnähe der Bilder auf einer Skala. 
Dies ermöglicht eine differenziertere Rückmeldung an den Generator. Der Kritiker vergibt in gewisserweise eine Punktzahl für die Qualität der Bilder, anstatt nur zu sagen, ob sie echt oder gefälscht sind.
Dabei sollen möglichst realistische Bilder eine hohe Punktzahl erhalten und weniger realistische Bilder eine niedrige Punktzahl.
Der Kritiker wird also darauf trainiert, echte Bilder von gefälschten zu unterscheiden und dabei eine Art „Qualitätsbewertung“ abzugeben.

Der Generator hingegen versucht, den Kritiker zu täuschen, indem er Bilder erzeugt, die so realistisch wie möglich wirken. Er verwendet 
die bisherigen Techniken wie den Perceptual Loss, um die Qualität der Bilder zu verbessern, aber zusätzlich wird er durch den Adversarial Loss motiviert, Bilder zu erzeugen, die der
Kritiker als realistisch bewertet.

.. image:: srcnn_bothloss.png
   :width: 600px
   :align: center



Das Zero-Sum Spiel
------------------ 

Das Training mit Adversarial Loss kann als ein Zero-Sum Spiel zwischen dem Generator und dem Kritiker betrachtet werden. 
Während der Generator versucht, die Punktzahl des Kritikers zu maximieren, indem er realistische Bilder erzeugt, versucht der 
Kritiker gleichzeitig, die Punktzahl des Generators zu minimieren, indem er gefälschte Bilder korrekt identifiziert. 
Dieses Spiel führt zu einem ständigen Wettlauf zwischen den beiden Netzwerken, wobei beide versuchen, sich gegenseitig zu 
überlisten und zu verbessern.

Alternierendes Training von Generator und Kritiker
--------------------------------------------------

Damit dieses Zero-Sum-Spiel überhaupt funktioniert, müssen Generator und Kritiker abwechselnd trainiert werden. Es ist nicht sinnvoll, beide Netzwerke gleichzeitig zu optimieren, da ihre Ziele direkt gegensätzlich sind. Stattdessen wird das Training in zwei Phasen aufgeteilt, die sich in jeder Iteration (oder jedem Batch) abwechseln:

1. **Trainingsschritt für den Kritiker (Discriminator):**

   - Zunächst wird der Kritiker optimiert.

   - Er erhält echte Bilder aus dem Datensatz sowie vom Generator erzeugte (gefälschte) Bilder.

   - Ziel ist es, die Unterscheidbarkeit zwischen echten und generierten Bildern zu maximieren.

   - Die Loss-Funktion wird minimiert, wenn der Kritiker echten Bildern hohe und gefälschten Bildern niedrige Scores zuweist.

2. **Trainingsschritt für den Generator:**

   - Danach wird der Generator aktualisiert, während der Kritiker eingefroren bleibt.

   - Der Generator erzeugt neue Bilder aus Low-Resolution-Eingaben.

   - Ziel ist es nun, Bilder zu erzeugen, die der Kritiker fälschlicherweise als „echt“ klassifiziert.

   - Die Loss-Funktion wird minimiert, wenn der Generator es schafft, den Kritiker zu täuschen.

Dieses abwechselnde Training zwingt beide Netzwerke dazu, sich stetig weiterzuentwickeln: Der Kritiker wird darin besser, subtile Unterschiede zwischen real und generiert zu erkennen – und der Generator lernt, genau diese feinen Merkmale realistischer darzustellen.

Ein entscheidender Aspekt dieses Trainingsverfahrens ist das **Gleichgewicht**: Wird der Kritiker zu stark, kann der Generator kaum lernen, da er ständig „verlieren“ würde. Ist der Generator zu stark, lernt der Kritiker nichts mehr. Daher ist es in der Praxis üblich, mehrere Kritiker-Updates pro Generator-Update durchzuführen oder den Lernfortschritt beider Netzwerke sorgfältig zu überwachen.

Im nächsten Abschnitt zeigen wir, wie sich dieser Prozess konkret umsetzen lässt – sowohl algorithmisch als auch mit PyTorch-Code.

Trainingsschleife für adversariales Upscaling
---------------------------------------------

Um das alternierende Training zwischen Generator und Kritiker effizient umzusetzen, strukturieren wir unsere Trainingsschleife in zwei Hauptblöcke pro Iteration: zuerst trainieren wir den Kritiker, dann den Generator. Dies ermöglicht es, die dynamische Balance zwischen beiden Netzwerken aufrechtzuerhalten und stabil zu lernen.

Der Ablauf in vereinfachter Form sieht wie folgt aus:

1. Hole einen Batch von korrespondierenden echten **High-Resolution-Bildern und Low-Resolution-Bildern** aus dem Trainingsdatensatz.

2. **Lasse den Generator hochskalierte Bilder erzeugen**.

3. **Trainiere den Kritiker** mit echten und generierten Bildern. Dabei soll die Punktzahl für echte Bilder maximiert und für generierte Bilder minimiert werden. 

4. **Lipschitz-Bedingung**: Wende Gewicht-Clipping oder Gradient Penalty an, um die Lipschitz-Bedingung zu gewährleisten. 

5. **Trainiere den Generator**, während der Kritiker eingefroren bleibt. Dabei soll die Punktzahl für die vom Generator erzeugten Bilder maximiert werden.

Pseudocode (algorithmisch):
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   for epoch in range(num_epochs):
       for real_lr_images, real_hr_images in dataloader:

           # === Kritiker mehrfach updaten (n_critic Schritte) ===
           for _ in range(n_critic):
               freeze(generator)
               unfreeze(critic)

               fake_hr_images = generator(real_lr_images).detach()

               # Kritiker-Output: hohe Werte für echte Bilder, niedrige für generierte
               critic_real = critic(real_hr_images)
               critic_fake = critic(fake_hr_images)

               d_loss = -torch.mean(critic_real) + torch.mean(critic_fake)

               d_loss.backward()
               optimizer_critic.step()

               # Gewicht-Clipping (Lipschitz-Bedingung. Alternative: Gradient Penalty)
               for p in critic.parameters():
                   p.data.clamp_(-clip_value, clip_value)

           # === Generator-Update ===
           freeze(critic)
           unfreeze(generator)

           generated_hr = generator(real_lr_images)
           critic_output = critic(generated_hr)

           # Wasserstein-Loss (negativer Score, weil Generator maximieren will)
           w_loss = -torch.mean(critic_output)

           # Optional: zusätzlicher Perceptual und TV-Loss
           perceptual = compute_vgg_loss(generated_hr, real_hr_images)
           tv = total_variation_loss(generated_hr)

           g_loss = λ_w * w_loss + λ_p * perceptual + λ_tv * tv

           g_loss.backward()
           optimizer_generator.step()

Der Generator
-------------

Wir verwenden die gleiche Architektur wie im vorherigen Abschnitt, um hochskalierte Bilder zu erzeugen. 
Der Generator nimmt Low-Resolution-Bilder als Eingabe und gibt hochskalierte Bilder zurück.
Wir verwenden zunächst eine Kaskade von ResNet-Blöcken, gefolgt von einem Upsampling-Schritt mit PixelShuffle, um die Auflösung zu erhöhen.
Dabei verwenden wir stets 7x7-Kernel, um die Details zu erhalten und die Bilder realistisch zu gestalten. Wir beginnen mit 16 Kanälen und verdoppeln 
die Anzahl der Kanäle in jedem Block, um die Komplexität zu erhöhen. Beim Upsampling verwenden wir PixelShuffle mit einem Skalierugsfaktor von 4, 
um die Auflösung zu erhöhen und die Anzahl der Kanäle zu reduzieren. Die verbleibenden 16 Kanäle in voller Auflösung werden dann durch eine weitere 
klassischen Faltung mit einer 7x7 Maske auf 3 Kanäle reduziert, um das finale hochskalierte Bild zu erzeugen. Die letzte Faltung verwendet keine 
Aktivierungsfunktion sondern wird wieder wie vorher zu dem klassisch hoch-skalierten Bild (Bilinear Upsampling) addiert. 

.. image:: generator.png
   :width: 600px
   :align: center

Der ResNet-Block ist bereits implementiert und kann verwendet werden.

.. automethod:: adversarialloss.misc.ResNetBlock.__init__   

**Aufgabe 1**: Generator-Architektur implementieren
---------------------------------------------------

Implementieren Sie nun die Generator-Klasse, welche die ResNet-Blöcke verwendet und die oben beschriebene Architektur umsetzt.

.. autoclass:: adversarialloss.main.Generator
   :members:
   :special-members: __init__, forward
   :undoc-members:
   :show-inheritance:

.. admonition :: Lösung anzeigen
   :class: toggle

   .. code-block:: python

      class Generator(nn.Module):
        def __init__(self):
          super(Generator, self).__init__()

          self.upBilinear = nn.Upsample(
              scale_factor=4, mode="bilinear", align_corners=True
          )

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
          x = self.upBilinear(x) + self.model(x)

          return x

          
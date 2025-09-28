Adversarial Loss
================

In vorherigen Abschnitten haben wir gesehen, wie sich die Qualität hochskalierter Bilder durch klassische Verlustfunktionen wie den Total Variation Loss und den Perceptual Loss (z. B. mit VGG16-Features) verbessern lässt. Diese Methoden optimieren bestimmte Bildmerkmale – z. B. Glattheit oder semantische Ähnlichkeit – und führen zu visuell ansprechenderen Resultaten als einfache Pixel-basierte Fehlermaße wie MSE oder MAE.

Allerdings stößt diese Optimierung an ihre Grenzen: Auch wenn das rekonstruierte Bild laut Perceptual-Loss „ähnlich“ zum Original ist, fehlt oft der visuelle Realismus, den unser menschliches Auge bei natürlichen Bildern erwartet. Diese Diskrepanz entsteht, weil die verwendeten Verlustfunktionen nicht explizit modellieren, wie ein *realistisches Bild* aussehen sollte – sie bestrafen lediglich Abweichungen vom Original in bestimmten Merkmalen.

Hier kommt der *Adversarial Loss* ins Spiel. Anstatt das hochskalierte Bild nur auf Basis vordefinierter Fehlermaße zu optimieren, wird das Problem als ein Spiel zwischen zwei Netzwerken formuliert: einem Generator, der Bilder erzeugt, und einem Diskriminator, der versucht zu erkennen, ob ein Bild real oder künstlich ist. Diese GAN-ähnliche Architektur führt dazu, dass der Generator lernt, Bilder zu erzeugen, die nicht nur „ähnlich genug“ sind, sondern statistisch realistisch wirken – so wie echte Bilder aus der Trainingsverteilung.

Kurz gesagt: Während TV- und Perceptual Loss eher lokale oder semantische Fehler reduzieren, ermöglicht der Adversarial Loss eine globale Verbesserung des Bildrealismus. Er verschiebt den Fokus von „rekonstruiere das Original“ hin zu „täusche den Betrachter“. Für das Upscaling bedeutet das oft: schärfere Kanten, realistischere Texturen und weniger künstlich wirkende Artefakte – insbesondere in fein strukturierten Bildbereichen wie Haaren, Gras oder Texturen.

Im nächsten Abschnitt implementieren wir diesen Ansatz mithilfe eines einfachen GAN-Setups. Damit erreichen wir eine signifikante Steigerung der Bildqualität und können realistischere hochskalierte Bilder generieren

.. image:: shot_flower3.png
   :width: 600px
   :align: center   

Das alte Setup
--------------

Im bisherigen Setup (vgl. frühere Aufgabe zum Perceptual Loss) haben wir ein einzelnes Netzwerk trainiert, um hochskalierte Bilder zu erzeugen. Dieses Netzwerk wurde mit dem Perceptual Loss optimiert, um die Qualität der Ergebnisse zu verbessern.

.. image:: srcnn_prevloss.png
   :width: 600px
   :align: center

Das neue Setup
--------------

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

Der Kritiker
------------

Der Kritiker (Diskriminator) ist ein neuronales Netzwerk, das darauf trainiert wird, zwischen echten und generierten Bildern zu unterscheiden.
Er gibt eine Punktzahl für jedes Bild zurück, die angibt, wie realistisch es ist.
Wir verwenden eine einfache Architektur, die aus mehreren Faltungsschichten besteht, gefolgt von einer linearen Schicht, die die Punktzahl für
jedes Bild berechnet. Es ist wichtig, dass der Kritiker am Ende keine Aktivierungsfunktion verwendet, da er eine unbeschränkte Punktzahl zurückgeben soll.
Das letzte Fully-Connected Layer verwendet darüber hinaus auch keinen lernbaren Bias-Term, da der Kritiker nur eine Punktzahl zurückgeben soll und keine Klassifikation vornehmen muss.

In diesem Beispiel verwenden wir eine Architektur mit 6 Faltungsschichten und unterschiedlicher Kernelgröße. 
Die Anzahl der Kanäle verdoppelt sich jeweils währen die Auflösung durch Verwendung von Strided Convolutions halbiert wird.

Als Nicht-linearität verwenden wir `LeakyReLU <https://docs.pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html>`_ mit einem negativen Slope von 0.01, um sicherzustellen, dass der Kritiker auch bei negativen Werten aktiv bleibt.
Diese verhält sich ähnlich wie ReLU, lässt aber negative Werte mit einer anderen Steigung durch, was für den Kritiker wichtig ist, um auch negative Punktzahlen vergeben zu können.

.. image:: LeakyReLU.png
   :width: 600px
   :align: center

.. image:: critic.png
   :width: 600px
   :align: center

**Aufgabe 2**: Kritiker-Architektur implementieren
--------------------------------------------------

Implementieren Sie nun die Kritiker-Klasse, welche die oben beschriebene Architektur umsetzt.

.. autoclass:: adversarialloss.main.Critic
   :members:
   :special-members: __init__, forward
   :undoc-members:
   :show-inheritance:

.. admonition:: Lösung anzeigen
   :class: toggle

   .. code-block:: python

      class Critic(nn.Module):
        def __init__(self):
          super(Critic, self).__init__()

          self.model = nn.Sequential(
              nn.Conv2d(3, 32, kernel_size=9, stride=2, padding=4),
              nn.LeakyReLU(inplace=True),  # 32x128x128
              nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
              nn.LeakyReLU(inplace=True),  # 64x64x64
              nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
              nn.LeakyReLU(inplace=True),  # 128x32x32
              nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
              nn.LeakyReLU(inplace=True),  # 256x16x16
              nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2),
              nn.LeakyReLU(inplace=True),  # 512x8x8
              nn.Conv2d(512, 1024, kernel_size=5, stride=2, padding=2),
              nn.LeakyReLU(inplace=True),  # 1024x4x4
              nn.AvgPool2d(kernel_size=(4, 4)),  # 1024x1x1
              nn.Flatten(),
              nn.Linear(1024, 1, bias=False),  # Final output layer
          )

        def forward(self, x):
          return self.model(x)


Der Generator Loss
------------------

Der Generator ist dafür verantwortlich, aus einem niedrig aufgelösten Bild eine realistisch wirkende hochaufgelöste Version zu erzeugen. 
Um dieses Ziel zu erreichen, kombinieren wir mehrere Teilverluste zu einer einzigen Loss-Funktion.

Die Erzeugung hochqualitativer Bilder ist ein komplexes Optimierungsproblem. Wenn wir uns nur auf den adversarialen Verlust stützen würden, könnte der Generator versuchen, „realistisch wirkende“ Texturen zu erzeugen – dabei aber den tatsächlichen Inhalt des Bildes ignorieren. 
Um dieses Problem zu vermeiden, kombinieren wir folgende Komponenten:

1. **Perceptual Loss:**  
   Dieser Verlust basiert auf Feature-Maps eines vortrainierten Netzwerks (VGG16) und sorgt dafür, dass das generierte Bild semantisch mit dem Original übereinstimmt – auch wenn es pixelweise Unterschiede gibt.

2. **Total Variation (TV) Loss:**  
   Dieser Term bestraft unnötige Rauscheffekte und sorgt für glatte Übergänge in homogenen Bildregionen.

3. **Adversarialer Verlust (Wasserstein-Loss):**  
   Der Generator versucht, Bilder zu erzeugen, denen der Kritiker (Discriminator) möglichst hohe Realismus-Scores zuweist. Der WGAN-Ansatz erlaubt es, diesen Score direkt zu verwenden (kein Cross-Entropy).

4. **Zeitliche Gewichtung des Adversarial Loss:**  
   Um das Training stabil zu halten, wird der Einfluss des adversarialen Teils langsam erhöht. In den ersten Epochen dominiert der Inhalt – erst später wird auf visuelle Details und Texturen fokussiert.

Der vollständige Verlust des Generators ergibt sich also wie folgt:

.. math::

   \mathcal{L}_{\text{Generator}} = 
   \mathcal{L}_{\text{Perceptual}} + 
   \lambda_{\text{TV}} \cdot \mathcal{L}_{\text{TV}} -
   \lambda_{\text{adv}} \cdot \mathbb{E}_{x \sim P_G} [D(x)]

Dabei ist:

- :math:`D(x)` der Score des Kritikers für ein generiertes Bild
- :math:`\lambda_{\text{adv}}` eine skalierende Gewichtung, die mit jeder Epoche wächst
- :math:`\lambda_{\text{TV}}` ein fester Hyperparameter (z. B. 0.1)

Diese Loss-Funktion kombiniert das Beste aus beiden Welten: Sie erzwingt eine semantisch korrekte Rekonstruktion durch Perceptual Loss, 
ein visuell stabiles Bild durch TV-Loss – und realistische Texturen durch den adversarialen Druck des Wasserstein-Kritikers. 
Gleichzeitig wird durch die stufenweise Einblendung des Adversarial Loss verhindert, dass das Training instabil wird oder der Generator 
frühzeitig „halluziniert“.

**Aufgabe 3**: Generator Loss implementieren
--------------------------------------------


Implementieren Sie nun die GeneratorLoss-Klasse, welche die oben beschriebene Architektur umsetzt.

.. autoclass:: adversarialloss.main.GeneratorLoss
   :members:
   :special-members: __init__, forward
   :undoc-members:
   :show-inheritance:

.. admonition:: Lösung anzeigen
   :class: toggle

   .. code-block:: python

      class Critic(nn.Module):
        def __init__(self):
          super(Critic, self).__init__()

          super(GeneratorLoss, self).__init__()
          self.perceptualLoss = VGG16PerceptualLoss()
          self.mseLoss = nn.MSELoss()
          self.tvLoss = TVLoss()
          self.critic = critic


        def forward(self, x):
          adversarial_loss = 0.01 * self.critic(output).mean()

          adversarial_lambda = min(1.0, epoch / 5.0)

          content_loss = (
              self.perceptualLoss(output, target)
              + 0.1 * self.tvLoss(output)
          )

          return (
              content_loss - adversarial_lambda * adversarial_loss,
              content_loss,
              adversarial_loss,
          )


Loss-Funktion des Kritikers mit Gradient Penalty (WGAN-GP)
----------------------------------------------------------

Im Wasserstein-GAN mit Gradient Penalty (WGAN-GP) wird der Kritiker so trainiert, dass er echte Bilder 
möglichst hoch und generierte Bilder möglichst niedrig bewertet. Zusätzlich wird er dazu gezwungen, 
eine mathematische Bedingung zu erfüllen – die sogenannte **1-Lipschitz-Stetigkeit**. 
Um das zu erreichen, erweitern wir den Verlust des Kritikers um einen sogenannten **Gradient Penalty**.

**Motivation: Warum braucht der Kritiker eine Regularisierung?**

Die theoretische Grundlage des Wasserstein-GANs basiert auf der **Wasserstein-1-Distanz** – auch bekannt als **Earth Mover’s Distance**. 
Diese Distanz misst, wie viel Aufwand es kosten würde, die Wahrscheinlichkeitsverteilung der generierten Bilder 
in die Verteilung der echten Bilder zu „transportieren“. Damit diese Metrik überhaupt sinnvoll funktioniert, 
muss der Kritiker (also die Funktion :math:`D(x)`) eine spezielle Eigenschaft erfüllen:

**Er muss 1-Lipschitz-stetig sein**, d. h. seine Ausgaben dürfen sich maximal proportional zur Eingangsänderung verändern:

.. math::

   \left| D(x_1) - D(x_2) \right| \leq \left\| x_1 - x_2 \right\|

Mit anderen Worten: Der Kritiker darf keine sprunghaften Ausgaben machen – er muss gleichmäßig „bewerten“. 
Ohne diese Bedingung würde die Wasserstein-Distanz nicht mehr garantiert konvergieren und das Training könnte instabil werden.

**Was ist der Gradient Penalty?**

Wir verwenden einen eleganteren Weg um das Lipschitz-Kriterium zu erfüllen: Es wird sichergestellt, dass die **Gradienten der Kritiker-Ausgabe bezogen auf die Eingaben** möglichst nahe an Norm 1 bleiben.

Dazu berechnen wir den Gradienten der Kritikerfunktion :math:`D(\hat{x})` bezüglich einer Zwischenprobe :math:`\hat{x}`, die linear zwischen echten und generierten Bildern liegt:

.. math::

   \hat{x} = \epsilon \cdot x_{\text{real}} + (1 - \epsilon) \cdot x_{\text{fake}}

Der Gradient Penalty ist dann definiert als:

.. math::

   \mathcal{L}_{\text{GP}} = \lambda \cdot \left( \| \nabla_{\hat{x}} D(\hat{x}) \|_2 - 1 \right)^2

Diese Regularisierung bestraft Abweichungen von der idealen Norm 1. Sie wird zum Kritiker-Loss addiert und wirkt stabilisierend.

**Gesamter Kritiker-Loss in WGAN-GP**

Dein Kritiker-Loss besteht also aus zwei Teilen:

1. **Wasserstein-Loss (ohne BCE!):**

   .. math::

      \mathcal{L}_{\text{WGAN}} = - \mathbb{E}[D(x_{\text{real}})] + \mathbb{E}[D(x_{\text{fake}})]

2. **Gradient Penalty:**

   .. math::

      \mathcal{L}_{\text{GP}} = \lambda \cdot \left( \| \nabla_{\hat{x}} D(\hat{x}) \|_2 - 1 \right)^2

Der kombinierte Kritiker-Loss lautet:

.. math::

   \mathcal{L}_{\text{critic}} = \mathcal{L}_{\text{WGAN}} + \mathcal{L}_{\text{GP}}

**Aufgabe 4**: Kritiker Loss implementieren
-------------------------------------------

Implementieren Sie nun die CriticLoss-Klasse, welche den oben beschriebenen Loss umsetzt.

.. autoclass:: adversarialloss.main.CriticLoss
   :members:
   :special-members: __init__, forward, compute_gradient_penalty
   :undoc-members:
   :show-inheritance:

   
.. admonition:: Lösung anzeigen
   :class: toggle

   .. code-block:: python

      class CriticLoss(nn.Module):
        def forward(self, real, fake):
          gp, gradient_norm = self.compute_gradient_penalty(real, fake)

          loss_c = -self.critic(real).mean() + self.critic(fake).mean()
          return loss_c + gp, gradient_norm, loss_c

Der Trainingsprozess
---------------------

Der Trainingsprozess für das adversariale Upscaling folgt dem bereits beschriebenen alternierenden Ansatz. Er ist etwas komplexer
als bei den vorherigen Aufgaben, da wir nun zwei Netzwerke (Generator und Kritiker) trainieren müssen. 

Die Klasse ``UpscaleTrainer`` kapselt den gesamten Trainingsprozess für ein adversariales Super-Resolution-Modell auf Basis eines Wasserstein-GANs mit Gradient Penalty (WGAN-GP). 
Sie übernimmt die getrennte Steuerung und Optimierung des Generators und des Kritikers, inklusive Verlustberechnung, Gradientenbehandlung und 
Optimierungsschritten.

Nochmal zur Erinnerung: Ziel ist es, zwei Netzwerke miteinander im Wettbewerb zu trainieren:

- Der **Generator** erzeugt hochaufgelöste Bilder aus niedrigaufgelösten Eingaben.
- Der **Kritiker** bewertet die Bildrealismus dieser Ausgaben im Vergleich zu echten hochaufgelösten Bildern.

**Struktur und Komponenten**

Die Klasse besteht aus drei zentralen Elementen:

1. **Initialisierung (`__init__`)**

   - Initialisiert die beiden Modelle: ``Generator`` und ``Critic``.
   - Verknüpft die passenden Loss-Klassen: ``GeneratorLoss`` und ``CriticLoss``.
   - Definiert zwei separate Optimierer mit unterschiedlichen Lernraten für Generator und Kritiker.
   - Gibt die Parameteranzahl beider Modelle aus, um das Modellverständnis zu fördern.

2. **Trainingsmethoden**

   a) ``train_critic(input, target)``

      - Führt einen Trainingsschritt für den Kritiker durch.
      - Verwendet die `CriticLoss`, die den Wasserstein-Loss sowie den Gradient Penalty beinhaltet.
      - Berechnet die Gradienten, führt Backpropagation durch und optimiert den Kritiker.
      - Nutzt Gradient Clipping zur Stabilisierung der Trainingsdynamik.
      - Gibt diagnostische Werte wie den Gradientennorm und den Verlust zurück.

   b) ``train_generator(input, target, epoch)``

      - Führt einen Trainingsschritt für den Generator durch.
      - Verwendet die `GeneratorLoss`, die aus Perceptual Loss, TV-Loss und adversarial Loss besteht.
      - Der Anteil des adversarialen Verlusts wird dabei dynamisch über die Epoche skaliert.
      - Gradienten werden berechnet, geclippt und zur Optimierung verwendet.
      - Gibt alle relevanten Metriken sowie das generierte Bild zurück.

3. **Trainingszyklus (`train_batch`)**

   - Trainiert bei jedem Aufruf den Kritiker.
   - Der Generator wird nur in jeder fünften Iteration oder während der ersten Epoche trainiert.
   - Dieses Verhältnis (5:1) entspricht der Empfehlung in WGAN-GP, um dem Kritiker einen Lernvorsprung zu geben.
   - Gibt die Ergebnisse beider Trainingsschritte sowie das aktuelle Ausgabebild des Generators zurück.

**Aufgabe 5**: UpscaleTrainer implementieren
--------------------------------------------

Implementieren Sie nun die UpscaleTrainer-Klasse, welche den oben beschriebenen Trainingsprozess umsetzt.

.. autoclass:: adversarialloss.main.UpscaleTrainer
   :members:
   :special-members: __init__, train_critic, train_generator, train_batch
   :undoc-members:
   :show-inheritance:

.. admonition:: Lösung anzeigen (train_critic)
   :class: toggle

   .. code-block:: python

      def train_critic(self, input, target):
        output = self.generator(input)

        self.optimCritic.zero_grad()
        critic_loss, gradient_norm, loss_c = self.criticLoss(
            self.critic, target, output
        )
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=5.0)
        self.optimCritic.step()

        return {
            "gradient_norm": gradient_norm.mean().item(),
            "loss_c": loss_c.item(),
        }

.. admonition:: Lösung anzeigen (train_generator)
   :class: toggle

   .. code-block:: python

      def train_generator(self, input, target, epoch):
        self.optimGenerator.zero_grad()
        output = self.generator(input)

        loss, content_loss, adversarial_loss = self.generatorLoss(output, target, epoch)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
        gen_norm = torch.nn.utils.clip_grad_norm_(
            self.generator.parameters(), max_norm=1e9
        )

        self.optimGenerator.step()

        return {
            "loss": loss.item(),
            "content_loss": content_loss.item(),
            "adversarial_loss": adversarial_loss.mean().item(),
            "gradient_norm": gen_norm.item(),
            "output": output.detach().cpu() if output is not None else None,
        }

.. admonition:: Lösung anzeigen (train_batch)
   :class: toggle

   .. code-block:: python

      def train_batch(self, input, target, epoch):
        # Train Critic every step
        scoresCritic = self.train_critic(input, target)
        self.criticUpdates += 1

        # Train Generator only every 4th step
        if self.criticUpdates == 5 or epoch < 1:
            scoresGenerator = self.train_generator(input, target, epoch)
            self.criticUpdates = 0
        else:
            scoresGenerator = None

        return scoresCritic, scoresGenerator

Das Training
-----------

Um das Training zu starten, brauchen Sie nur die main.py Datei auszuführen.

.. code-block:: bash

   python main.py



Starten Sie parallel das TensorBoard, um den Fortschritt zu verfolgen:

.. code-block:: bash

   tensorboard --logdir=logs

Öffnen Sie dann Ihren Browser und gehen Sie zu `http://localhost:6006 <http://localhost:6006>`_, um die TensorBoard-Oberfläche zu sehen.

Das Training kann je nach Hardware und Datensatz einige Stunden dauern, aber Sie sollten bereits nach kurzer Zeit eine Verbesserung der Bildqualität feststellen können.
Nach etwa 15 Epochen sollten Sie bereits erste Fortschritte sehen. Die Bilder sollten zunehmend realistischer werden und weniger Artefakte aufweisen.

Die TensorBoard-Diagramme zeigen den Verlauf der verschiedenen Loss-Werte und Metriken während des Trainings.

Die wichtigsten Metriken sind:




**Der Content-Loss**
Perceptual Loss und TV-Loss ohne Adversarial Loss

 .. image:: shot_contentloss.png
   :width: 600px
   :align: center

**Der Adversarial Loss**   
Der Wasserstein-Loss des Generators, der den Kritiker täuschen soll

.. image:: shot_adversarialloss.png
   :width: 600px
   :align: center

**Der Verlust des Generators (Generator Loss)**
Summe aus Perceptual Loss, TV-Loss und Adversarial Loss

.. image:: shot_generatorloss.png
   :width: 600px
   :align: center

**Der Verlust des Diskriminators (Discriminator Loss)**

.. math::

      \mathcal{L}_{\text{WGAN}} = - \mathbb{E}[D(x_{\text{real}})] + \mathbb{E}[D(x_{\text{fake}})]

.. image:: shot_lossc.png
   :width: 600px
   :align: center

**Die Bildqualität (z.B. LPIPS, PSNR, MSE)**

.. image:: shot_lpips.png
   :width: 600px
   :align: center

.. image:: shot_psnr.png
   :width: 600px
   :align: center

.. image:: shot_mse.png
   :width: 600px
   :align: center

**Die Norm des Generator-Gradienten**

.. image:: shot_generator_gradientnorm.png
   :width: 600px
   :align: center

**Die Norm des Kritiker-Gradienten**

.. image:: shot_criticgradientnorm.png
   :width: 600px
   :align: center

**Die generierten Bilder**

Links sehen Sie das niedrig aufgelöste Eingangsbild, in der Mitte das hochskalierte Ergebnis des Generators und 
rechts das Originalbild zum Vergleich. 

.. image:: shot_flower1.png
   :width: 600px
   :align: center

.. image:: shot_flower2.png
   :width: 600px
   :align: center   

.. image:: shot_flower4.png
   :width: 600px
   :align: center   

.. image:: shot_flower5.png
   :width: 600px
   :align: center

Musterlösung
------------

:doc:`solution`
# Data Science Workshop - FHNW

Im Rahmen des Data Science Workshops an der FHNW habe ich mich intensiv mit mehreren aktuellen wissenschaftlichen Papers auseinandergesetzt. Die daraus gewonnenen Erkenntnisse dienen als Grundlage für die Hypothesen, die ich im Workshop aufstellen und validieren werde.

Folgende Papers wurden analysiert, wobei das Hauptaugenmerk auf dem letzten Paper lag:

- [Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://openreview.net/pdf?id=YicbFdNTTy)
- [CvT: Introducing Convolutions to Vision Transformers](https://openaccess.thecvf.com/content/ICCV2021/papers/Wu_CvT_Introducing_Convolutions_to_Vision_Transformers_ICCV_2021_paper.pdf)

## Hypothese: Convolutions in Vision Transformers

> Die gute Performance der CvT-Architektur lässt sich primär durch die zu Beginn ausgeführte Embedding Convolution erklären. Spätere Convolutions in und zwischen den Transformer-Blöcken tragen nur marginal zur Gesamtleistung bei und können weggelassen werden, ohne Performanceverlust bei den betrachteten Metriken zu verursachen.

### Aufbau des Experiments

Zur Validierung der Hypothese werden die bestehenden Architekturen aus den obigen Papers als auch eigene vereinfachte Varianten des CvT-Modells verglichen.

#### Vergleichsmodelle

- [CvT (Convolutional Vision Transformer) – Microsoft](https://github.com/microsoft/CvT)
- [ViT (Vision Transformer) – Google Research](https://github.com/google-research/vision_transformer)
- [ResNet50](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html)

#### Vereinfachte CvT-Modelle

Die vereinfachten CvT-Modelle übernehmen den Transformer-Block, wie er im ViT-Paper beschrieben wurde. Anstelle einer linearen Projektion als Embedding zu verwenden, kommt jedoch ein Convolutional Embedding zum Einsatz. Im letzten Versuch wird zusätzlich auch noch ein Convolutional Block zwischen den Transformer-Blöcken eingefügt.

1. **Convolutional Embedding:**\
    Das Embedding erfolgt duch eine einfache Convolution, wie diese bereits im CvT-Paper verwendet wurde.\
    <img src="./CvT-SimplifiedHead.drawio.png" alt="CvT-Modell mit Convolutional Embedding" title="CvT-Modell mit Convolutional Embedding" height="400" />

2. **ResNet-Head Embedding:**\
    Das Embedding erfolgt durch eine komplexere Kombination von Convolutions, welche von der ResNet-Architektur übernommen wurden. \
    <img src="./CvT-ResNetHead.drawio.png" alt="CvT-Modell mit ResNet-Head Embedding" title="CvT-Modell mit ResNet-Head Embedding" height="400"/>

3. **Recurrent Convolutions:**\
    Zusätzlich zu einem Convolutional Embedding (ResNet-Head oder einfache Convolution) werden auch zwischen den Transformer-Blöcken Convolutions eingeführt, um die Token-Dimensionen progressiv zu reduzieren. Dies sollte vor allem zur Reduktion der Rechenkomplexität in der Attention-Berechnung beitragen.\
    <img src="./CvT-RecurrentConvolutions.drawio.png" alt="CvT-Modell mit recurrent Convolutions" title="CvT-Modell mit recurrent Convolutions" height="400"/> \

### Dataset

Verwendet wird der [Tiny Imagenet-Datensatz](https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet):

- 200 Klassen, je 500 Trainingsbilder (insgesamt 100'000)
- Auflösung: 64×64 Pixel, RGB
- Unterteilt in: Training (100'000), Validierung (10'000), Test (10'000)

### Metriken

Zur Bewertung der Modelle werden folgende Metriken herangezogen:

- Accuracy (Hauptmetrik)
- Cross Entropy Loss
- Anzahl der trainierbaren Parameter
- Trainingszeit pro Epoche
- Inference-Zeit pro Bild

## Hypothese 2

> Durch den Einsatz von Grad-CAM kann gezeigt werden, dass sich die Convolutions in der CvT-Architektur auf die für die Klassifizierung relevanten Bildregionen konzentrieren.

### Aufbau des Experiments

Zur Validierung dieser Hypothese wird Grad-CAM (Gradient-weighted Class Activation Mapping) eingesetzt, um visuell darzustellen, auf welche Bildregionen sich unterschiedliche Modellbestandteile bei der Klassifikation konzentrieren. Grad-CAM wird dabei in die Convolutions der CvT-Architektur integriert, um aus den Gradienten der Convolutions die Heatmap zu berechnen. Diese Heatmap wird dann mit dem Originalbild überlagert, um die für die Klassifikation relevanten Bildregionen hervorzuheben.
Es wird untersucht, ob sich ein ähnliches Verfahren auch auf die Attention-Maps der Transformer-Blöcke anwenden lässt.

### Dataset

Wie in Hypothese 1 wird der Tiny ImageNet-Datensatz verwendet. Zusätzlich werden einige Beispielbilder manuell ausgewählt, bei denen die Aktivierungen visuell besonders aussagekräftig interpretiert werden können (z. B. Objekte klar vom Hintergrund getrennt).

### Metriken

TBD

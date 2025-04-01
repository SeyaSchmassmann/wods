# Data Science Workshop - FHNW

## Intro

Im Rahmen des Data Science Workshops an der FHNW habe ich mich intensiv mit mehreren aktuellen wissenschaftlichen Papers auseinandergesetzt. Die gewonnenen Erkenntnisse aus diesen Papers dienen als Grundlage für die Hypothesen, die ich im Workshop aufstellen und validieren werde.

Folgende Papers wurden analysiert, wobei das Hauptaugenmerk auf dem letzten Paper lag:

- [Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://openreview.net/pdf?id=YicbFdNTTy)
- [CvT: Introducing Convolutions to Vision Transformers](https://openaccess.thecvf.com/content/ICCV2021/papers/Wu_CvT_Introducing_Convolutions_to_Vision_Transformers_ICCV_2021_paper.pdf)

## Hypotheses

>Ein hybrides Modell, das eine CNN-Schicht zur lokalen Feature-Extraction vor der Erstellung der Patches für einen Transformer verwendet, kann eine vergleichbare Leistung zu ViT, CNN und CvT erreichen.

### Aufbau des Experiments

Das hybride Modell wird mit bestehenden Architekturen verglichen:

- [CvT (Convolutional Vision Transformer) – Microsoft](https://github.com/microsoft/CvT)
- [ViT (Vision Transformer) – Google Research](https://github.com/google-research/vision_transformer)
- Ein klassisches CNN, voraussichtlich ein **ResNet** (noch zu bestimmen)
- Das **eigene hybride Modell** (siehe unten)

#### Hybrides Modell

Das hybride Modell basiert auf der Idee aus dem Paper [Early Convolutions Help Transformers See Better](https://proceedings.neurips.cc/paper_files/paper/2021/file/ff1418e8cc993fe8abcfe3ce2003e5c5-Paper.pdf).  
Anstelle des standardmässigen Patch-Embeddings von ViTs wird eine **CNN-Schicht** eingesetzt, um **lokale Features** zu extrahieren, bevor die Bildbereiche in Patches aufgeteilt und in den Transformer eingespeist werden.  

Ziel des hybriden Modells:

- Die Vorteile von **Translationsequivarianz** (wie bei CNNs) mit der **globalen Kontextverarbeitung** von Transformern kombinieren.
- Eine Architektur entwickeln, die mit weniger Trainingsdaten robust bleibt.  

TODO: Visualisierung des Modellaufbaus erstellen.

### Daten

TBD

### Metriken

- **Accuracy** als Hauptmetrik zur Leistungsbewertung.  
- TBD: Weitere Metriken wie **Trainingszeit**, **Inference-Zeit**, **Parameteranzahl** oder **Speicherverbrauch** zur Effizienzanalyse.

## Hypothese 2

> Die unterschiedlichen Anforderungen an die Menge der Trainingsdaten zwischen CNNs und ViTs lassen sich primär durch Translationsequivarianz und den induktiven Bias erklären.

### Aufbau des Experiments

Um die Ursachen für die unterschiedlichen Datenanforderungen von CNNs und ViTs zu analysieren, werden zwei Hauptaspekte untersucht:

#### **1. Translationsequivarianz testen**

- Ein **CNN**, ein **ViT**, ein **CvT** und das **hybride Modell** werden auf einem normalen Datensatz trainiert.  
- Anschliessend wird ein modifiziertes **Test-Set mit verschobenen Objekten** erstellt (mittels Data Augmentation).  
- Wenn ein Modell auch bei verschobenen Objekten gut abschneidet, spricht das für **Translationsequivarianz**.
- Erwartung:
  - **CNNs sollten robuster** gegen Verschiebungen sein.
  - **ViTs sollten stärker einbrechen**, da sie keine inhärente Translationsequivarianz besitzen.
  - **CvTs und das hybride Modell** könnten dazwischen liegen, je nachdem, wie stark die CNN-Schicht die Translationsequivarianz erhält.

#### **2. Einfluss des Inductive Bias untersuchen**

- Die vom Transformer gelernten Features werden mit den Features eines CNNs verglichen.
- Hierzu werden die Transformer-Features extrahiert sowie die CNN-Features visualisiert.
- Erwartung:
  - TBD

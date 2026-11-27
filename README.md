# Intelligence Engineering

In diesem Uni-Projekt vergleichen wir zwei 2D-Objektdetektoren auf KITTI-Straßenszenen. Git enthält Code, die feste Bildauswahl und DVC-Zeiger. DVC verwaltet den vollständigen Rohdatensatz und den KITTI-Modellstand für die Kommandozeile.

## Projekt einrichten

Die folgenden Befehle laufen unter macOS oder Linux mit Python 3.13.

```bash
git clone https://github.com/nicolasb49/intelligence-engineering.git
cd intelligence-engineering
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
dvc pull
dvc status
```

Der voreingestellte DVC-Remote `storage` liegt im Ordner `intelligence-engineering-dvc-storage` neben dem Repository. Für einen neuen Klon auf einem anderen Rechner brauchst du Zugriff auf eine befüllte Kopie dieses Speichers. Wenn er bei dir unter einem anderen Pfad eingebunden ist, setze vor `dvc pull` die lokale Konfiguration mit `dvc remote modify --local storage url /pfad/zum/speicher`. Um ausschließlich die Rohdaten abzurufen, verwende `dvc pull data/raw/kitti.dvc`.

## Datenbestand

Die Lieferung enthielt die Bildordner in `data/data_object_image_2` und die Labels separat in `data/training/label_2`. Unter `data/raw/kitti/` liegen sie jetzt in der KITTI-Ordnerstruktur. Der Datenstand umfasst 12,57 GB.

| Ordner unter `data/raw/kitti/` | Inhalt |
| --- | --- |
| `training/image_2/` | 7.481 PNG-Straßenbilder der linken Farbkamera |
| `training/label_2/` | 7.481 gleichnamige Textdateien mit insgesamt 51.865 Annotationszeilen |
| `testing/image_2/` | 7.518 Testbilder ohne mitgelieferte Labels |

Jede Annotationszeile enthält die Kategorie, Trunkierung, Verdeckung, Beobachtungswinkel, vier 2D-Boxkoordinaten in Pixeln sowie 3D-Abmessungen, Position und Rotation. Im Bestand kommen Car, Van, Truck, Pedestrian, Person_sitting, Cyclist, Tram, Misc und DontCare vor. Die Experimente verwenden die 2D-Boxen für **Car, Truck, Cyclist und Pedestrian**. Sie lassen die übrigen Kategorien sowie DontCare-Regionen bei der Zielerstellung aus. Die Auswertung ist ein vereinfachter Funktionstest.

## Experimente ausführen

`data/kitti-smoke-split.json` hält die mit Seed 42 gezogenen Bild-IDs fest. Beide Modelle trainieren drei Epochen auf denselben **40 Bildern** und verwenden **10 weitere Bilder** zur Validierung. Alle 50 stammen aus dem beschrifteten Trainingsordner. Das entspricht rund 0,67 % dieses Ordners. Die offiziellen Testbilder bleiben für spätere Inferenz verfügbar.

```bash
python -m jupyterlab
```

- [Aufgabe 2 mit YOLOv8n](nicolas_task2_detection.ipynb) trainiert ab zufälligen Gewichten, vergleicht Vorhersagen mit den KITTI-Boxen und misst CPU-Latenz und Durchsatz für vier Batchgrößen. Der Trainingslauf erzeugt seinen kleinen YOLO-Export temporär. Ergebnisse und Gewichte liegen unter `runs/kitti/yolo*/`.
- [CNN mit SSDLite und MobileNetV3](nicolas_cnn_detection.ipynb) liest die Pixelboxen direkt und vergleicht die AP50 je Klasse sowie die mAP50 vor und nach dem Training. Das Notebook speichert die Gewichte unter `runs/kitti/cnn/`.

Mit dem in DVC gespeicherten YOLO-Checkpoint kannst du die Inferenz direkt ausführen. Die drei Varianten verarbeiten das erste Validierungsbild, ein eigenes Bild oder die 50 Bilder der festen Auswahl.

```bash
python -m src.detect_road_objects
python -m src.detect_road_objects --image /pfad/zum/bild.png
python -m src.detect_road_objects --all
```

Die Ergebnisse stehen unter `runs/kitti/task2/`. Für ein Einzelbild entstehen ein annotiertes Bild und ein JSON-Bericht, für die gesamte Auswahl ein gemeinsamer Bericht. Nach dem kurzen Training können bei der Konfidenzschwelle 0,25 alle Vorhersagen herausfallen.

## Datenstand aktualisieren

Ändere die Rohdaten unter `data/raw/kitti/` und prüfe bei einer neuen Auswahl auch die Bild-IDs in der Split-Datei. Versioniere anschließend die Daten und übertrage sie vor dem Git-Commit in den DVC-Speicher.

```bash
dvc add data/raw/kitti
dvc push data/raw/kitti.dvc
dvc status -c data/raw/kitti.dvc
git add data/raw/kitti.dvc data/kitti-smoke-split.json
```

Für einen neuen CLI-Modellstand liefert `python -m src.train_kitti_yolo` den Pfad zum letzten Checkpoint. Kopiere diesen nach `models/kitti-yolov8n.pt`, führe `dvc add models/kitti-yolov8n.pt` und `dvc push models/kitti-yolov8n.pt.dvc` aus und committe den geänderten Modellzeiger in Git.

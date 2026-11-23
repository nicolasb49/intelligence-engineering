# intelligence-engineering

Dieses Projekt entsteht im Rahmen einer Vorlesung.

## COCO + YOLOv8 mit DVC

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
dvc pull
```

Die große virtuelle Umgebung liegt unter `.venv/` und wird nicht committed.

### Datenablage

```text
data/
  raw/        Original-COCO-Dateien
  processed/  Optional: vorbereitete Daten
models/       Trainierte YOLOv8-Modelle
runs/         Lokale YOLO-Training-Ausgaben
notebooks/    Exploration und Experimente
src/          Skripte und wiederholbare Trainingslogik
```

Lade COCO-Daten nach `data/raw/coco/` und verwalte das Verzeichnis mit DVC:

```bash
dvc add data/raw/coco
git add data/raw/coco.dvc data/raw/.gitignore
git commit -m "Track COCO dataset"
dvc push
```

Neue Datenstände werden so gemeinsam genutzt:

```bash
git pull
dvc pull
```

Nach Änderungen an den Daten:

```bash
dvc add data/raw/coco
dvc push
git add data/raw/coco.dvc
git commit -m "Update COCO dataset"
git push
```

### Lokales DVC-Remote

Das konfigurierte lokale Remote liegt außerhalb des Git-Repositories:

```text
../intelligence-engineering-dvc-storage
```

Für gemeinsame Arbeit mit mehreren Rechnern muss dieser Pfad durch einen gemeinsam erreichbaren Speicher ersetzt werden, zum Beispiel S3, Azure Blob Storage oder SSH.

### YOLOv8-Beispiel

Ein COCO-Dataset muss eine YOLO-kompatible `dataset.yaml` enthalten. Danach kann ein Training gestartet werden:

```bash
yolo detect train model=yolov8n.pt data=data/processed/coco.yaml epochs=100 imgsz=640
```

Trainingsartefakte werden standardmäßig unter `runs/` abgelegt. Für eine reproduzierbare Trainingspipeline kann die DVC-Konfiguration in `dvc.yaml` ergänzt werden.

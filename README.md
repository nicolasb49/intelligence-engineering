# Intelligence Engineering

In diesem Uni-Projekt arbeiten wir mit einem Ausschnitt aus COCO 2017 für die Objekterkennung mit YOLOv8. Der gemeinsame Ausgangsstand umfasst 50 Bilder aus dem COCO-Trainingssplit und 739 Bounding Boxes der Klassen `car`, `truck`, `bicycle` und `person`. Git versioniert den Code, die Bildauswahl und den DVC-Verweis. DVC verwaltet den fertigen Export.

## Projekt einrichten

Voraussetzungen sind Git, Python 3.13 und eine Internetverbindung für die Erstinstallation. Die folgenden Befehle sind für macOS und Linux.

```bash
git clone https://github.com/nicolasb49/intelligence-engineering.git
cd intelligence-engineering
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Den gemeinsamen Datenstand laden

**Auf einem neuen Rechner** lässt sich der Datenstand direkt aus COCO aufbauen. Die 50 Bild-IDs stehen in `src/coco-2017-train-50.txt`. FiftyOne lädt beim ersten Lauf das COCO-Annotationsarchiv und die ausgewählten JPEGs. Bereits vorhandene Downloads verwendet es wieder. Ein erneuter Export ersetzt den bisherigen Exportordner.

```bash
python -m src.download_coco_subset
dvc commit data/processed/coco-2017-yolo-50.dvc
dvc status
```

`dvc commit` prüft den Export gegen die versionierte Prüfsumme und übernimmt ihn in den lokalen DVC-Cache. Fragt DVC nach einer Bestätigung für geänderte Daten, mit `n` abbrechen und die Abweichung prüfen. Bei identischem Stand meldet `dvc status` „Data and pipelines are up to date.“

**Mit Zugriff auf einen befüllten DVC-Speicher** genügt nach der Installation ein Pull. Er stellt den im Git-Commit festgelegten Export wieder her.

```bash
dvc pull
dvc status
```

Das voreingestellte Remote `storage` ist ein lokaler Ordner namens `intelligence-engineering-dvc-storage` neben dem Git-Repository. Auf einem anderen Rechner ist dieser Speicher erst verfügbar, wenn das Team ihn dorthin kopiert oder gemeinsam einbindet. Für ein eingebundenes Teamlaufwerk lässt sich der Pfad pro Klon setzen.

```bash
dvc remote modify --local storage url /absoluter/pfad/zum/team-speicher
```

Den Beispielpfad durch den tatsächlichen Speicherort ersetzen. Ein Teammitglied mit geprüftem lokalem Datenstand befüllt diesen Speicher einmal mit `dvc push`. Danach können alle mit Zugriff neue Stände mit `git pull` und anschließend `dvc pull` übernehmen.

## Mit den Daten arbeiten

| Pfad unter `data/processed/coco-2017-yolo-50/` | Inhalt |
| --- | --- |
| `images/train/` | 50 JPEGs mit der COCO-Bild-ID im Dateinamen |
| `labels/train/` | Eine YOLO-Textdatei pro Bild mit Klassen-ID, Mittelpunkt, Breite und Höhe in normierten Koordinaten |
| `dataset.yaml` | Relativer Bildpfad und Klassenreihenfolge von 0 (`car`) bis 3 (`person`) |

Der Export enthält ausschließlich Trainingsbilder. Für ein Training mit separater Validierung zuerst Validierungsbilder festlegen und den zugehörigen `val`-Pfad in der Dataset-YAML ergänzen.

## Einen neuen Datenstand teilen

Wenn sich die Bildauswahl oder Aufbereitung ändert, die Bildliste oder den Exportcode anpassen und das Skript erneut ausführen. Den aktualisierten Export im gemeinsamen DVC-Speicher ablegen, bevor der zugehörige Git-Commit veröffentlicht wird.

```bash
dvc add data/processed/coco-2017-yolo-50
dvc push
git add data/processed/coco-2017-yolo-50.dvc src/coco-2017-train-50.txt src/download_coco_subset.py
git commit -m "Update COCO subset"
git push
```

# Intelligence Engineering

In diesem Uni-Projekt nutzen wir einen Ausschnitt aus COCO 2017 für die Objekterkennung mit YOLOv8. Der aktuelle Datenstand umfasst **50 Trainingsbilder und 739 Bounding Boxes** für die Klassen `car`, `truck`, `bicycle` und `person`.

Git liefert den Code, die Liste der Bild-IDs und die Prüfsumme des zugehörigen Datenstands. Mit DVC lässt sich prüfen, ob die Bilder und Annotationen auf deinem Rechner zu diesem Git-Stand passen.

## Zum ersten Mal einrichten

Du brauchst Git, Python 3.13 und eine Internetverbindung. Die folgenden Befehle sind für macOS und Linux. Klone das Projekt und richte eine virtuelle Python-Umgebung im Repository ein.

```bash
git clone https://github.com/nicolasb49/intelligence-engineering.git
cd intelligence-engineering
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Führe anschließend **einen** der beiden Wege unter „Datenstand herstellen“ aus. Alle weiteren Befehle laufen im Projektverzeichnis mit aktivierter Umgebung.

## Einen vorhandenen Klon aktualisieren

Wechsle im Terminal in dein Projektverzeichnis. Aktiviere die vorhandene Umgebung, hole den aktuellen Stand von `main` und gleiche die installierten Pakete mit der aktualisierten `requirements.txt` ab.

```bash
source .venv/bin/activate
git switch main
git pull --ff-only
python -m pip install -r requirements.txt
```

Liegt dein bisheriger Export noch unter `data/processed/`, verschiebe ihn nach dem Pull einmal an den neuen Ort. So kannst du die vorhandenen Dateien weiterverwenden.

```bash
if [ -d data/processed/coco-2017-yolo-50 ] && [ ! -e data/coco-2017-yolo-50 ]; then
    mv data/processed/coco-2017-yolo-50 data/
fi
```

Stelle anschließend den passenden Datenstand her. Wenn du Zugriff auf den DVC-Speicher hast, verwende Weg B. Ohne diesen Zugriff kannst du zuerst `dvc status` ausführen. Meldet DVC bereits `Data and pipelines are up to date.`, bist du fertig. Fehlt der Export oder weicht er vom neuen Git-Stand ab, kannst du ihn mit Weg A wiederherstellen. Das gilt auch, wenn DVC `not in cache` meldet.

## Datenstand herstellen

### Weg A · Aus COCO aufbauen

Diesen Weg kannst du auf jedem neu eingerichteten Rechner ohne Zugang zum Teamspeicher verwenden. Das Skript liest die 50 Bild-IDs aus `data/coco-2017-train-50.txt`. FiftyOne lädt das COCO-Annotationsarchiv und nur die ausgewählten JPEGs. Vorhandene Downloads verwendet es wieder. Anschließend exportiert das Skript die Bilder und ihre Bounding Boxes im YOLO-Format. **Ein erneuter Lauf ersetzt den bisherigen Exportordner.** Sichere eigene Änderungen darin, falls du sie behalten möchtest.

```bash
python -m src.download_coco_subset
dvc commit data/coco-2017-yolo-50.dvc
dvc status
```

`dvc commit` vergleicht den Export mit der in Git gespeicherten Prüfsumme und übernimmt ihn in den lokalen DVC-Cache. Fragt DVC, ob du geänderte Daten übernehmen möchtest, antworte mit `n` und prüfe die Abweichung. Bei erfolgreicher Wiederherstellung meldet `dvc status` am Ende `Data and pipelines are up to date.`.

### Weg B · Aus dem DVC-Speicher laden

Dieser Weg setzt Zugriff auf einen Speicher voraus, der den gewünschten Export bereits enthält. Standardmäßig sucht DVC im Ordner `intelligence-engineering-dvc-storage` neben dem Repository. Wenn dieser Ordner bei dir vorhanden und befüllt ist, kannst du direkt `dvc pull` ausführen.

Liegt der Export auf einem eingebundenen Teamlaufwerk, trage dessen Pfad einmal für deinen Klon ein. Ersetze den Beispielpfad durch den tatsächlichen Speicherort.

```bash
dvc remote modify --local storage url /absoluter/pfad/zum/team-speicher
```

Lade nun den Export für deinen Git-Stand und prüfe das Ergebnis.

```bash
dvc pull
dvc status
```

Sobald DVC `Data and pipelines are up to date.` meldet, kannst du mit demselben Export wie der Rest des Teams arbeiten.

## Wo die Daten liegen

Der fertige Export liegt unter `data/coco-2017-yolo-50/`.

| Pfad im Export | Inhalt |
| --- | --- |
| `images/train/` | 50 JPEGs. Der Dateiname entspricht der COCO-Bild-ID. |
| `labels/train/` | Zu jedem Bild gehört eine gleichnamige Textdatei. Jede Zeile beschreibt eine Bounding Box mit Klassen-ID, Mittelpunkt (x, y), Breite und Höhe. Die Koordinaten sind auf die Bildgröße normiert. |
| `dataset.yaml` | Relativer Pfad zu den Bildern und Zuordnung der Klassen-IDs (0 = car, 1 = truck, 2 = bicycle, 3 = person). |

Der Export enthält ausschließlich Trainingsbilder. Für ein Training mit separater Validierung musst du zusätzlich Validierungsbilder festlegen und den zugehörigen `val`-Pfad in `dataset.yaml` eintragen.

## Aufgabe 2 · Objekte auf einem Straßenbild erkennen

Führe im Projektverzeichnis mit aktivierter Umgebung den Detektor auf einem Straßenbild aus dem COCO-Export aus.

```bash
python -m src.detect_road_objects
```

Beim ersten Lauf lädt Ultralytics das auf COCO vortrainierte Modell nach `models/yolov8n.pt`. Danach kannst du den Detektor offline auf der CPU ausführen. Unter `runs/task2/` findest du das Bild `000000061181-annotated.jpg` mit eingezeichneten Boxen und die Datei `000000061181.json`. Sie enthält die Anzahl der Vorhersagen je Klasse sowie Klasse, Konfidenz und Boxkoordinaten in Pixeln für jede Erkennung. Der Detektor berücksichtigt Autos, Lkw, Fahrräder und Personen ab einer Konfidenz von 0,25. Die numerischen Klassen-IDs im Ergebnis stammen aus dem vortrainierten Modell. Für den Vergleich mit den Exportlabels verwende die Klassennamen.

Für ein eigenes Foto verwende `python -m src.detect_road_objects --image /pfad/zum/bild.jpg`. Mit `python -m src.detect_road_objects --all` erhältst du die Vorhersagen für alle Bilder des Teilsets und die Gesamtzahl je Klasse in `runs/task2/subset.json`.

Im [Notebook zur ersten Detektion](task2_detection.ipynb) kannst du das Originalfoto, eine einzelne Erkennung und die Modellboxen im Vergleich zu den COCO-Annotationen ansehen. Um die Konfidenzschwelle selbst zu ändern, öffne es mit JupyterLab in der aktivierten Projektumgebung.

```bash
python -m pip install jupyterlab
python -m jupyterlab task2_detection.ipynb
```

## Einen neuen Datenstand mit dem Team teilen

Wenn ihr den gemeinsamen DVC-Speicher zum ersten Mal befüllt, genügt ein `dvc push` aus einem Klon mit geprüftem Datenstand und eingerichtetem Teamspeicher.

Wenn du später die Bildauswahl oder Aufbereitung änderst, passe die Bildliste oder das Exportskript an. Erzeuge den Export erneut und speichere seine neue Prüfsumme mit `dvc add`. Lade die Daten in den gemeinsamen DVC-Speicher hoch, bevor du den zugehörigen Git-Commit veröffentlichst. So können die anderen den neuen Stand direkt abrufen.

```bash
python -m src.download_coco_subset
dvc add data/coco-2017-yolo-50
dvc push
git add data/coco-2017-train-50.txt src/download_coco_subset.py data/coco-2017-yolo-50.dvc
git commit -m "Update COCO subset"
git push
```

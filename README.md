# Curatare si Pregatire Dataset pentru Antrenare YOLO

Acest proiect contine scripturi pentru curatarea, verificarea si standardizarea unui dataset de detectie pentru reciclare, folosit la antrenarea modelelor YOLO.

Scopul repo-ului:
- unificarea mai multor seturi de date intr-un singur dataset coerent;
- eliminarea duplicatelor exacte (la nivel de pixeli);
- conversia adnotarilor de segmentare in YOLO Bounding Box;
- validarea structurii si calitatii label-urilor;
- uniformizarea formatului imaginilor (RGB + extensie majoritara).

## Dataset-ul curent

Dataset final: `date/ecology_dataset`

Clase:
- `glass`
- `metal`
- `paper`
- `plastic`

`data.yaml`:
- `nc: 4`
- `names: ['glass', 'metal', 'paper', 'plastic']`

## Statistici dataset

Numar imagini (si label-uri pereche):
- `train`: 85,182 imagini (`85.00%`)
- `valid`: 10,021 imagini (`10.00%`)
- `test`: 5,012 imagini (`5.00%`)
- `total`: 100,215 imagini

Label-uri YOLO (dupa conversie finala):
- `bbox_only_files`: 98,853
- `segmentation_only_files`: 0
- `mixed_bbox_segmentation_files`: 0
- `invalid_files`: 0
- `empty_files`: 1,362
- `total_lines`: 207,217
- `bbox_lines`: 207,217
- `segmentation_lines`: 0
- `invalid_lines`: 0

Consistenta perechi imagine-label:
- `matched_image_label_pairs`: 100,215
- `images_without_label`: 0
- `labels_without_image`: 0

Format imagini:
- extensie: `100% .jpg` (100,215 / 100,215)
- mod culoare: `100% RGB` (conform verificarii cu scriptul de RGB)

## Ce face fiecare script

- `utils/merge_yolo_datasets.py`
  - combina seturi de date YOLO in unul singur;
  - mentine split-urile `train/valid/test`;
  - copiaza `images` + `labels` si remapeaza corect clasa pe baza `data.yaml`.

- `utils/remove_exact_duplicate_images.py`
  - detecteaza duplicate exacte pe pixeli;
  - sterge in-place imaginea duplicat + label-ul pereche;
  - are mod `--dry-run`.

- `utils/conversie_segmentare_la_detectie.py`
  - converteste adnotari YOLO segmentare in YOLO bounding box;
  - poate suprascrie fisierele originale cu `--suprascrie`.

- `utils/verificare/verifica_labeluri_yolo_bbox.py`
  - auditeaza label-urile si raporteaza tipurile de fisiere/linii;
  - confirma daca datasetul este complet bbox;
  - verifica si pairing-ul imagine-label pe fiecare split.

- `utils/verificare/verifica_si_converteste_rgb.py`
  - verifica modurile de culoare;
  - converteste imaginile non-RGB in RGB.

- `utils/verificare/verifica_si_converteste_format_majoritar.py`
  - detecteaza formatul imagine majoritar;
  - converteste restul imaginilor la formatul majoritar.

- `utils/reimparte_random_split_yolo.py`
  - reimparte random datasetul in `train/valid/test` dupa procentele dorite;
  - muta corect perechile `image + label`;
  - suporta `--seed` (reproductibil) si `--dry-run`.

## De ce e util acest repo

Repo-ul reduce riscul de probleme la antrenare YOLO:
- elimina duplicate care pot distorsiona metricele;
- elimina inconsistente de label-uri (segmentare vs bbox);
- asigura consistenta de input (format si culoare);
- ofera scripturi reproductibile pentru pregatirea datasetului.

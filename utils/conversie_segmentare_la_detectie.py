"""
Script pentru conversia adnotarilor din format YOLO Segmentare in format YOLO Detectie.

Format YOLO Segmentare (intrare):
    <clasa> <x1> <y1> <x2> <y2> ... <xN> <yN>
    - coordonatele definesc un poligon (masca de segmentare)

Format YOLO Detectie (iesire):
    <clasa> <x_centru> <y_centru> <latime> <inaltime>
    - coordonatele definesc un bounding box normalizat

Utilizare:
    python conversie_segmentare_la_detectie.py --sursa <cale_folder_sursa> --destinatie <cale_folder_destinatie>
    python conversie_segmentare_la_detectie.py --sursa <cale_folder_sursa> --suprascrie

Exemple:
    # Converteste si salveaza intr-un folder nou
    python conversie_segmentare_la_detectie.py \\
        --sursa seturi_date/Plastic.v1i.yolo26/train/labels \\
        --destinatie seturi_date/Plastic.v1i.yolo26/train/labels_detectie

    # Converteste si suprascrie fisierele originale
    python conversie_segmentare_la_detectie.py \\
        --sursa seturi_date/Plastic.v1i.yolo26/train/labels \\
        --suprascrie
"""

import argparse
import os
import sys
from pathlib import Path


def parseaza_linie_segmentare(linie: str) -> tuple[int, list[float]]:
    """
    Parseaza o linie din formatul YOLO Segmentare.

    Parametri:
        linie: linia de text din fisierul de adnotare

    Returneaza:
        (id_clasa, lista_coordonate) - clasa si lista de coordonate [x1, y1, x2, y2, ...]
    """
    valori = linie.strip().split()
    id_clasa = int(valori[0])
    coordonate = [float(v) for v in valori[1:]]
    return id_clasa, coordonate


def calculeaza_bounding_box(coordonate: list[float]) -> tuple[float, float, float, float]:
    """
    Calculeaza bounding box-ul (dreptunghiul incadrator) dintr-o lista de coordonate de poligon.

    Parametri:
        coordonate: lista de coordonate [x1, y1, x2, y2, ...] normalizate (0-1)

    Returneaza:
        (x_centru, y_centru, latime, inaltime) - bounding box in format YOLO
    """
    # Separam coordonatele x si y
    coordonate_x = coordonate[0::2]  # elementele de pe pozitii pare
    coordonate_y = coordonate[1::2]  # elementele de pe pozitii impare

    # Gasim limitele poligonului
    x_min = min(coordonate_x)
    x_max = max(coordonate_x)
    y_min = min(coordonate_y)
    y_max = max(coordonate_y)

    # Calculam formatul YOLO: centru + dimensiuni
    x_centru = (x_min + x_max) / 2
    y_centru = (y_min + y_max) / 2
    latime = x_max - x_min
    inaltime = y_max - y_min

    return x_centru, y_centru, latime, inaltime


def converteste_fisier(cale_intrare: Path, cale_iesire: Path) -> int:
    """
    Converteste un singur fisier de adnotari din segmentare in detectie.

    Parametri:
        cale_intrare: calea catre fisierul sursa (.txt)
        cale_iesire:  calea catre fisierul destinatie (.txt)

    Returneaza:
        numarul de adnotari convertite
    """
    linii_convertite = []

    with open(cale_intrare, "r") as f:
        for numar_linie, linie in enumerate(f, start=1):
            linie = linie.strip()
            if not linie:
                continue

            valori = linie.split()

            # Verificam daca linia are suficiente valori (clasa + minim 3 puncte = 7 valori)
            if len(valori) < 7:
                # Daca are exact 5 valori, e deja in format detectie - o pastram
                if len(valori) == 5:
                    linii_convertite.append(linie)
                    continue
                print(
                    f"  Atentie: {cale_intrare.name}, linia {numar_linie} "
                    f"- numar invalid de valori ({len(valori)}), linia a fost ignorata"
                )
                continue

            id_clasa, coordonate = parseaza_linie_segmentare(linie)

            # Verificam ca avem perechi complete de coordonate
            if len(coordonate) % 2 != 0:
                print(
                    f"  Atentie: {cale_intrare.name}, linia {numar_linie} "
                    f"- numar impar de coordonate, linia a fost ignorata"
                )
                continue

            x_centru, y_centru, latime, inaltime = calculeaza_bounding_box(coordonate)

            # Formatam linia in format YOLO Detectie (6 zecimale)
            linie_noua = f"{id_clasa} {x_centru:.6f} {y_centru:.6f} {latime:.6f} {inaltime:.6f}"
            linii_convertite.append(linie_noua)

    # Scriem fisierul convertit
    with open(cale_iesire, "w") as f:
        f.write("\n".join(linii_convertite))
        if linii_convertite:
            f.write("\n")

    return len(linii_convertite)


def converteste_folder(cale_sursa: Path, cale_destinatie: Path) -> None:
    """
    Converteste toate fisierele .txt dintr-un folder sursa si le salveaza in folderul destinatie.

    Parametri:
        cale_sursa:      calea catre folderul cu adnotari de segmentare
        cale_destinatie: calea catre folderul unde se salveaza adnotarile de detectie
    """
    # Cream folderul destinatie daca nu exista
    cale_destinatie.mkdir(parents=True, exist_ok=True)

    # Gasim toate fisierele .txt din folderul sursa
    fisiere = sorted(cale_sursa.glob("*.txt"))

    if not fisiere:
        print(f"Nu s-au gasit fisiere .txt in: {cale_sursa}")
        sys.exit(1)

    print(f"S-au gasit {len(fisiere)} fisiere de adnotari in: {cale_sursa}")
    print(f"Rezultatele vor fi salvate in: {cale_destinatie}")
    print("-" * 60)

    total_adnotari = 0

    for fisier in fisiere:
        cale_iesire = cale_destinatie / fisier.name
        numar_adnotari = converteste_fisier(fisier, cale_iesire)
        total_adnotari += numar_adnotari

    print("-" * 60)
    print(f"Conversie finalizata!")
    print(f"  Fisiere procesate:   {len(fisiere)}")
    print(f"  Adnotari convertite: {total_adnotari}")
    print(f"  Salvate in:          {cale_destinatie}")


def main():
    parser = argparse.ArgumentParser(
        description="Converteste adnotarile YOLO din format segmentare in format detectie (bounding box).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemple de utilizare:
  %(prog)s --sursa train/labels --destinatie train/labels_detectie
  %(prog)s --sursa train/labels --suprascrie
        """,
    )

    parser.add_argument(
        "--sursa",
        type=str,
        required=True,
        help="Calea catre folderul cu adnotari de segmentare (fisiere .txt)",
    )

    parser.add_argument(
        "--destinatie",
        type=str,
        default=None,
        help="Calea catre folderul unde se salveaza rezultatele (implicit: <sursa>_detectie)",
    )

    parser.add_argument(
        "--suprascrie",
        action="store_true",
        help="Suprascrie fisierele originale in loc sa creeze un folder nou",
    )

    args = parser.parse_args()

    cale_sursa = Path(args.sursa)

    # Verificam ca folderul sursa exista
    if not cale_sursa.is_dir():
        print(f"Eroare: Folderul sursa nu exista: {cale_sursa}")
        sys.exit(1)

    # Determinam folderul destinatie
    if args.suprascrie:
        cale_destinatie = cale_sursa
        print("Atentie: Fisierele originale vor fi suprascrise!")
    elif args.destinatie:
        cale_destinatie = Path(args.destinatie)
    else:
        # Implicit: adaugam sufixul '_detectie' la numele folderului sursa
        cale_destinatie = cale_sursa.parent / (cale_sursa.name + "_detectie")

    converteste_folder(cale_sursa, cale_destinatie)


if __name__ == "__main__":
    main()

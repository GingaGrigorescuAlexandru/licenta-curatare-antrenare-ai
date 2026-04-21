"""
Script pentru incarcarea unui set de date in Roboflow.

Utilizare:
    python upload_roboflow.py --api-key CHEIA_TA --workspace WORKSPACE --proiect ID_PROIECT --dataset cale/catre/dataset

Exemple:
    python upload_roboflow.py \
        --api-key "abc123" \
        --workspace "finetunezedmodels" \
        --proiect "plastic-detectie" \
        --dataset "seturi_date/Plastic.v1i.yolo26"

    # Cu optiuni suplimentare
    python upload_roboflow.py \
        --api-key "abc123" \
        --workspace "finetunezedmodels" \
        --proiect "plastic-detectie" \
        --dataset "seturi_date/Plastic.v1i.yolo26" \
        --tip-proiect "object-detection" \
        --licenta "MIT" \
        --workeri 10
"""

import argparse
import sys

try:
    import roboflow
except ImportError:
    print("Eroare: Pachetul 'roboflow' nu este instalat.")
    print("Instaleaza-l cu: pip install roboflow")
    sys.exit(1)

from pathlib import Path


def upload(args):
    """
    Incarca setul de date in Roboflow folosind parametrii primiti.
    """
    # Verificam ca folderul dataset exista
    cale_dataset = Path(args.dataset)
    if not cale_dataset.is_dir():
        print(f"Eroare: Folderul dataset nu exista: {cale_dataset}")
        sys.exit(1)

    print(f"Conectare la Roboflow...")
    rf = roboflow.Roboflow(api_key=args.api_key)

    print(f"Accesare workspace: {args.workspace}")
    workspace = rf.workspace(args.workspace)

    print(f"Incarcare dataset din: {cale_dataset}")
    print(f"  Proiect:     {args.proiect}")
    print(f"  Tip proiect: {args.tip_proiect}")
    print(f"  Licenta:     {args.licenta}")
    print(f"  Workeri:     {args.workeri}")
    print("-" * 60)

    workspace.upload_dataset(
        str(cale_dataset),
        args.proiect,
        num_workers=args.workeri,
        project_license=args.licenta,
        project_type=args.tip_proiect,
        batch_name=args.batch,
        num_retries=args.retries,
    )

    print("-" * 60)
    print("Incarcare finalizata!")


def main():
    parser = argparse.ArgumentParser(
        description="Incarca un set de date in Roboflow.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemple de utilizare:
  %(prog)s --api-key "abc123" --workspace "finetunezedmodels" --proiect "plastic-v1" --dataset "seturi_date/Plastic.v1i.yolo26"
        """,
    )

    # Argumente obligatorii
    parser.add_argument(
        "--api-key",
        type=str,
        required=True,
        help="Cheia API Roboflow (gasesti in Settings > API Key)",
    )

    parser.add_argument(
        "--workspace",
        type=str,
        required=True,
        help="ID-ul workspace-ului Roboflow (ex: 'finetunezedmodels')",
    )

    parser.add_argument(
        "--proiect",
        type=str,
        required=True,
        help="ID-ul proiectului Roboflow (se creeaza automat daca nu exista)",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Calea catre folderul cu setul de date",
    )

    # Argumente optionale
    parser.add_argument(
        "--tip-proiect",
        type=str,
        default="object-detection",
        help="Tipul proiectului (implicit: 'object-detection')",
    )

    parser.add_argument(
        "--licenta",
        type=str,
        default="MIT",
        help="Licenta proiectului (implicit: 'MIT')",
    )

    parser.add_argument(
        "--workeri",
        type=int,
        default=10,
        help="Numarul de workeri pentru incarcare paralela (implicit: 10)",
    )

    parser.add_argument(
        "--batch",
        type=str,
        default=None,
        help="Numele batch-ului (optional)",
    )

    parser.add_argument(
        "--retries",
        type=int,
        default=0,
        help="Numarul de reincercari in caz de eroare (implicit: 0)",
    )

    args = parser.parse_args()
    upload(args)


if __name__ == "__main__":
    main()

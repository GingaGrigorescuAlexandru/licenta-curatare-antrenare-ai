"""
Script simplu pentru descarcarea dataset-ului din Roboflow.

Instalare dependinta:
    pip install roboflow
"""

from pathlib import Path
import os

from roboflow import Roboflow


def main() -> None:
    workspace = "finetunezedmodels"
    project_slug = "plastic-merge-6"
    version_no = 2
    model_format = "yolo26"

    api_key = os.getenv("ROBOFLOW_API_KEY", "")
    if api_key == "":
        raise ValueError(
            "Seteaza cheia in script sau in variabila de mediu ROBOFLOW_API_KEY."
        )

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_slug)
    version = project.version(version_no)

    target_dir = Path("date") / f"{project_slug}-v{version_no}"
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    dataset = version.download(model_format, location=str(target_dir))
    print(f"Dataset descarcat in: {dataset.location}")

    data_yaml = Path(dataset.location) / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Lipseste data.yaml in: {data_yaml}")

    downloaded_project = None
    with data_yaml.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("project:"):
                downloaded_project = stripped.split(":", 1)[1].strip()
                break

    if downloaded_project != project_slug:
        raise RuntimeError(
            f"Proiect descarcat diferit: '{downloaded_project}' (asteptat '{project_slug}')."
        )

    print(f"Proiect confirmat: {downloaded_project}")


if __name__ == "__main__":
    main()

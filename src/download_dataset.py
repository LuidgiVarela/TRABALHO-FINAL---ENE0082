import argparse
import shutil
from pathlib import Path

from src.data import EXPECTED_CLASSES

DATASET_SLUG = "paultimothymooney/chest-xray-pneumonia"
SPLITS = ["train", "val", "test"]


def flatten_nested_folder(target_dir: Path):
    """O zip do Kaggle extrai como chest_xray/chest_xray/... em algumas versoes; remove o aninhamento."""
    nested_dir = target_dir / "chest_xray"
    if nested_dir.exists() and nested_dir.is_dir():
        for item in nested_dir.iterdir():
            shutil.move(str(item), str(target_dir / item.name))
        nested_dir.rmdir()


def remove_known_artifacts(target_dir: Path):
    macosx_dir = target_dir / "__MACOSX"
    if macosx_dir.exists():
        shutil.rmtree(macosx_dir)

    for ds_store in target_dir.rglob(".DS_Store"):
        ds_store.unlink()


def validate_structure(target_dir: Path):
    missing = []
    for split in SPLITS:
        for class_name in EXPECTED_CLASSES:
            class_dir = target_dir / split / class_name
            if not class_dir.exists():
                missing.append(str(class_dir))

    if missing:
        raise FileNotFoundError(
            "Estrutura incompleta apos o download. Pastas ausentes:\n" + "\n".join(missing)
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Baixa o dataset Chest X-Ray Images (Pneumonia) do Kaggle e organiza em "
            "data/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}."
        )
    )
    parser.add_argument("--output-dir", default="data/chest_xray")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as error:
        raise SystemExit(
            "Pacote 'kaggle' nao encontrado. Instale com 'pip install kaggle' "
            "e configure suas credenciais em ~/.kaggle/kaggle.json antes de rodar este script."
        ) from error

    api = KaggleApi()
    api.authenticate()

    print(f"Baixando dataset '{DATASET_SLUG}' para {output_dir} ...")
    api.dataset_download_files(DATASET_SLUG, path=str(output_dir), unzip=True, quiet=False)

    flatten_nested_folder(output_dir)
    remove_known_artifacts(output_dir)
    validate_structure(output_dir)

    print(f"Dataset pronto em: {output_dir}")


if __name__ == "__main__":
    main()

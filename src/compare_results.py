import argparse
import csv
import json
from pathlib import Path


METRIC_COLUMNS = ["model", "accuracy", "precision", "recall", "f1_score"]


def load_metrics(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as file:
        metrics = json.load(file)

    missing = [column for column in METRIC_COLUMNS if column not in metrics]
    if missing:
        raise ValueError(f"Arquivo {path} nao contem as chaves: {', '.join(missing)}")

    return {column: metrics[column] for column in METRIC_COLUMNS}


def main():
    parser = argparse.ArgumentParser(description="Compara CNN simples e Transfer Learning.")
    parser.add_argument("--simple-cnn", required=True, help="JSON com metricas da CNN simples.")
    parser.add_argument("--transfer", default="results/transfer_metrics.json")
    parser.add_argument(
        "--extra-metrics",
        nargs="*",
        default=[],
        help="Arquivos JSON adicionais de metricas para incluir na tabela.",
    )
    parser.add_argument("--output", default="results/comparison_table.csv")
    parser.add_argument("--markdown-output", default="results/comparison_table.md")
    args = parser.parse_args()

    metric_paths = [args.simple_cnn, args.transfer, *args.extra_metrics]
    rows = [load_metrics(path) for path in metric_paths]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    formatted_rows = []
    for row in rows:
        formatted_rows.append(
            {
                "model": row["model"],
                "accuracy": f"{row['accuracy']:.4f}",
                "precision": f"{row['precision']:.4f}",
                "recall": f"{row['recall']:.4f}",
                "f1_score": f"{row['f1_score']:.4f}",
            }
        )

    header = "| Modelo | Acurácia | Precisão | Recall | F1-score |"
    separator = "|---|---:|---:|---:|---:|"
    markdown_lines = [header, separator]
    for row in formatted_rows:
        markdown_lines.append(
            "| "
            + " | ".join(
                [
                    row["model"],
                    row["accuracy"],
                    row["precision"],
                    row["recall"],
                    row["f1_score"],
                ]
            )
            + " |"
        )

    markdown_path = Path(args.markdown_output)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")

    print("\n".join(markdown_lines))
    print(f"Tabela salva em: {output_path}")
    print(f"Tabela em Markdown salva em: {markdown_path}")


if __name__ == "__main__":
    main()

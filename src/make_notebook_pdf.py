from pathlib import Path
import textwrap

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_PATH = PROJECT_ROOT / "docs" / "pessoa2_transfer_learning_resultados.pdf"

BLUE = "#1f4e79"
LIGHT_BLUE = "#d9eaf7"
LIGHT_GRAY = "#f3f5f7"
DARK = "#1f2933"
MUTED = "#52606d"

METRICS_ROWS = [
    ["CNN simples", "0.8766", "0.9003", "0.9026", "0.9014"],
    ["ResNet18\nTransfer Learning", "0.9054", "0.9046", "0.9487", "0.9262"],
    ["DenseNet121\nTransfer Learning", "0.8734", "0.8676", "0.9410", "0.9028"],
]
METRICS_COLUMNS = ["Modelo", "Acurácia", "Precisão", "Recall", "F1-score"]


def add_page_number(fig, page: int) -> None:
    fig.text(0.5, 0.025, f"Pessoa 2 - Transfer Learning | Página {page}", ha="center", fontsize=8, color=MUTED)


def add_header(fig, title: str, subtitle: str | None = None) -> None:
    fig.text(0.06, 0.93, title, ha="left", va="top", fontsize=20, weight="bold", color=DARK)
    if subtitle:
        fig.text(0.06, 0.89, subtitle, ha="left", va="top", fontsize=10, color=MUTED)
    fig.add_artist(Rectangle((0.06, 0.865), 0.88, 0.004, color=BLUE, transform=fig.transFigure, clip_on=False))


def wrap_text(text: str, width: int = 95) -> str:
    paragraphs = []
    for paragraph in text.split("\n"):
        if paragraph:
            paragraphs.append("\n".join(textwrap.wrap(paragraph, width=width)))
        else:
            paragraphs.append("")
    return "\n".join(paragraphs)


def add_box(fig, x: float, y: float, w: float, h: float, title: str, body: str) -> None:
    fig.add_artist(Rectangle((x, y), w, h, facecolor=LIGHT_GRAY, edgecolor="#c9d4df", linewidth=0.8))
    fig.text(x + 0.02, y + h - 0.035, title, fontsize=11, weight="bold", color=BLUE, va="top")
    fig.text(x + 0.02, y + h - 0.075, wrap_text(body, 64), fontsize=9.5, color=DARK, va="top", linespacing=1.2)


def add_metrics_table(fig, bbox: list[float], font_size: int = 9) -> None:
    ax = fig.add_axes(bbox)
    ax.axis("off")
    table = ax.table(
        cellText=METRICS_ROWS,
        colLabels=METRICS_COLUMNS,
        colWidths=[0.35, 0.16, 0.16, 0.16, 0.17],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, 1.55)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#2f3a45")
        cell.set_linewidth(0.7)
        if row == 0:
            cell.set_facecolor(LIGHT_BLUE)
            cell.set_text_props(weight="bold", color=DARK, ha="center", va="center")
        elif col == 0:
            cell.set_text_props(ha="left", va="center")
        else:
            cell.set_text_props(ha="center", va="center")


def add_image(fig, image_path: Path, bbox: list[float]) -> None:
    ax = fig.add_axes(bbox)
    ax.imshow(mpimg.imread(image_path))
    ax.axis("off")


def save_summary_page(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    add_header(
        fig,
        "Pessoa 2 - Transfer Learning e comparação final",
        "Anexo técnico de resultados para o trabalho final de NIA / ENE0082",
    )

    fig.text(
        0.06,
        0.825,
        wrap_text(
            "Este documento resume a etapa de Transfer Learning do projeto de classificação de "
            "radiografias torácicas em NORMAL e PNEUMONIA. Ele reúne a tabela comparativa final, "
            "as matrizes de confusão, os históricos de treinamento e a conclusão da Pessoa 2.",
            108,
        ),
        ha="left",
        va="top",
        fontsize=10.5,
        color=DARK,
        linespacing=1.25,
    )

    add_box(
        fig,
        0.06,
        0.61,
        0.42,
        0.14,
        "Modelos comparados",
        "CNN simples treinada do zero; ResNet18 com Transfer Learning; DenseNet121 com Transfer Learning como experimento adicional.",
    )
    add_box(
        fig,
        0.52,
        0.61,
        0.42,
        0.14,
        "Configuração dos modelos",
        "ResNet18 e DenseNet121 usaram pesos pré-treinados no ImageNet. O backbone foi congelado e apenas a camada final foi treinada.",
    )

    fig.text(0.06, 0.545, "Tabela comparativa final", fontsize=14, weight="bold", color=DARK, va="top")
    add_metrics_table(fig, [0.06, 0.34, 0.88, 0.17], font_size=9)

    add_box(
        fig,
        0.06,
        0.14,
        0.88,
        0.13,
        "Síntese",
        "A ResNet18 apresentou o melhor desempenho geral no conjunto de teste, com maior acurácia, precisão, recall e F1-score. "
        "Por isso, ela foi escolhida como modelo final da etapa de Transfer Learning.",
    )

    add_page_number(fig, 1)
    pdf.savefig(fig)
    plt.close(fig)


def save_confusion_page(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")
    add_header(fig, "Matrizes de confusão", "Linhas representam a classe real; colunas representam a classe predita.")

    report_figures = RESULTS_DIR / "report_figures"
    add_image(fig, report_figures / "cnn_simples_confusion_matrix.png", [0.035, 0.22, 0.30, 0.55])
    add_image(fig, report_figures / "resnet18_confusion_matrix.png", [0.35, 0.22, 0.30, 0.55])
    add_image(fig, report_figures / "densenet121_confusion_matrix.png", [0.665, 0.22, 0.30, 0.55])

    fig.text(
        0.06,
        0.13,
        wrap_text(
            "A ResNet18 reduziu os falsos negativos de pneumonia em relação à CNN simples: "
            "20 casos contra 38. Esse ponto é relevante porque falsos negativos são especialmente indesejáveis "
            "em problemas de triagem médica.",
            145,
        ),
        fontsize=10,
        color=DARK,
        va="top",
        linespacing=1.2,
    )

    add_page_number(fig, 2)
    pdf.savefig(fig)
    plt.close(fig)


def save_history_page(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")
    add_header(fig, "Histórico de treinamento", "Curvas de perda, acurácia e F1-score de validação dos modelos de Transfer Learning.")

    report_figures = RESULTS_DIR / "report_figures"
    add_image(fig, report_figures / "resnet18_loss_history.png", [0.055, 0.50, 0.42, 0.32])
    add_image(fig, report_figures / "resnet18_metrics_history.png", [0.525, 0.50, 0.42, 0.32])
    add_image(fig, report_figures / "densenet121_loss_history.png", [0.055, 0.13, 0.42, 0.32])
    add_image(fig, report_figures / "densenet121_metrics_history.png", [0.525, 0.13, 0.42, 0.32])

    fig.text(0.06, 0.455, "ResNet18", fontsize=11, weight="bold", color=BLUE)
    fig.text(0.06, 0.085, "DenseNet121", fontsize=11, weight="bold", color=BLUE)

    add_page_number(fig, 3)
    pdf.savefig(fig)
    plt.close(fig)


def save_analysis_page(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    add_header(fig, "Análise e conclusão", "Resumo interpretativo para apoiar relatório e apresentação.")

    paragraphs = [
        (
            "Desempenho geral",
            "A ResNet18 com Transfer Learning apresentou o melhor desempenho geral entre os modelos avaliados. "
            "Ela superou a CNN simples e a DenseNet121 em acurácia, precisão, recall e F1-score no conjunto de teste.",
        ),
        (
            "Experimento adicional",
            "A DenseNet121 foi mantida como experimento adicional. Embora seja uma arquitetura forte e comum em aplicações "
            "de imagens médicas, neste experimento ela não superou a ResNet18. Isso reforça a escolha da ResNet18 como "
            "modelo final da Pessoa 2.",
        ),
        (
            "Importância do recall",
            "O recall é especialmente relevante neste problema, pois falsos negativos de pneumonia são indesejáveis. "
            "A ResNet18 obteve o maior recall entre os modelos comparados.",
        ),
        (
            "Limitação metodológica",
            "O conjunto de validação original possui apenas 16 imagens, o que torna suas métricas instáveis. Por isso, "
            "a discussão final deve priorizar as métricas no conjunto de teste, que possui 624 imagens.",
        ),
    ]

    y = 0.80
    for title, body in paragraphs:
        add_box(fig, 0.06, y - 0.13, 0.88, 0.115, title, body)
        y -= 0.16

    add_box(
        fig,
        0.06,
        0.12,
        0.88,
        0.12,
        "Conclusão final",
        "A ResNet18 é o modelo recomendado para apresentação como resultado final da Pessoa 2, mantendo a DenseNet121 "
        "como comparação adicional e a CNN simples como baseline do trabalho.",
    )

    add_page_number(fig, 4)
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    report_figures = RESULTS_DIR / "report_figures"
    required_images = [
        report_figures / "cnn_simples_confusion_matrix.png",
        report_figures / "resnet18_confusion_matrix.png",
        report_figures / "densenet121_confusion_matrix.png",
        report_figures / "resnet18_loss_history.png",
        report_figures / "resnet18_metrics_history.png",
        report_figures / "densenet121_loss_history.png",
        report_figures / "densenet121_metrics_history.png",
    ]
    missing = [path for path in required_images if not path.exists()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Figuras ausentes:\n{missing_text}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUTPUT_PATH) as pdf:
        save_summary_page(pdf)
        save_confusion_page(pdf)
        save_history_page(pdf)
        save_analysis_page(pdf)

    print(f"PDF salvo em: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

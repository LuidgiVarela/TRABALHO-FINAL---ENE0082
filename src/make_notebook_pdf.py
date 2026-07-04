from pathlib import Path
import textwrap

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_PATH = PROJECT_ROOT / "docs" / "pessoa2_transfer_learning_resultados.pdf"


METRICS_TABLE = [
    ["Modelo", "Acurácia", "Precisão", "Recall", "F1-score"],
    ["CNN simples", "0.8766", "0.9003", "0.9026", "0.9014"],
    ["ResNet18 Transfer Learning", "0.9054", "0.9046", "0.9487", "0.9262"],
    ["DenseNet121 Transfer Learning", "0.8734", "0.8676", "0.9410", "0.9028"],
]


def add_text(ax, text: str, x: float, y: float, size: int = 10, width: int = 90) -> float:
    lines = []
    for paragraph in text.split("\n"):
        if paragraph:
            lines.extend(textwrap.wrap(paragraph, width=width))
        else:
            lines.append("")

    ax.text(x, y, "\n".join(lines), ha="left", va="top", fontsize=size)
    return y - 0.035 * max(len(lines), 1)


def new_page(pdf: PdfPages, title: str):
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.07, 0.95, title, ha="left", va="top", fontsize=18, weight="bold")
    return fig, ax


def save_intro_page(pdf: PdfPages) -> None:
    fig, ax = new_page(pdf, "Pessoa 2 - Transfer Learning e comparação final")
    y = 0.88
    y = add_text(
        ax,
        "Este anexo técnico resume a etapa de Transfer Learning do trabalho. "
        "Ele reúne a tabela comparativa, matrizes de confusão, gráficos de treinamento "
        "e conclusão da Pessoa 2.",
        0.07,
        y,
        size=11,
    )
    y -= 0.02
    y = add_text(
        ax,
        "Modelos comparados:\n"
        "- CNN simples treinada do zero pela Pessoa 1;\n"
        "- ResNet18 com Transfer Learning;\n"
        "- DenseNet121 com Transfer Learning, como experimento adicional.",
        0.07,
        y,
        size=11,
    )
    y -= 0.02
    add_text(
        ax,
        "A ResNet18 e a DenseNet121 foram usadas com pesos pré-treinados no ImageNet. "
        "Em ambos os casos, o backbone foi congelado e apenas a camada final foi treinada "
        "para classificar as imagens em NORMAL e PNEUMONIA.",
        0.07,
        y,
        size=11,
    )
    pdf.savefig(fig)
    plt.close(fig)


def save_table_page(pdf: PdfPages) -> None:
    fig, ax = new_page(pdf, "Tabela comparativa final")
    table_ax = fig.add_axes([0.06, 0.47, 0.88, 0.32])
    table_ax.axis("off")
    table = table_ax.table(cellText=METRICS_TABLE[1:], colLabels=METRICS_TABLE[0], loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)

    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#d9eaf7")

    add_text(
        ax,
        "A ResNet18 apresentou o melhor desempenho geral no conjunto de teste, "
        "com maior acurácia, precisão, recall e F1-score entre os modelos comparados.",
        0.07,
        0.38,
        size=11,
    )
    pdf.savefig(fig)
    plt.close(fig)


def save_image_page(pdf: PdfPages, title: str, image_path: Path) -> None:
    fig, ax = new_page(pdf, title)
    image = mpimg.imread(image_path)
    image_ax = fig.add_axes([0.08, 0.12, 0.84, 0.75])
    image_ax.imshow(image)
    image_ax.axis("off")
    pdf.savefig(fig)
    plt.close(fig)


def save_analysis_page(pdf: PdfPages) -> None:
    fig, ax = new_page(pdf, "Análise e conclusão")
    y = 0.88
    y = add_text(
        ax,
        "A ResNet18 com Transfer Learning apresentou o melhor desempenho geral entre os "
        "modelos avaliados. Ela superou a CNN simples e a DenseNet121 em acurácia, "
        "precisão, recall e F1-score no conjunto de teste.",
        0.07,
        y,
        size=11,
    )
    y -= 0.02
    y = add_text(
        ax,
        "A DenseNet121 foi mantida como experimento adicional. Apesar de ser uma arquitetura "
        "forte e comum em aplicações de imagens médicas, neste experimento ela não superou "
        "a ResNet18. Isso reforça a escolha da ResNet18 como modelo final da Pessoa 2.",
        0.07,
        y,
        size=11,
    )
    y -= 0.02
    y = add_text(
        ax,
        "O recall é uma métrica especialmente relevante neste problema, pois falsos negativos "
        "de pneumonia são indesejáveis. A ResNet18 obteve o maior recall entre os modelos "
        "comparados.",
        0.07,
        y,
        size=11,
    )
    y -= 0.02
    add_text(
        ax,
        "A principal limitação metodológica observada foi o tamanho reduzido do conjunto de "
        "validação original, que possui apenas 16 imagens. Por isso, a discussão dos resultados "
        "deve priorizar as métricas no conjunto de teste, que possui 624 imagens.",
        0.07,
        y,
        size=11,
    )
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
        save_intro_page(pdf)
        save_table_page(pdf)
        save_image_page(pdf, "Matriz de confusão - CNN simples", required_images[0])
        save_image_page(pdf, "Matriz de confusão - ResNet18", required_images[1])
        save_image_page(pdf, "Matriz de confusão - DenseNet121", required_images[2])
        save_image_page(pdf, "Histórico de perda - ResNet18", required_images[3])
        save_image_page(pdf, "Histórico de métricas - ResNet18", required_images[4])
        save_image_page(pdf, "Histórico de perda - DenseNet121", required_images[5])
        save_image_page(pdf, "Histórico de métricas - DenseNet121", required_images[6])
        save_analysis_page(pdf)

    print(f"PDF salvo em: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

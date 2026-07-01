# Trabalho Final - ENE0082 / NIA

Classificacao de radiografias toracicas para deteccao de pneumonia usando CNNs e Transfer Learning.

## Tema

O projeto classifica imagens de raio-X do torax em duas classes:

- `NORMAL`
- `PNEUMONIA`

Dataset: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

## Divisao de tarefas

### Pessoa 1 - Dataset, pre-processamento e CNN simples

Responsavel por baixar e organizar o dataset, criar o carregamento das imagens no PyTorch, aplicar transformacoes basicas e treinar uma CNN simples feita do zero.

### Pessoa 2 - Transfer Learning e comparacao final

Responsavel por treinar um modelo com Transfer Learning, ajustar a ultima camada para duas classes, avaliar o modelo e comparar seus resultados com a CNN simples.

Arquivos principais desta parte:

- `src/train_transfer.py`: treino da ResNet18 com Transfer Learning.
- `src/data.py`: carregamento e pre-processamento reutilizavel.
- `src/evaluate.py`: metricas e matriz de confusao.
- `src/compare_results.py`: tabela comparativa entre CNN simples e Transfer Learning.

### Pessoa 3 - Relatorio final e apresentacao

Responsavel por organizar a entrega escrita, inserir graficos/tabelas e escrever a analise dos resultados.

## Estrutura esperada do dataset

Apos baixar e extrair o dataset, organize os arquivos assim:

```text
data/chest_xray/
  train/
    NORMAL/
    PNEUMONIA/
  val/
    NORMAL/
    PNEUMONIA/
  test/
    NORMAL/
    PNEUMONIA/
```

## Preparacao do ambiente

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Treinar Transfer Learning

```bash
python -m src.train_transfer --data-dir data/chest_xray --epochs 10 --batch-size 32
```

O treinamento salva metricas, matriz de confusao e checkpoint em `results/`.

Arquivos gerados pela etapa de Transfer Learning:

- `results/transfer_metrics.json`: metricas finais no teste.
- `results/transfer_confusion_matrix.png`: matriz de confusao.
- `results/transfer_training_history.csv`: historico por epoca.
- `results/transfer_loss_history.png`: grafico da perda de treino e validacao.
- `results/transfer_metrics_history.png`: grafico de acuracia e F1-score.
- `results/resnet18_transfer_best.pth`: melhor checkpoint pelo F1-score de validacao.

## Comparar com a CNN simples

Quando a Pessoa 1 gerar as metricas da CNN simples, salve um arquivo JSON no formato:

```json
{
  "model": "CNN simples",
  "accuracy": 0.85,
  "precision": 0.86,
  "recall": 0.84,
  "f1_score": 0.85
}
```

Tambem ha um exemplo em `configs/simple_cnn_metrics.example.json`.

Depois execute:

```bash
python -m src.compare_results --simple-cnn results/simple_cnn_metrics.json --transfer results/transfer_metrics.json
```

Isso gera `results/comparison_table.csv`.

Tambem e gerada uma versao em Markdown em `results/comparison_table.md`, util para copiar a tabela para o relatorio.

## Materiais da Pessoa 2

- `docs/pessoa2_metodologia.md`: texto base para a metodologia de Transfer Learning.
- `docs/checklist_pessoa2.md`: checklist operacional da sua parte.

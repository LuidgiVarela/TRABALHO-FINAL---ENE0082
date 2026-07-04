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

Arquivos principais desta parte:

- `src/download_dataset.py`: baixa e organiza o dataset via API do Kaggle.
- `src/explore_dataset.py`: confere a estrutura de pastas, conta imagens por classe e gera exemplos de radiografias.
- `src/data.py`: carregamento e pre-processamento reutilizavel (compartilhado com a Pessoa 2).
- `src/models.py`: define a `SimpleCNN`, treinada do zero.
- `src/train_simple_cnn.py`: treino e avaliacao da CNN simples.

### Pessoa 2 - Transfer Learning e comparacao final

Responsavel por treinar um modelo com Transfer Learning, ajustar a ultima camada para duas classes, avaliar o modelo e comparar seus resultados com a CNN simples.

Arquivos principais desta parte:

- `src/train_transfer.py`: treino de modelos com Transfer Learning, incluindo ResNet18 e DenseNet121.
- `src/data.py`: carregamento e pre-processamento reutilizavel.
- `src/evaluate.py`: metricas e matriz de confusao.
- `src/compare_results.py`: tabela comparativa entre CNN simples e Transfer Learning.
- `notebooks/pessoa2_transfer_learning_resultados.ipynb`: notebook com tabela, graficos e analise dos resultados.

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

Em maquina com GPU NVIDIA, confira se o PyTorch instalado reconhece CUDA:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Se aparecer `False`, instale a build CUDA do PyTorch:

```bash
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## Baixar e explorar o dataset

Configure suas credenciais do Kaggle em `~/.kaggle/kaggle.json` (gere o token em https://www.kaggle.com/settings) e rode:

```bash
python -m src.download_dataset
```

Isso baixa e organiza o dataset em `data/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}`.

Para conferir a estrutura de pastas, contar imagens por classe e gerar exemplos de radiografias:

```bash
python -m src.explore_dataset
```

Isso gera `results/dataset_class_counts.json` e `results/dataset_sample_images.png`.

## Treinar a CNN simples

```bash
python -m src.train_simple_cnn --data-dir data/chest_xray --epochs 15 --batch-size 32
```

O treinamento salva metricas, matriz de confusao e checkpoint em `results/`.

Arquivos gerados pela etapa da CNN simples:

- `results/simple_cnn_metrics.json`: metricas finais no teste.
- `results/simple_cnn_confusion_matrix.png`: matriz de confusao.
- `results/simple_cnn_training_history.csv`: historico por epoca.
- `results/simple_cnn_loss_history.png`: grafico da perda de treino e validacao.
- `results/simple_cnn_metrics_history.png`: grafico de acuracia e F1-score.
- `results/simple_cnn_best.pth`: melhor checkpoint pelo F1-score de validacao.

## Treinar Transfer Learning

```bash
python -m src.train_transfer --data-dir data/chest_xray --epochs 10 --batch-size 32
```

Configuracao final usada pela Pessoa 2:

```bash
python -m src.train_transfer --data-dir data/chest_xray --output-dir results/transfer_frozen_modelo --model resnet18 --epochs 10 --batch-size 32 --lr 0.001 --freeze-backbone
```

Experimento adicional com DenseNet121:

```bash
python -m src.train_transfer --data-dir data/chest_xray --output-dir results/transfer_densenet121_modelo --model densenet121 --epochs 10 --batch-size 32 --lr 0.001 --freeze-backbone
```

O treinamento salva metricas, matriz de confusao e checkpoint em `results/`.

Arquivos gerados pela etapa de Transfer Learning:

- `results/transfer_metrics.json`: metricas finais no teste.
- `results/transfer_confusion_matrix.png`: matriz de confusao.
- `results/transfer_training_history.csv`: historico por epoca.
- `results/transfer_loss_history.png`: grafico da perda de treino e validacao.
- `results/transfer_metrics_history.png`: grafico de acuracia e F1-score.
- `results/<modelo>_transfer_best.pth`: melhor checkpoint pelo F1-score de validacao.

Os resultados finais da Pessoa 2 estao resumidos em `docs/resultados_pessoa2.md` e no notebook `notebooks/pessoa2_transfer_learning_resultados.ipynb`.

## Comparar com a CNN simples

Ao rodar `python -m src.train_simple_cnn`, o arquivo `results/simple_cnn_metrics.json` e gerado automaticamente no formato:

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

Para incluir modelos adicionais na mesma tabela:

```bash
python -m src.compare_results --simple-cnn results/notebook_modelo/simple_cnn_metrics.json --transfer results/transfer_frozen_modelo/transfer_metrics.json --extra-metrics results/transfer_densenet121_modelo/transfer_metrics.json
```

Isso gera `results/comparison_table.csv`.

Tambem e gerada uma versao em Markdown em `results/comparison_table.md`, util para copiar a tabela para o relatorio.

## Materiais da Pessoa 1

- `docs/pessoa1_metodologia.md`: texto base para a metodologia do dataset e da CNN simples.
- `docs/checklist_pessoa1.md`: checklist operacional da sua parte.

## Materiais da Pessoa 2

- `docs/pessoa2_metodologia.md`: texto base para a metodologia de Transfer Learning.
- `docs/checklist_pessoa2.md`: checklist operacional da sua parte.
- `docs/resultados_pessoa2.md`: resumo dos resultados finais e comparacao entre modelos.
- `notebooks/pessoa2_transfer_learning_resultados.ipynb`: notebook leve para relatorio e apresentacao.

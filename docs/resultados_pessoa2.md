# Resultados da Pessoa 2 - Transfer Learning

## Configuracao final escolhida

O modelo final de Transfer Learning foi uma ResNet18 pre-treinada, com o backbone congelado e apenas a camada final treinada para as classes `NORMAL` e `PNEUMONIA`.

Configuracao usada:

- Epocas: 10
- Batch size: 32
- Learning rate: 0.001
- Backbone congelado: sim
- Dispositivo: GPU CUDA

Essa configuracao foi escolhida porque apresentou melhor desempenho no conjunto de teste do que a versao com fine-tuning completo do modelo.

## Experimento adicional

Tambem foi treinado um modelo DenseNet121 pre-treinado, usando a mesma estrategia da ResNet18: backbone congelado, 10 epocas, batch size 32 e learning rate 0.001. A intencao foi verificar se uma arquitetura frequentemente usada em imagens medicas superaria a ResNet18 no mesmo conjunto de dados.

## Comparacao final

| Modelo | Acuracia | Precisao | Recall | F1-score |
|---|---:|---:|---:|---:|
| CNN simples | 0.8766 | 0.9003 | 0.9026 | 0.9014 |
| ResNet18 Transfer Learning | 0.9054 | 0.9046 | 0.9487 | 0.9262 |
| DenseNet121 Transfer Learning | 0.8734 | 0.8676 | 0.9410 | 0.9028 |

## Analise breve

A ResNet18 com Transfer Learning apresentou o melhor desempenho geral entre os modelos avaliados. Ela superou a CNN simples em todas as metricas e tambem ficou acima da DenseNet121 no conjunto de teste.

A DenseNet121 teve desempenho valido e recall alto, mas nao superou a ResNet18. Esse resultado reforca a escolha da ResNet18 como modelo final, pois ela apresentou melhor equilibrio entre acuracia, precisao, recall e F1-score.

Como o conjunto de validacao original possui apenas 16 imagens, as metricas de validacao oscilaram bastante entre as epocas. Por isso, a comparacao final deve ser baseada principalmente no conjunto de teste, que possui 624 imagens e e mais representativo para avaliar os modelos.

## Matriz de confusao da ResNet18

Linhas representam a classe real e colunas representam a classe predita.

| Classe real | Predito NORMAL | Predito PNEUMONIA |
|---|---:|---:|
| NORMAL | 195 | 39 |
| PNEUMONIA | 20 | 370 |

## Artefatos gerados

- `results/transfer_frozen_modelo/transfer_metrics.json`
- `results/transfer_frozen_modelo/transfer_confusion_matrix.png`
- `results/transfer_frozen_modelo/transfer_confusion_matrix.json`
- `results/transfer_frozen_modelo/transfer_training_history.csv`
- `results/transfer_frozen_modelo/transfer_loss_history.png`
- `results/transfer_frozen_modelo/transfer_metrics_history.png`
- `results/transfer_densenet121_modelo/transfer_metrics.json`
- `results/transfer_densenet121_modelo/transfer_confusion_matrix.png`
- `results/transfer_densenet121_modelo/transfer_training_history.csv`
- `results/comparison_table.csv`
- `results/comparison_table.md`
- `notebooks/pessoa2_transfer_learning_resultados.ipynb`

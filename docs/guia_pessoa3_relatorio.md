# Guia para a Pessoa 3 - Relatório e apresentação

Este arquivo reúne os principais materiais já prontos no repositório para facilitar a montagem do relatório final e dos slides.

## Resultado principal

O modelo final recomendado para a etapa de Transfer Learning é a **ResNet18 com backbone congelado**. Ela apresentou o melhor desempenho geral no conjunto de teste, superando a CNN simples e a DenseNet121 nas métricas avaliadas.

| Modelo | Acurácia | Precisão | Recall | F1-score |
|---|---:|---:|---:|---:|
| CNN simples | 0.8766 | 0.9003 | 0.9026 | 0.9014 |
| ResNet18 Transfer Learning | 0.9054 | 0.9046 | 0.9487 | 0.9262 |
| DenseNet121 Transfer Learning | 0.8734 | 0.8676 | 0.9410 | 0.9028 |

## Arquivos mais importantes

- `notebooks/pessoa2_transfer_learning_resultados.ipynb`: notebook visual com tabela, matrizes de confusão, gráficos e conclusão da Pessoa 2.
- `docs/resultados_pessoa2.md`: resumo textual dos resultados da Pessoa 2.
- `docs/pessoa2_metodologia.md`: texto-base da metodologia de Transfer Learning.
- `results/comparison_table.md`: tabela comparativa em Markdown.
- `results/comparison_table.csv`: tabela comparativa em CSV.

## Figuras recomendadas para o relatório

- `results/notebook_modelo/dataset_sample_images.png`: exemplos do dataset.
- `results/report_figures/cnn_simples_confusion_matrix.png`: matriz de confusão da CNN simples, com título pronto para relatório.
- `results/report_figures/resnet18_confusion_matrix.png`: matriz de confusão da ResNet18, com título pronto para relatório.
- `results/report_figures/densenet121_confusion_matrix.png`: matriz de confusão da DenseNet121, com título pronto para relatório.
- `results/report_figures/resnet18_loss_history.png`: histórico de perda da ResNet18.
- `results/report_figures/resnet18_metrics_history.png`: histórico de métricas da ResNet18.
- `results/report_figures/densenet121_loss_history.png`: histórico de perda da DenseNet121.
- `results/report_figures/densenet121_metrics_history.png`: histórico de métricas da DenseNet121.

## Texto-base para a metodologia da Pessoa 2

Na etapa de Transfer Learning, foi utilizada uma ResNet18 pré-treinada no ImageNet. A camada final original foi substituída por uma nova camada totalmente conectada com duas saídas, correspondentes às classes `NORMAL` e `PNEUMONIA`. O backbone do modelo foi mantido congelado, treinando-se apenas a nova camada final.

As imagens foram redimensionadas para `224 x 224` pixels, convertidas para tensores e normalizadas com média e desvio padrão do ImageNet. No conjunto de treino, foram aplicadas transformações simples de aumento de dados, incluindo espelhamento horizontal aleatório e pequenas rotações, com o objetivo de melhorar a generalização.

O treinamento utilizou a função de perda `CrossEntropyLoss` e o otimizador Adam, com taxa de aprendizado igual a `0.001`, batch size `32` e `10` épocas. O melhor checkpoint foi selecionado com base no F1-score obtido no conjunto de validação.

Também foi treinada uma DenseNet121 pré-treinada, usando a mesma estratégia de congelamento do backbone. Esse experimento adicional teve como objetivo verificar se uma arquitetura frequentemente utilizada em tarefas de imagem médica superaria a ResNet18 no mesmo conjunto de dados.

## Texto-base para análise dos resultados

A ResNet18 com Transfer Learning apresentou o melhor desempenho geral entre os modelos avaliados. Em comparação com a CNN simples, a ResNet18 obteve maior acurácia, precisão, recall e F1-score. O ganho em recall é especialmente relevante, pois, em um problema relacionado à detecção de pneumonia, falsos negativos são indesejáveis.

A DenseNet121 também apresentou desempenho válido, com recall alto, mas não superou a ResNet18. Dessa forma, a ResNet18 foi escolhida como modelo final da etapa de Transfer Learning por apresentar o melhor equilíbrio entre as métricas avaliadas.

É importante destacar que o conjunto de validação original do dataset possui apenas 16 imagens, o que torna as métricas de validação instáveis. Por esse motivo, a comparação final entre os modelos deve se basear principalmente no conjunto de teste, que possui 624 imagens.

## Matriz de confusão da ResNet18

Linhas representam a classe real e colunas representam a classe predita.

| Classe real | Predito NORMAL | Predito PNEUMONIA |
|---|---:|---:|
| NORMAL | 195 | 39 |
| PNEUMONIA | 20 | 370 |

Interpretação:

- 195 imagens normais foram classificadas corretamente como `NORMAL`.
- 39 imagens normais foram classificadas incorretamente como `PNEUMONIA`.
- 370 imagens com pneumonia foram classificadas corretamente como `PNEUMONIA`.
- 20 imagens com pneumonia foram classificadas incorretamente como `NORMAL`.

## Sugestão de estrutura para o relatório

1. Introdução
2. Objetivo
3. Fundamentação teórica
4. Dataset
5. Pré-processamento
6. CNN simples
7. Transfer Learning
8. Métricas de avaliação
9. Resultados
10. Discussão
11. Conclusão
12. Referências

## Pontos que devem aparecer na conclusão

- A CNN simples serviu como modelo de referência treinado do zero.
- A ResNet18 com Transfer Learning apresentou o melhor resultado geral.
- A DenseNet121 foi testada como alternativa, mas não superou a ResNet18.
- O uso de Transfer Learning foi vantajoso para o problema.
- A validação pequena é uma limitação metodológica importante.
- Os resultados não devem ser interpretados como validação clínica, mas como experimento acadêmico de classificação de imagens.

## Sugestão para slides

- Slide 1: título, integrantes e tema.
- Slide 2: problema e objetivo.
- Slide 3: dataset e exemplos de imagens.
- Slide 4: pré-processamento.
- Slide 5: CNN simples.
- Slide 6: Transfer Learning com ResNet18 e DenseNet121.
- Slide 7: tabela comparativa.
- Slide 8: matriz de confusão da ResNet18.
- Slide 9: análise dos resultados.
- Slide 10: conclusão e limitações.

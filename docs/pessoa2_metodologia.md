# Pessoa 2 - Transfer Learning e comparacao final

## Objetivo da etapa

Esta etapa tem como objetivo treinar um modelo mais avancado para classificar radiografias toracicas nas classes `NORMAL` e `PNEUMONIA`, utilizando Transfer Learning. A proposta e comparar esse modelo com a CNN simples treinada pela Pessoa 1, usando as mesmas divisoes de treino, validacao e teste.

## Modelo utilizado

O modelo escolhido foi a ResNet18, uma rede neural convolucional profunda com conexoes residuais. A arquitetura original foi pre-treinada no ImageNet, o que permite aproveitar filtros visuais ja aprendidos em uma base grande de imagens. Para adaptar o modelo ao problema deste trabalho, a camada final totalmente conectada foi substituida por uma nova camada com duas saidas, correspondentes as classes `NORMAL` e `PNEUMONIA`.

## Pre-processamento

As imagens sao redimensionadas para `224 x 224` pixels, formato esperado pela ResNet18. Em seguida, sao convertidas para tensores e normalizadas com a media e o desvio padrao do ImageNet. No conjunto de treino, tambem sao aplicadas transformacoes simples de aumento de dados, como espelhamento horizontal aleatorio e pequenas rotacoes, com o objetivo de melhorar a generalizacao do modelo.

## Treinamento

O treinamento utiliza a funcao de perda `CrossEntropyLoss`, adequada para classificacao multiclasse, mesmo quando ha apenas duas classes. O otimizador utilizado e o Adam, com taxa de aprendizado configuravel. Durante o treinamento, o desempenho no conjunto de validacao e acompanhado a cada epoca. O checkpoint salvo corresponde ao modelo que obteve o melhor F1-score de validacao.

## Avaliacao

Apos o treinamento, o melhor checkpoint e avaliado no conjunto de teste. As metricas calculadas sao acuracia, precisao, recall e F1-score. Tambem e gerada uma matriz de confusao para visualizar os acertos e erros entre as classes `NORMAL` e `PNEUMONIA`.

## Comparacao final

Para a comparacao final, as metricas da CNN simples devem ser fornecidas pela Pessoa 1 no mesmo conjunto de teste. A tabela comparativa deve conter:

| Modelo | Acuracia | Precisao | Recall | F1-score |
|---|---:|---:|---:|---:|
| CNN simples | a preencher | a preencher | a preencher | a preencher |
| ResNet18 Transfer Learning | a preencher | a preencher | a preencher | a preencher |

Espera-se que a ResNet18 com Transfer Learning apresente desempenho competitivo ou superior a CNN simples, pois parte de representacoes visuais previamente aprendidas em larga escala. Mesmo assim, a conclusao deve ser baseada nos resultados obtidos no conjunto de teste, especialmente em recall e F1-score, que sao metricas importantes em problemas de classificacao medica.

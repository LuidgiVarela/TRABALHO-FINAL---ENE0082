# Pessoa 1 - Dataset, pre-processamento e CNN simples

## Objetivo da etapa

Esta etapa prepara a base comum do projeto (dataset, carregamento e pre-processamento) e treina o primeiro modelo, uma CNN simples construida do zero, para classificar radiografias toracicas nas classes `NORMAL` e `PNEUMONIA`.

## Dataset

O dataset utilizado e o Chest X-Ray Images (Pneumonia), disponivel publicamente no Kaggle. Ele contem radiografias de torax organizadas em tres divisoes (`train`, `val`, `test`), cada uma com duas subpastas de classe (`NORMAL` e `PNEUMONIA`). A estrutura e a contagem de imagens por classe sao conferidas pelo script `src/explore_dataset.py`, que tambem gera exemplos visuais de radiografias normais e com pneumonia.

## Pre-processamento

O carregamento e o pre-processamento das imagens ficam em `src/data.py` e sao compartilhados pelos dois modelos do projeto (CNN simples e Transfer Learning), conforme combinado no grupo. As imagens sao redimensionadas para `224 x 224` pixels, convertidas para tensor e normalizadas com a media e o desvio padrao do ImageNet, o que mantem a compatibilidade com a ResNet18 usada pela Pessoa 2. No conjunto de treino, tambem sao aplicadas transformacoes simples de aumento de dados (espelhamento horizontal e pequenas rotacoes).

## Modelo utilizado

O modelo `SimpleCNN` (`src/models.py`) e uma rede convolucional pequena, treinada inteiramente do zero (sem pesos pre-treinados), com quatro blocos de convolucao + normalizacao + ReLU + max pooling, seguidos de uma camada de pooling global e uma camada linear final para as duas classes. O objetivo dela e servir de linha de base simples para comparacao com o modelo de Transfer Learning.

## Treinamento

O treinamento (`src/train_simple_cnn.py`) utiliza `CrossEntropyLoss` e o otimizador Adam, atualizando todos os parametros do modelo. O desempenho no conjunto de validacao e acompanhado a cada epoca e o checkpoint salvo corresponde ao modelo com melhor F1-score de validacao.

## Avaliacao

Apos o treinamento, o melhor checkpoint e avaliado no conjunto de teste, usando as funcoes de `src/evaluate.py` (compartilhadas com a Pessoa 2). As metricas calculadas sao acuracia, precisao, recall e F1-score, alem da matriz de confusao.

## Entrega para a Pessoa 2

O arquivo `results/simple_cnn_metrics.json`, gerado automaticamente ao final do treinamento, e o insumo que a Pessoa 2 usa em `src/compare_results.py` para montar a tabela comparativa final entre a CNN simples e a ResNet18 com Transfer Learning.

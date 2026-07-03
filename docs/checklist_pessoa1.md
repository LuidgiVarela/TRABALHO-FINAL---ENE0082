# Checklist da Pessoa 1

- [ ] Configurar as credenciais do Kaggle em `~/.kaggle/kaggle.json`.
- [ ] Instalar as dependencias com `pip install -r requirements.txt`.
- [ ] Baixar e organizar o dataset com `python -m src.download_dataset`.
- [ ] Conferir se o dataset esta em `data/chest_xray/train`, `data/chest_xray/val` e `data/chest_xray/test`, cada um com `NORMAL` e `PNEUMONIA`.
- [ ] Rodar `python -m src.explore_dataset` para conferir a estrutura, contar imagens por classe e gerar exemplos de radiografias.
- [ ] Conferir `results/dataset_class_counts.json` e `results/dataset_sample_images.png`.
- [ ] Treinar a CNN simples com `python -m src.train_simple_cnn --data-dir data/chest_xray --epochs 15 --batch-size 32`.
- [ ] Conferir `results/simple_cnn_metrics.json`.
- [ ] Conferir `results/simple_cnn_confusion_matrix.png`.
- [ ] Conferir os graficos `results/simple_cnn_loss_history.png` e `results/simple_cnn_metrics_history.png`.
- [ ] Enviar para a Pessoa 2 o arquivo `results/simple_cnn_metrics.json` para a comparacao final.
- [ ] Enviar para a Pessoa 3 as imagens de exemplo, a contagem por classe, a matriz de confusao e o texto de metodologia.

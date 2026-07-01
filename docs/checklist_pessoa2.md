# Checklist da Pessoa 2

- [ ] Baixar o dataset Chest X-Ray Images (Pneumonia).
- [ ] Conferir se o dataset esta em `data/chest_xray/train`, `data/chest_xray/val` e `data/chest_xray/test`.
- [ ] Instalar as dependencias com `pip install -r requirements.txt`.
- [ ] Treinar a ResNet18 com `python -m src.train_transfer --data-dir data/chest_xray --epochs 10 --batch-size 32`.
- [ ] Conferir `results/transfer_metrics.json`.
- [ ] Conferir `results/transfer_confusion_matrix.png`.
- [ ] Conferir os graficos `results/transfer_loss_history.png` e `results/transfer_metrics_history.png`.
- [ ] Receber as metricas da CNN simples da Pessoa 1.
- [ ] Gerar a comparacao final com `python -m src.compare_results --simple-cnn results/simple_cnn_metrics.json --transfer results/transfer_metrics.json`.
- [ ] Enviar para a Pessoa 3 a tabela comparativa, a matriz de confusao e o texto de metodologia.

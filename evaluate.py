from utils.metrics import compute_metrics

metrics = compute_metrics(y_true, y_pred, y_probs)
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

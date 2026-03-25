import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000/evaluate"

MODELS = [
    {
        "model_id": "logistic_regression",
        "csv_file": "outputs/predictions_logreg.csv",
        "dataset": "test_data"
    },
    {
        "model_id": "random_forest",
        "csv_file": "outputs/predictions_rf.csv",
        "dataset": "test_data"
    }
]

test_df = pd.read_csv("model_data/test.csv")
test_df.columns = test_df.columns.str.strip()

label_column = "Label"
true_labels = (
    test_df[label_column].astype(str).str.strip().str.upper() != "BENIGN"
).astype(int).tolist()

all_metrics = []

for model_info in MODELS:
    model_id = model_info["model_id"]
    csv_file = model_info["csv_file"]
    dataset = model_info["dataset"]

    print(f"\n=== Evaluating {model_id} ===")

    df = pd.read_csv(csv_file)

    if len(df) != len(true_labels):
        raise ValueError(
            f"{csv_file}: number of predictions ({len(df)}) "
            f"does not match number of test labels ({len(true_labels)})"
        )

    predictions = []
    for idx, row in df.iterrows():
        predictions.append({
            "id": str(idx),
            "predicted_score": float(row["confidence"]),
            "predicted_class": int(row["prediction"]),
            "true_label": int(true_labels[idx])
        })

    request_data = {
        "model_id": model_id,
        "dataset": dataset,
        "predictions": predictions
    }

    response = requests.post(API_URL, json=request_data)

    if response.status_code == 200:
        metrics = response.json()

        print("Тест успешен! Метрики:")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-score:  {metrics['f1_score']:.4f}")
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"FAR:       {metrics['far']:.4f}")
        print(f"FRR:       {metrics['frr']:.4f}")

        all_metrics.append({
            "model": model_id,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "roc_auc": metrics["roc_auc"],
            "far": metrics["far"],
            "frr": metrics["frr"]
        })
    else:
        print(f"Ошибка для {model_id}: {response.status_code}")
        print(response.text)

metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_csv("metrics_comparison.csv", index=False)

print("\nМетрики сохранены в metrics_comparison.csv")
print(metrics_df)
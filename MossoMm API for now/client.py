import requests
import pandas as pd
from typing import Optional, List, Dict
# 1. Загружаем тестовые данные
df = pd.read_csv("test_predictions.csv")

# 2. Превращаем в формат для API
predictions = []
for _, row in df.iterrows():
    pred_class = 1 if row["score"] >= 0.5 else 0
    predictions.append({
        "id": str(row["id"]),
        "predicted_score": float(row["score"]),
        "predicted_class": pred_class,
        "true_label": int(row["true_label"])
    })

# 3. Формируем запрос
request_data = {
    "model_id": "test_model",
    "dataset": "test_data",
    "predictions": predictions
}

# 4. Отправляем на API
response = requests.post("http://127.0.0.1:8000/evaluate", json=request_data)

# 5. Смотрим результат
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
else:
    print(f"Ошибка: {response.status_code}")
    print(response.text)

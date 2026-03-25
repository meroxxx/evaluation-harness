import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

train_df = pd.read_csv('model_data/train.csv')
val_df = pd.read_csv('model_data/val.csv')
test_df = pd.read_csv('model_data/test.csv')

train_df.columns = train_df.columns.str.strip()
val_df.columns = val_df.columns.str.strip()
test_df.columns = test_df.columns.str.strip()

print(f"Train shape: {train_df.shape}")
print(f"Val shape:   {val_df.shape}")
print(f"Test shape:  {test_df.shape}")

label_column = 'Label'

print("\nUnique values in train label:")
print(train_df[label_column].value_counts())

print("\nUnique values in val label:")
print(val_df[label_column].value_counts())

print("\nUnique values in test label:")
print(test_df[label_column].value_counts())

train_df['target'] = (
    train_df[label_column].astype(str).str.strip().str.upper() != 'BENIGN'
).astype(int)

val_df['target'] = (
    val_df[label_column].astype(str).str.strip().str.upper() != 'BENIGN'
).astype(int)

test_df['target'] = (
    test_df[label_column].astype(str).str.strip().str.upper() != 'BENIGN'
).astype(int)

print("\nTarget distribution in train:")
print(train_df['target'].value_counts())

train_df = train_df.drop(label_column, axis=1)
val_df = val_df.drop(label_column, axis=1)
test_df = test_df.drop(label_column, axis=1)

X_train = train_df.drop('target', axis=1)
y_train = train_df['target']

X_val = val_df.drop('target', axis=1)
X_test = test_df.drop('target', axis=1)

print("\nPreprocessing data...")

X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_val = X_val.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)

X_train = X_train.fillna(X_train.median(numeric_only=True))
X_val = X_val.fillna(X_train.median(numeric_only=True))
X_test = X_test.fillna(X_train.median(numeric_only=True))

categorical_cols = X_train.select_dtypes(include=['object']).columns
if len(categorical_cols) > 0:
    print(f"Encoding {len(categorical_cols)} categorical columns...")
    for col in categorical_cols:
        le = LabelEncoder()
        combined = pd.concat([X_train[col], X_val[col], X_test[col]], axis=0).astype(str)
        le.fit(combined)
        X_train[col] = le.transform(X_train[col].astype(str))
        X_val[col] = le.transform(X_val[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train_scaled, y_train)

print("\nMaking predictions on test set...")

y_proba = model.predict_proba(X_test_scaled)[:, 1]

y_pred = (y_proba >= 0.5).astype(int)

results_df = pd.DataFrame({
    'prediction': y_pred,
    'confidence': y_proba
})

output_file = 'predictions_logreg.csv'
results_df.to_csv(output_file, index=False)

print(f"\nResults saved to: {output_file}")
print(f"Total predictions: {len(results_df)}")
print("\nSample of results (first 10 rows):")
print(results_df.head(10).to_string())

print("\nPrediction distribution:")
print(f"Class 0 (normal): {(y_pred == 0).sum()} samples ({((y_pred == 0).sum()/len(y_pred)*100):.1f}%)")
print(f"Class 1 (attack): {(y_pred == 1).sum()} samples ({((y_pred == 1).sum()/len(y_pred)*100):.1f}%)")
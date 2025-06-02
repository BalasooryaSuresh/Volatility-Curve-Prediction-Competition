import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# File paths
train_path = "train_data.parquet"
test_path = "test_data.parquet"
sample_path = "sample_submission.csv"
output_path = "final_optimized_submission.csv"

# Load data
train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)
sample = pd.read_csv(sample_path)

# Target columns
target_cols = [col for col in sample.columns if col != 'timestamp' and col in train.columns]
feature_cols = [f"X{i}" for i in range(42)] + ["underlying"]

# Timestamp feature
for df in (train, test):
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['sec_from_start'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()

feature_cols.append("sec_from_start")

# Feature engineering function
def add_features(df):
    df = df.copy()
    df['underlying_diff'] = df['underlying'].diff().fillna(0)
    df['underlying_roll_mean_5'] = df['underlying'].rolling(window=5, min_periods=1).mean()
    df['underlying_roll_std_5'] = df['underlying'].rolling(window=5, min_periods=1).std().fillna(0)
    return df

# Apply feature engineering
X_train = add_features(train[feature_cols])
X_test = add_features(test[feature_cols])
y_train = train[target_cols]

# Standard scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA to reduce noise
pca = PCA(n_components=0.999)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Model pipeline
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("regressor", MultiOutputRegressor(
        HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.03,
            max_depth=4,
            l2_regularization=1.0,
            random_state=42
        )
    ))
])

# Train model
model.fit(X_train_pca, y_train)

# Predict
preds = pd.DataFrame(model.predict(X_test_pca), columns=target_cols)

# Fill in predictions
submission = sample.copy()
for col in target_cols:
    submission[col] = submission[col].combine_first(preds[col])

# Save output
submission.to_csv(output_path, index=False)
print(f"Saved optimized submission to: {output_path}")

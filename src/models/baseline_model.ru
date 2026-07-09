import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATA_PATH = os.path.join("data", "raw", "weight_change_dataset.csv")
MODEL_OUT_PATH = os.path.join("models", "baseline_linear_regression.joblib")
RESULTS_PATH = os.path.join("reports", "baseline_metrics.csv")
TARGET_COL = "Weight Change (lbs)"
MANUAL_DROP_COLS = ["Participant ID", "Final Weight (lbs)"]

RANDOM_STATE = 42
TEST_SIZE = 0.2

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Загружено {df.shape[0]} строк и {df.shape[1]} столбцов из {path}")
    print("\nКолонки в датасете:")
    print(list(df.columns))
    print("\nПервые строки:")
    print(df.head())
    print("\nТипы данных и пропуски:")
    print(df.info())
    return df


def find_target_column(df: pd.DataFrame) -> str:
    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Колонки '{TARGET_COL}' нет в датасете. "
            f"Доступные колонки: {list(df.columns)}"
        )
    print(f"\nЦелевая колонка: '{TARGET_COL}'")
    return TARGET_COL

def split_features_target(df: pd.DataFrame, target_col: str):
    drop_cols = MANUAL_DROP_COLS + [target_col]
    X = df.drop(columns=drop_cols, errors="ignore")
    y = df[target_col]

    mask = y.notna()
    X, y = X[mask], y[mask]
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    print(f"\nЧисловые признаки ({len(numeric_cols)}): {numeric_cols}")
    print(f"Категориальные признаки ({len(categorical_cols)}): {categorical_cols}")
    return X, y, numeric_cols, categorical_cols


def build_preprocessor(numeric_cols, categorical_cols) -> ColumnTransformer:
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ])
    return preprocessor

def train_and_evaluate(X, y, numeric_cols, categorical_cols):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"\nTrain: {X_train.shape[0]} строк, Test: {X_test.shape[0]} строк")
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression()),
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print("\n Метрики baseline-модели ")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R^2  : {r2:.4f}")

    naive_pred = np.full_like(y_test, fill_value=y_train.mean(), dtype=float)
    naive_mae = mean_absolute_error(y_test, naive_pred)
    naive_rmse = np.sqrt(mean_squared_error(y_test, naive_pred))
    print("\n Наивный прогноз (среднее значение)")
    print(f"MAE  : {naive_mae:.4f}")
    print(f"RMSE : {naive_rmse:.4f}")
    return pipeline, {"MAE": mae, "RMSE": rmse, "R2": r2}


def save_model(pipeline: Pipeline, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)
    print(f"\nМодель сохранена в {path}")


def save_metrics(metrics: dict, path: str):
    row = {"Model": "LinearRegression (baseline)", **metrics}
    results_df = pd.DataFrame([row])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    results_df.to_csv(path, index=False)
    print(f"Метрики сохранены в {path}")


def main():
    df = load_data(DATA_PATH)
    target_col = find_target_column(df)
    X, y, numeric_cols, categorical_cols = split_features_target(df, target_col)
    pipeline, metrics = train_and_evaluate(X, y, numeric_cols, categorical_cols)
    save_model(pipeline, MODEL_OUT_PATH)
    save_metrics(metrics, RESULTS_PATH)
if __name__ == "__main__":
    main()

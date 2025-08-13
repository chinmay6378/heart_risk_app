# train_model.py
import os
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, accuracy_score

DATA_PATH = Path("data/heart.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "heart_model.pkl"

def load_data():
    df = pd.read_csv(DATA_PATH)
    # Standard Kaggle/UCI columns:
    # age, sex, cp, trestbps, chol, fbs, restecg, thalach,
    # exang, oldpeak, slope, ca, thal, target
    return df

def build_pipeline(df):
    X = df.drop(columns=["target"])
    y = df["target"]

    # Categorical vs numeric columns
    cat_cols = ["sex","cp","fbs","restecg","exang","slope","ca","thal"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    # Base model
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        class_weight="balanced_subsample"
    )

    # Calibrate probabilities with sigmoid on validation folds
    clf = CalibratedClassifierCV(rf, method="sigmoid", cv=5)

    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])
    return pipe, X, y

def train_and_save():
    df = load_data()
    pipe, X, y = build_pipeline(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe.fit(X_train, y_train)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("ROC-AUC:", round(roc_auc_score(y_test, y_proba), 4))

    joblib.dump({"model": pipe, "feature_names": list(X.columns)}, MODEL_PATH)
    print(f"Saved model → {MODEL_PATH.resolve()}")

if __name__ == "__main__":
    if not DATA_PATH.exists():
        raise FileNotFoundError("Place dataset at data/heart.csv (UCI/Kaggle heart dataset).")
    train_and_save()

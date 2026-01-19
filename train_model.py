import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# ================= Load Data =================
df = pd.read_csv("Telco-Customer-Churn.csv")

# Drop ID
df.drop("customerID", axis=1, inplace=True)

# Fix TotalCharges
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# Target encoding
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Feature engineering
df["AvgCharges"] = df["TotalCharges"] / (df["tenure"] + 1)
df["Is_Long_Term"] = (df["tenure"] > 24).astype(int)

X = df.drop("Churn", axis=1)
y = df["Churn"]

# Column types
categorical_cols = X.select_dtypes(include="object").columns
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
    ]
)

# Model pipeline
model = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("rf", RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            class_weight="balanced",
            random_state=42
        ))
    ]
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Train
model.fit(X_train, y_train)

# Save model
joblib.dump(
    model,
    r"C:\Users\mulye\PycharmProjects\PythonProject1\Customer-Churn-Prediction\rf_churn_model.pkl"
)


print("✅ Model trained & saved as rf_churn_model.pkl")

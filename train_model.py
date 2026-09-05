
import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

RANDOM_STATE = 42

os.makedirs("models", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

# Load data
df = pd.read_csv("data/tourism.csv")

# Remove unnecessary column
if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)

# Data cleaning
df["Gender"] = df["Gender"].replace({
    "Fe Male": "Female"
})

df["MaritalStatus"] = df["MaritalStatus"].replace({
    "Unmarried": "Single"
})

df["Occupation"] = df["Occupation"].replace({
    "Free Lancer": "Freelancer"
})

df.drop_duplicates(inplace=True)

# Features and target
X = df.drop(columns=["ProdTaken", "CustomerID"])
y = df["ProdTaken"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

# Save splits
train_df = X_train.copy()
train_df["ProdTaken"] = y_train.values

test_df = X_test.copy()
test_df["ProdTaken"] = y_test.values

train_df.to_csv("artifacts/train.csv", index=False)
test_df.to_csv("artifacts/test.csv", index=False)

# Feature types
categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_features = X.select_dtypes(
    exclude=["object"]
).columns.tolist()

# Preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numerical_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# Model
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        random_state=RANDOM_STATE,
        class_weight="balanced"
    ))
])

# Hyperparameter tuning
param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [10, None],
    "model__min_samples_split": [2, 5]
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

grid = GridSearchCV(
    pipeline,
    param_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

# Evaluation
pred = best_model.predict(X_test)
prob = best_model.predict_proba(X_test)[:, 1]

results = {
    "accuracy": accuracy_score(y_test, pred),
    "precision": precision_score(y_test, pred),
    "recall": recall_score(y_test, pred),
    "f1_score": f1_score(y_test, pred),
    "roc_auc": roc_auc_score(y_test, prob),
    "best_parameters": grid.best_params_
}

print("MODEL RESULTS")
print(json.dumps(results, indent=4))

# Save experiment
with open("artifacts/experiment.json", "w") as f:
    json.dump(results, f, indent=4)

# Save model
joblib.dump(
    best_model,
    "models/tourism_purchase_model.pkl"
)

print("Model saved successfully.")

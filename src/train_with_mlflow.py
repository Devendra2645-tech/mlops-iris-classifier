import os
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# -------------------------------------------------
# 1. MLflow configuration
# -------------------------------------------------

mlflow.set_tracking_uri(
    os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
)

mlflow.set_experiment("iris-classification-baseline")


# -------------------------------------------------
# 2. Load dataset
# -------------------------------------------------

df = pd.read_csv("data/processed/iris_features.csv")

print("Dataset shape:", df.shape)


# -------------------------------------------------
# 3. Features and target
# -------------------------------------------------

feature_cols = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
    "sepal_area",
    "petal_area",
    "sepal_to_petal_length_ratio"
]

X = df[feature_cols].copy()
y = df["species"].copy()


# -------------------------------------------------
# 4. Handle missing values
# -------------------------------------------------

X = X.fillna(X.median())


# -------------------------------------------------
# 5. Encode target
# -------------------------------------------------

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


# -------------------------------------------------
# 6. Train-test split
# -------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)


# -------------------------------------------------
# 7. Models
# -------------------------------------------------

models = [
    (
        "Logistic Regression",
        LogisticRegression(max_iter=200, C=1.0),
        {
            "model_type": "LogisticRegression",
            "max_iter": 200,
            "C": 1.0
        }
    ),
    (
        "Random Forest Shallow",
        RandomForestClassifier(
            n_estimators=50,
            max_depth=3,
            random_state=42
        ),
        {
            "model_type": "RandomForestClassifier",
            "n_estimators": 50,
            "max_depth": 3
        }
    ),
    (
        "Random Forest Deep",
        RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=42
        ),
        {
            "model_type": "RandomForestClassifier",
            "n_estimators": 200,
            "max_depth": "None"
        }
    )
]


# -------------------------------------------------
# 8. Train and log each model
# -------------------------------------------------

results = []

for name, model, params in models:

    print(f"\nTraining: {name}")

    with mlflow.start_run(run_name=name):

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        )
        recall = recall_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        )
        f1 = f1_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        )

        # Log parameters
        mlflow.log_params(params)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision_macro", precision)
        mlflow.log_metric("recall_macro", recall)
        mlflow.log_metric("f1_macro", f1)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=label_encoder.classes_
        )

        disp.plot()
        plt.title(f"Confusion Matrix - {name}")
        plt.tight_layout()

        cm_file = f"confusion_matrix_{name.replace(' ', '_')}.png"

        plt.savefig(cm_file)
        plt.close()

        # Log confusion matrix
        mlflow.log_artifact(cm_file)

        # Log model
        mlflow.sklearn.log_model(
            model,
            "model"
        )

        # Store results
        results.append({
            "model": name,
            "accuracy": accuracy,
            "precision_macro": precision,
            "recall_macro": recall,
            "f1_macro": f1
        })

        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")


# -------------------------------------------------
# 9. Compare models
# -------------------------------------------------

results_df = pd.DataFrame(results)

print("\n========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))


# -------------------------------------------------
# 10. Select best model
# -------------------------------------------------

best_model = results_df.loc[
    results_df["f1_macro"].idxmax()
]

print("\n========== BEST MODEL ==========")
print(best_model)
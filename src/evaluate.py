
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score( y_test, y_pred, average="macro", zero_division=0 ),
        "recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0)
    }


    cm = confusion_matrix(y_test, y_pred)

    # affichage
    print("\n===== MODEL EVALUATION =====")

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("\n===== CLASSIFICATION REPORT =====")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("\n===== CONFUSION MATRIX =====")
    print(cm)

    return metrics, cm

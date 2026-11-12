from src.data_loader import load_data
from src.preprocessing import prepare_data, split_data
from src.evaluate import evaluate_model
import joblib

def main():
    # chargement
    df = load_data()

    # préparation
    X, y = prepare_data(df)

    # séparation Train/Test
    X_train, X_test, y_train, y_test = split_data(X, y)


    # entraînement
    model = joblib.load("models/mentalrisk_model.joblib")

    # evaluation
    metrics,cm = evaluate_model(model, X_test, y_test)
    print("Résultats :", metrics)
    print("\nConfusion Matrix :", cm)


if __name__ == "__main__":
    main()
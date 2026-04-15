import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def create_dataset(samples=120):
    """
    Generate a custom classification dataset.
    """
    X = np.random.rand(samples, 4)
    y = (X[:, 0] + 0.5 * X[:, 1] + X[:, 2] > 1.2).astype(int)
    return X, y


def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=37,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    accuracy = model.score(X_test, y_test)
    print(f"Model Accuracy: {accuracy:.3f}")
    print("Feature Importances:", model.feature_importances_)


def main():
    print("Starting Random Forest training pipeline...")

    X, y = create_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=1
    )

    model = train_model(X_train, y_train)

    print("Training completed successfully.")
    evaluate_model(model, X_test, y_test)

    print("Pipeline execution finished.")


if __name__ == "__main__":
    main()
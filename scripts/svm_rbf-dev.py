import argparse
import numpy as np

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    cross_validate
)
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from pretraitement import X, y


def main():
    parser = argparse.ArgumentParser(description="Pipeline SVM complet (linéaire + RBF)")

    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=50000)
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="none")

    args = parser.parse_args()

    class_weight_value = "balanced" if args.class_weight == "balanced" else None

    # =========================
    # 1. SPLIT TEST (final)
    # =========================
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=42,
        stratify=y
    )

    # =========================
    # 2. SPLIT TRAIN / DEV
    # =========================
    X_train, X_dev, y_train, y_dev = train_test_split(
        X_temp, y_temp,
        test_size=0.25,
        random_state=42,
        stratify=y_temp
    )

    print("Paramètres utilisés :")
    print("test_size =", args.test_size)
    print("n_splits =", args.n_splits)
    print("max_iter =", args.max_iter)
    print("class_weight =", class_weight_value)

    # =========================
    # 3. GRID SEARCH LinearSVC
    # =========================
    param_grid = {
        "C": [0.001, 0.01, 0.1, 0.2, 1, 10, 100]
    }

    grid = GridSearchCV(
        LinearSVC(
            max_iter=args.max_iter,
            random_state=42,
            class_weight=class_weight_value
        ),
        param_grid,
        cv=3,
        scoring="f1_macro",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print("\nMeilleur C (LinearSVC) :", grid.best_params_)

    best_linear = grid.best_estimator_

    # =========================
    # 4. ÉVAL DEV (Linear)
    # =========================
    y_dev_pred = best_linear.predict(X_dev)

    print("\n=== DEV LinearSVC ===")
    print("Accuracy :", accuracy_score(y_dev, y_dev_pred))
    print(classification_report(y_dev, y_dev_pred))

    # =========================
    # 5. VALIDATION CROISÉE
    # =========================
    cv = StratifiedKFold(
        n_splits=args.n_splits,
        shuffle=True,
        random_state=42
    )

    print("\n=== CROSS-VALIDATION (TRAIN) ===")

    cv_linear = cross_validate(
        best_linear,
        X_train,
        y_train,
        cv=cv,
        scoring=["accuracy", "f1_macro"]
    )

    print("LinearSVC - Accuracy :", cv_linear["test_accuracy"].mean())
    print("LinearSVC - F1 macro :", cv_linear["test_f1_macro"].mean())

    # =========================
    # 6. GRID SEARCH RBF
    # =========================
    param_grid_rbf = {
        "C": [0.001, 0.01, 0.1, 0.2, 1, 10, 100],
        "gamma": ["scale", 0.01, 0.1, 1]
    }

    grid_rbf = GridSearchCV(
        SVC(
            kernel="rbf",
            class_weight=class_weight_value,
            random_state=42
        ),
        param_grid_rbf,
        cv=3,
        scoring="f1_macro",
        n_jobs=-1
    )

    grid_rbf.fit(X_train, y_train)

    print("\nMeilleurs paramètres RBF :", grid_rbf.best_params_)

    best_rbf = grid_rbf.best_estimator_

    # =========================
    # 7. ÉVAL DEV (RBF)
    # =========================
    y_dev_pred_rbf = best_rbf.predict(X_dev)

    print("\n=== DEV RBF ===")
    print("Accuracy :", accuracy_score(y_dev, y_dev_pred_rbf))
    print(classification_report(y_dev, y_dev_pred_rbf))

    # =========================
    # 8. TRAIN FINAL (TRAIN + DEV)
    # =========================
    X_final = np.vstack((X_train, X_dev))
    y_final = np.hstack((y_train, y_dev))

    best_linear.fit(X_final, y_final)
    best_rbf.fit(X_final, y_final)

    # =========================
    # 9. TEST FINAL
    # =========================
    print("\n================ TEST FINAL ================")

    # Linear
    y_test_linear = best_linear.predict(X_test)

    print("\n--- LinearSVC ---")
    print("Accuracy :", accuracy_score(y_test, y_test_linear))
    print(classification_report(y_test, y_test_linear))
    print(confusion_matrix(y_test, y_test_linear))

    # RBF
    y_test_rbf = best_rbf.predict(X_test)

    print("\n--- SVM RBF ---")
    print("Accuracy :", accuracy_score(y_test, y_test_rbf))
    print(classification_report(y_test, y_test_rbf))
    print(confusion_matrix(y_test, y_test_rbf))


if __name__ == "__main__":
    main()
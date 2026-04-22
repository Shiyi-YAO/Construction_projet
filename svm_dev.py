import argparse
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pretraitement import X, y

def main():
    parser = argparse.ArgumentParser(description="Classification avec LinearSVC + Dev set")

    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Proportion du jeu de test")
    parser.add_argument("--max-iter", type=int, default=50000,
                        help="Nombre maximal d'itérations")
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="none",
                        help="Poids des classes")

    args = parser.parse_args()

    # 1. Split TEST (jamais touché avant la fin)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=42,
        stratify=y
    )

    # 2. Split TRAIN / DEV
    X_train, X_dev, y_train, y_dev = train_test_split(
        X_temp, y_temp,
        test_size=0.25,  # 0.25 de 0.8 = 0.2    
        random_state=42,
        stratify=y_temp
    )

    # 3. class_weight
    if args.class_weight == "balanced":
        class_weight_value = "balanced"
    else:
        class_weight_value = None

    print("Paramètres utilisés :")
    print("test_size =", args.test_size)
    print("max_iter =", args.max_iter)
    print("class_weight =", class_weight_value)

    # 4. GridSearch sur TRAIN uniquement
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

    print("\nMeilleurs paramètres trouvés (sur TRAIN) :")
    print(grid.best_params_)

    # 5. Évaluation sur DEV
    y_dev_pred = grid.predict(X_dev)

    print("\nRésultats sur DEV :")
    print("Accuracy :", accuracy_score(y_dev, y_dev_pred))
    print("\nClassification report (DEV) :")
    print(classification_report(y_dev, y_dev_pred))

    # 6. Réentraîner sur TRAIN + DEV
    best_model = grid.best_estimator_

    X_final = np.vstack((X_train, X_dev))
    y_final = np.hstack((y_train, y_dev))

    best_model.fit(X_final, y_final)

    # 7. Évaluation finale sur TEST (une seule fois)
    y_test_pred = best_model.predict(X_test)

    print("\n==============================")
    print("Résultats finaux sur TEST :")
    print("==============================")

    print("Accuracy :", accuracy_score(y_test, y_test_pred))

    print("\nClassification report (TEST) :")
    print(classification_report(y_test, y_test_pred))

    print("\nConfusion matrix :")
    print(confusion_matrix(y_test, y_test_pred))


if __name__ == "__main__":
    main()
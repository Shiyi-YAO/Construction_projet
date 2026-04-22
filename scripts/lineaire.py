import argparse
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pretraitement import X, y

def main():
    parser = argparse.ArgumentParser(description="Classification linéaire avec LinearSVC")
    # Pourcentage de test
    parser.add_argument("--test-size", type=float, default=0.2, help="Proportion du jeu de test")
    # Validation croisée n-folds
    parser.add_argument("--n-splits", type=int, default=5, help="Nombre de folds pour la validation croisée")
    # Nombre max d'itérations
    parser.add_argument("--max-iter", type=int, default=50000, help="Nombre maximal d'itérations")
    # Régularisation (c)
    parser.add_argument("--C", type=float, default=1.0, help="Paramètre de régularisation")
    # Poids des classes
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="none",
                        help="Poids des classes")
    args = parser.parse_args()

    # 1. Séparer d'abord un jeu de test indépendant
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=42,
        stratify=y
    )

    # 2. Définir la validation croisée sur le train seulement
    cv = StratifiedKFold(
        n_splits=args.n_splits,
        shuffle=True,
        random_state=42
    )

    # 3. Définir le poids des classes
    if args.class_weight == "balanced":
        class_weight_value = "balanced"
    else:
        class_weight_value = None

    # 4. Créer le modèle
    clf = LinearSVC(
        C=args.C,
        max_iter=args.max_iter,
        random_state=42,
        class_weight=class_weight_value
    )

    # 5. Validation croisée uniquement sur le train
    cv_scores = cross_validate(
        clf,
        X_train,
        y_train,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "precision_macro": "precision_macro",
            "recall_macro": "recall_macro",
            "f1_macro": "f1_macro"
        },
        return_train_score=False
    )

    print("Paramètres utilisés :")
    print("test_size =", args.test_size)
    print("n_splits =", args.n_splits)
    print("max_iter =", args.max_iter)
    print("C =", args.C)
    print("class_weight =", class_weight_value)

    print("\nRésultats de la validation croisée sur le train :")
    print("Accuracy moyen :", cv_scores["test_accuracy"].mean())
    print("Precision macro moyenne :", cv_scores["test_precision_macro"].mean())
    print("Recall macro moyen :", cv_scores["test_recall_macro"].mean())
    print("F1 macro moyen :", cv_scores["test_f1_macro"].mean())

    print("\nScores par fold :")
    print("Accuracy :", cv_scores["test_accuracy"])
    print("Precision macro :", cv_scores["test_precision_macro"])
    print("Recall macro :", cv_scores["test_recall_macro"])
    print("F1 macro :", cv_scores["test_f1_macro"])

    # 6. Réentraîner le modèle sur tout le train
    clf.fit(X_train, y_train)

    # 7. Évaluer une seule fois sur le test
    y_pred = clf.predict(X_test)

    print("\nRésultats finaux sur le test :")
    print("Accuracy :", accuracy_score(y_test, y_pred))

    print("\nClassification report :")
    print(classification_report(y_test, y_pred))

    print("\nConfusion matrice :")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    main()

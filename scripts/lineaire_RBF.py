import argparse
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from pretraitement import X, y


def main():

    parser = argparse.ArgumentParser(description="Classification SVM linéaire et SVM à noyau RBF")

    # proportion du jeu de test
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Proportion du jeu de test")

    # nombre de folds pour la validation croisée
    parser.add_argument("--n-splits", type=int, default=5,
                        help="Nombre de folds pour la validation croisée")

    # nombre maximal d'itérations pour LinearSVC
    parser.add_argument("--max-iter", type=int, default=50000,
                        help="Nombre maximal d'itérations pour LinearSVC")

    # paramètre de régularisation C
    parser.add_argument("--C", type=float, default=1.0,
                        help="Paramètre de régularisation (C)")

    # gestion du déséquilibre des classes
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="none",
                        help="Gestion du déséquilibre des classes")

    args = parser.parse_args()

    # 1. Séparation du jeu de données en train / test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=42,
        stratify=y  # permet de conserver la distribution des classes
    )

    # 2. Définition de la validation croisée stratifiée
    cv = StratifiedKFold(
        n_splits=args.n_splits,
        shuffle=True,
        random_state=42
    )

    # définition du poids des classes
    class_weight_value = "balanced" if args.class_weight == "balanced" else None

    # 3. Modèle SVM linéaire
    clf_linear = LinearSVC(
        C=args.C,
        max_iter=args.max_iter,
        random_state=42,
        class_weight=class_weight_value
    )

    # 4. Modèle SVM à noyau RBF (non linéaire)
    clf_rbf = SVC(
        kernel="rbf",  # noyau gaussien
        C=args.C,
        gamma="scale",
        random_state=42,
        class_weight=class_weight_value
    )

    print("Paramètres utilisés :")
    print("test_size =", args.test_size)
    print("n_splits =", args.n_splits)
    print("C =", args.C)
    print("class_weight =", class_weight_value)

    # 5. Validation croisée - modèle linéaire
    cv_scores_linear = cross_validate(
        clf_linear,
        X_train,
        y_train,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "precision_macro": "precision_macro",
            "recall_macro": "recall_macro",
            "f1_macro": "f1_macro"
        }
    )

    print("\n================ MODELE LINEAIRE ================")
    print("Accuracy moyenne :", cv_scores_linear["test_accuracy"].mean())
    print("F1 macro moyen :", cv_scores_linear["test_f1_macro"].mean())

    # 6. Validation croisée - modèle RBF
    cv_scores_rbf = cross_validate(
        clf_rbf,
        X_train,
        y_train,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "precision_macro": "precision_macro",
            "recall_macro": "recall_macro",
            "f1_macro": "f1_macro"
        }
    )

    print("\n================ MODELE RBF ================")
    print("Accuracy moyenne :", cv_scores_rbf["test_accuracy"].mean())
    print("F1 macro moyen :", cv_scores_rbf["test_f1_macro"].mean())

    # 7. Entraînement final + test - modèle linéaire
    clf_linear.fit(X_train, y_train)
    y_pred_linear = clf_linear.predict(X_test)

    print("\n================ TEST LINEAIRE ================")
    print("Accuracy :", accuracy_score(y_test, y_pred_linear))
    print(classification_report(y_test, y_pred_linear))
    print(confusion_matrix(y_test, y_pred_linear))

    # 8. Entraînement final + test - modèle RBF
    clf_rbf.fit(X_train, y_train)
    y_pred_rbf = clf_rbf.predict(X_test)

    print("\n================ TEST RBF ================")
    print("Accuracy :", accuracy_score(y_test, y_pred_rbf))
    print(classification_report(y_test, y_pred_rbf))
    print(confusion_matrix(y_test, y_pred_rbf))


if __name__ == "__main__":
    main()

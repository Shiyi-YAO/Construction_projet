import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Lire le corpus
df = pd.read_csv('docs/dataset_erreurs_reprises.csv')

# Supprimer les ligne avec la valeur vide dans la colonne "TypeErreur1"
df = df.dropna(subset=['TypeErreur1']).copy()

# ---------------------------------------------------------
# Etape A : traitement des embeddings
# Dans le csv, les embeddings sont des chaînes de caractères(.dtypes)
# Il faut les convertir en vecteurs pour pouvoir les utiliser
# ---------------------------------------------------------

# Transformer une chaîne en tableau numpy
def parse_embedding(text):
    text = text.strip().strip("[]")
    values = [float(x.strip()) for x in text.split(",") if x.strip() != ""]
    return np.array(values, dtype=float)

# Trouver la dimension la plus fréquente dans une colonne
def detect_dim(series):
    dims = series.dropna().apply(
        lambda x: len([v for v in x.strip().strip("[]").split(",") if v.strip() != ""])
    )
    return int(dims.mode()[0])

# Détecter la dimension la plus fréquente pour chaque colonne
dim_reprise = detect_dim(df["Embedding_reprise"])
dim_ante = detect_dim(df["Embedding_antecedent"])

# Convertir un embedding et unifier la dimension
def convert_embedding(text, expected_dim):
    # Si c'est vide : on met un vecteur de zéros
    if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
        return np.zeros(expected_dim)

    vec = parse_embedding(text)

    # Si c'est trop court : on ajoute des zéros
    if vec.size < expected_dim:
        complement = np.zeros(expected_dim - vec.size)
        vec = np.concatenate([vec, complement])
    # Si c'est trop long : on coupe
    elif vec.size > expected_dim:
        vec = vec[:expected_dim]

    return vec

print("Traitement des embeddings...")

emb_reprise = np.stack(
    df["Embedding_reprise"].apply(lambda x: convert_embedding(x, dim_reprise)).values
)

emb_ante = np.stack(
    df["Embedding_antecedent"].apply(lambda x: convert_embedding(x, dim_ante)).values
)

print("Dimension fréquente de Embedding_reprise :", dim_reprise)
print("Dimension fréquente de Embedding_antecedent :", dim_ante)
print("Shape de emb_reprise :", emb_reprise.shape)
print("Shape de emb_ante :", emb_ante.shape)

# ---------------------------------------------------------
# Etape B : traitement des variables numériques
# On sélectionne les colonnes numériques et on remplit les valeurs manquantes par 0.
# ---------------------------------------------------------
num_cols = ['Distance_caracteres', 'Distance_mots', 'Distance_phrases', 'Similarite_reprise_antecedent']
num_features = df[num_cols].fillna(0).values

# Normalisation : z-score
scaler = StandardScaler()
num_features_scaled = scaler.fit_transform(num_features)

# ---------------------------------------------------------
# Etape C : traitement des variables catégorielles
# On remplace les valeurs manquantes par "Unknown", puis on applique un encodage one-hot.
# ---------------------------------------------------------
# Les colonnes concernant les catégories
cat_cols = ['TypeReprise', 'Type_pronom', 'Fonction_reprise', 'Fonction_antecedent']

# Remplir les valeurs manquants
df[cat_cols] = df[cat_cols].fillna("Unknown")

# One-hot encodage
cat_features_encoded = pd.get_dummies(df[cat_cols]).values

# ---------------------------------------------------------
# Etape D : concaténation des variables
# On regroupe toutes les variables dans une seule matrice X.
# ---------------------------------------------------------
# np.hstack pour concaténer des attribues
X = np.hstack([emb_reprise, emb_ante, num_features_scaled, cat_features_encoded])

# y - Variable cible
y = df['TypeErreur1'].values

print("--- Conversion terminée ---")
print("Shape de X :", X.shape)
print("Nombre d'exemples dans y :", len(y))

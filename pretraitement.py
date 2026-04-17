import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import StandardScaler

# 1. 读取数据
df = pd.read_csv('dataset_erreurs_reprises.xlsx - dataset_erreurs_reprises.csv')

# 2. 预处理：删除没有标签（TypeErreur1）的行（没标签模型没法学）
df = df.dropna(subset=['TypeErreur1']).copy()

# ---------------------------------------------------------
# 步骤 A: 处理语义型特征 (Embeddings)
# ---------------------------------------------------------
def safe_convert_embedding(text):
    # 如果缺失，返回 384 维全 0 向量 (根据 CamemBERT 常见维度)
    if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
        return np.zeros(384)
    try:
        # 将字符串 "[0.1, -0.2...]" 转换为真正的 numpy 数组
        return np.array(ast.literal_eval(text))
    except:
        return np.zeros(384)

print("正在处理 Embeddings...")
emb_reprise = np.stack(df['Embedding_reprise'].apply(safe_convert_embedding).values)
emb_ante = np.stack(df['Embedding_antecedent'].apply(safe_convert_embedding).values)

# ---------------------------------------------------------
# 步骤 B: 处理数值型特征 (Numerical Features)
# ---------------------------------------------------------
# 选择数值列并填充缺失值（用 0 或 -1）
num_cols = ['Distance_caracteres', 'Distance_mots', 'Distance_phrases', 'Similarite_reprise_antecedent']
num_features = df[num_cols].fillna(0).values

# 重要：对数值进行标准化（Scaling），让 SVM 表现更好
scaler = StandardScaler()
num_features_scaled = scaler.fit_transform(num_features)

# ---------------------------------------------------------
# 步骤 C: 处理类别型特征 (Categorical Features - One-Hot)
# ---------------------------------------------------------
# 选择需要转换的列
cat_cols = ['TypeReprise', 'Type_pronom', 'Fonction_reprise', 'Fonction_antecedent']

# 先把缺失值填为 "Unknown" 字符串，保证 One-Hot 正常运行
df[cat_cols] = df[cat_cols].fillna("Unknown")

# 使用 get_dummies 自动生成 One-Hot 编码矩阵
cat_features_encoded = pd.get_dummies(df[cat_cols]).values

# ---------------------------------------------------------
# 步骤 D: 最终合并 (The Matrix X)
# ---------------------------------------------------------
# 使用 np.hstack 将所有矩阵横向拼接成一个大矩阵 X
X = np.hstack([emb_reprise, emb_ante, num_features_scaled, cat_features_encoded])

# 准备目标标签 y
y = df['TypeErreur1'].values

print("--- 转换完成 ---")
print(f"最终矩阵 X 的形状 (样本数, 特征数): {X.shape}")
print(f"目标标签 y 的样本数: {len(y)}")# Coming soon

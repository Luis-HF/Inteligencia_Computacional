import numpy as np
import joblib
import time
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

# ==========================================
# 1. CARREGAMENTO DOS DADOS REAIS
# ==========================================
print("[*] Carregando dados de treino e validação...")
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_test  = np.load('X_test.npy')
y_test  = np.load('y_test.npy')
print(f"    X_train: {X_train.shape} | X_test: {X_test.shape}")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
print("[*] Features normalizadas com StandardScaler.")

pca = PCA(n_components=0.95, random_state=42)
X_train = pca.fit_transform(X_train)
X_test  = pca.transform(X_test)
print(f"[*] PCA aplicado: {X_train.shape[1]} componentes (95% da variância).")


# ==========================================
# 2. DEFINIÇÃO DOS 20 CLASSIFICADORES
# ==========================================
print("[*] Inicializando a grade de 20 classificadores...")

modelos = {
    "knn_k3": KNeighborsClassifier(n_neighbors=3),
    "knn_k5": KNeighborsClassifier(n_neighbors=5),
    "knn_k7": KNeighborsClassifier(n_neighbors=7),
    "knn_k9": KNeighborsClassifier(n_neighbors=9),

    "dt_depth5":     DecisionTreeClassifier(max_depth=5,    random_state=42),
    "dt_depth10":    DecisionTreeClassifier(max_depth=10,   random_state=42),
    "dt_depth20":    DecisionTreeClassifier(max_depth=20,   random_state=42),
    "dt_sem_limite": DecisionTreeClassifier(max_depth=None, random_state=42),

    "svm_linear_C01": CalibratedClassifierCV(estimator=SVC(kernel='linear', C=0.1,    random_state=42), ensemble=False),
    "svm_linear_C1":  CalibratedClassifierCV(estimator=SVC(kernel='linear', C=1.0,    random_state=42), ensemble=False),
    "svm_rbf_C1":     CalibratedClassifierCV(estimator=SVC(kernel='rbf',    C=1.0,    random_state=42), ensemble=False),
    "svm_poly":       CalibratedClassifierCV(estimator=SVC(kernel='poly',   degree=3, random_state=42), ensemble=False),

    "rf_10_trees":  RandomForestClassifier(n_estimators=10,  random_state=42),
    "rf_50_trees":  RandomForestClassifier(n_estimators=50,  random_state=42),
    "rf_100_trees": RandomForestClassifier(n_estimators=100, random_state=42),
    "rf_200_trees": RandomForestClassifier(n_estimators=200, random_state=42),

    "mlp_simples":  MLPClassifier(hidden_layer_sizes=(50,),             max_iter=1000, random_state=42),
    "mlp_profundo": MLPClassifier(hidden_layer_sizes=(100, 50),         max_iter=1000, random_state=42),
    "mlp_largo":    MLPClassifier(hidden_layer_sizes=(200,),            max_iter=1000, random_state=42),
    "mlp_tanh":     MLPClassifier(hidden_layer_sizes=(100,), activation='tanh', max_iter=1000, random_state=42),
}


# ==========================================
# 3. TREINAMENTO DOS 20 CLASSIFICADORES BASE
# ==========================================
print(f"[*] Iniciando o treinamento de {len(modelos)} modelos.")

matriz_probabilidades = {}
tempo_inicio_total = time.time()

for nome, modelo in modelos.items():
    print(f"    -> Treinando {nome}...", end=" ")
    inicio_modelo = time.time()
    modelo.fit(X_train, y_train)
    matriz_probabilidades[nome] = modelo.predict_proba(X_test)
    print(f"Concluído em {time.time() - inicio_modelo:.2f} segundos.")

tempo_total = time.time() - tempo_inicio_total
print(f"\n[*] Todos os {len(modelos)} modelos treinados em {tempo_total:.2f} segundos!")


# ==========================================
# 4. STACKING — META-CLASSIFICADOR
# ==========================================
print("\n[*] Gerando meta-features via stacking (5-fold)...")
print("    (isso pode levar alguns minutos devido aos SVMs)\n")

n_classes    = len(np.unique(y_train))
skf          = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
meta_X_train = np.zeros((len(X_train), len(modelos) * n_classes))

for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    print(f"    Fold {fold_idx}/5...", end=" ", flush=True)
    inicio_fold = time.time()
    X_f_tr, X_f_val = X_train[tr_idx], X_train[val_idx]
    y_f_tr = y_train[tr_idx]

    for i, (nome, modelo_base) in enumerate(modelos.items()):
        modelo_fold = clone(modelo_base)
        modelo_fold.fit(X_f_tr, y_f_tr)
        probs = modelo_fold.predict_proba(X_f_val)
        meta_X_train[val_idx, i * n_classes:(i + 1) * n_classes] = probs

    print(f"Concluído em {time.time() - inicio_fold:.1f}s")

meta_X_test = np.hstack(list(matriz_probabilidades.values()))

print("\n[*] Treinando meta-classificador (LogisticRegression)...")
meta_clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
meta_clf.fit(meta_X_train, y_train)
probs_stacking = meta_clf.predict_proba(meta_X_test)
print("[*] Meta-classificador treinado.")


# ==========================================
# 5. EXPORTAÇÃO
# ==========================================
pacote_entrega = {
    "probabilidades": matriz_probabilidades,
    "probs_stacking": probs_stacking,
    "y_test_real":    y_test,
}

joblib.dump(pacote_entrega, 'previsoes_20_modelos.pkl')
print(f"\n[*] Arquivo 'previsoes_20_modelos.pkl' salvo com sucesso na raiz do projeto.")

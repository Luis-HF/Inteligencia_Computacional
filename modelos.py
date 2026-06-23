import numpy as np
import joblib
import time
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier

# ==========================================
# 1. MÓDULO DE TESTE (Apague quando a Pessoa A entregar os dados)
# ==========================================
print("[*] Gerando dados fictícios para teste do pipeline...")
# Simulando 10 classes (ex: 10 personagens dos Simpsons)
X_train, y_train = make_classification(n_samples=1000, n_features=64, n_informative=20, n_classes=10, random_state=42)
X_test, y_test = make_classification(n_samples=300, n_features=64, n_informative=20, n_classes=10, random_state=99)

# Quando a Pessoa A entregar, você usará algo como:
# X_train, y_train = np.load('X_train.npy'), np.load('y_train.npy')
# X_test, y_test = np.load('X_test.npy'), np.load('y_test.npy')


# ==========================================
# 2. DEFINIÇÃO DOS 20 CLASSIFICADORES
# ==========================================
print("[*] Inicializando a grade de 20 classificadores...")

modelos = {
    "knn_k3": KNeighborsClassifier(n_neighbors=3),
    "knn_k5": KNeighborsClassifier(n_neighbors=5),
    "knn_k7": KNeighborsClassifier(n_neighbors=7),
    "knn_k9": KNeighborsClassifier(n_neighbors=9),
    
    "dt_depth5": DecisionTreeClassifier(max_depth=5, random_state=42),
    "dt_depth10": DecisionTreeClassifier(max_depth=10, random_state=42),
    "dt_depth20": DecisionTreeClassifier(max_depth=20, random_state=42),
    "dt_sem_limite": DecisionTreeClassifier(max_depth=None, random_state=42),
    
    "svm_linear_C01": CalibratedClassifierCV(estimator=SVC(kernel='linear', C=0.1, random_state=42), ensemble=False),
    "svm_linear_C1": CalibratedClassifierCV(estimator=SVC(kernel='linear', C=1.0, random_state=42), ensemble=False),
    "svm_rbf_C1": CalibratedClassifierCV(estimator=SVC(kernel='rbf', C=1.0, random_state=42), ensemble=False),
    "svm_poly": CalibratedClassifierCV(estimator=SVC(kernel='poly', degree=3, random_state=42), ensemble=False),
    
    "rf_10_trees": RandomForestClassifier(n_estimators=10, random_state=42),
    "rf_50_trees": RandomForestClassifier(n_estimators=50, random_state=42),
    "rf_100_trees": RandomForestClassifier(n_estimators=100, random_state=42),
    "rf_200_trees": RandomForestClassifier(n_estimators=200, random_state=42),
    
    "mlp_simples": MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42),
    "mlp_profundo": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42),
    "mlp_largo": MLPClassifier(hidden_layer_sizes=(200,), max_iter=1000, random_state=42),
    "mlp_tanh": MLPClassifier(hidden_layer_sizes=(100,), activation='tanh', max_iter=1000, random_state=42)
}


# ==========================================
# 3. PIPELINE DE TREINAMENTO E INFERÊNCIA
# ==========================================
print(f"[*] Iniciando o treinamento de {len(modelos)} modelos.")

# Dicionário que a Pessoa C vai receber
# Chave: Nome do modelo | Valor: Matriz de probabilidades de teste
matriz_probabilidades = {} 

tempo_inicio_total = time.time()

for nome, modelo in modelos.items():
    print(f"    -> Treinando {nome}...", end=" ")
    inicio_modelo = time.time()
    
    modelo.fit(X_train, y_train)
    
    probabilidades = modelo.predict_proba(X_test)
    matriz_probabilidades[nome] = probabilidades
    
    tempo_modelo = time.time() - inicio_modelo
    print(f"Concluído em {tempo_modelo:.2f} segundos.")

tempo_total = time.time() - tempo_inicio_total
print(f"\n[*] Todos os {len(modelos)} modelos treinados com sucesso em {tempo_total:.2f} segundos!")


# ==========================================
# 4. EXPORTAÇÃO
# ==========================================
pacote_entrega = {
    "probabilidades": matriz_probabilidades,
    "y_test_real": y_test
}

nome_arquivo = 'previsoes_20_modelos.pkl'
joblib.dump(pacote_entrega, nome_arquivo)
print(f"[*] Arquivo '{nome_arquivo}' salvo com sucesso na raiz do projeto.")
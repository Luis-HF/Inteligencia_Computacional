import numpy as np
import joblib
import time
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

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
# 2. DEFINIÇÃO DOS 20 CLASSIFICADORES (A exigência do edital)
# ==========================================
print("[*] Inicializando a grade de 20 classificadores...")

# ATENÇÃO: No SVC, 'probability=True' é OBRIGATÓRIO, senão a Pessoa C não consegue fazer o Soft Voting.
modelos = {
    # 5 Variações de k-NN (mudando o valor de K)
    "knn_k1": KNeighborsClassifier(n_neighbors=1),
    "knn_k3": KNeighborsClassifier(n_neighbors=3),
    "knn_k5": KNeighborsClassifier(n_neighbors=5),
    "knn_k7": KNeighborsClassifier(n_neighbors=7),
    "knn_k9": KNeighborsClassifier(n_neighbors=9),
    
    # 5 Variações de Árvore de Decisão (mudando a profundidade máxima)
    "dt_depth3": DecisionTreeClassifier(max_depth=3, random_state=42),
    "dt_depth5": DecisionTreeClassifier(max_depth=5, random_state=42),
    "dt_depth10": DecisionTreeClassifier(max_depth=10, random_state=42),
    "dt_depth20": DecisionTreeClassifier(max_depth=20, random_state=42),
    "dt_sem_limite": DecisionTreeClassifier(max_depth=None, random_state=42),
    
    # 5 Variações de SVM (mudando Kernel e penalidade C)
    "svm_linear_C01": SVC(kernel='linear', C=0.1, probability=True, random_state=42),
    "svm_linear_C1": SVC(kernel='linear', C=1.0, probability=True, random_state=42),
    "svm_rbf_C1": SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
    "svm_rbf_C10": SVC(kernel='rbf', C=10.0, probability=True, random_state=42),
    "svm_poly": SVC(kernel='poly', degree=3, probability=True, random_state=42),
    
    # 5 Variações de Random Forest (mudando número de árvores)
    "rf_10_trees": RandomForestClassifier(n_estimators=10, random_state=42),
    "rf_50_trees": RandomForestClassifier(n_estimators=50, random_state=42),
    "rf_100_trees": RandomForestClassifier(n_estimators=100, random_state=42),
    "rf_200_trees": RandomForestClassifier(n_estimators=200, random_state=42),
    "rf_depth10": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
}


# ==========================================
# 3. PIPELINE DE TREINAMENTO E INFERÊNCIA
# ==========================================
print(f"[*] Iniciando o treinamento de {len(modelos)} modelos. Isso pode levar alguns minutos...")

# Dicionário que a Pessoa C vai receber
# Chave: Nome do modelo | Valor: Matriz de probabilidades de teste
matriz_probabilidades = {} 

tempo_inicio_total = time.time()

for nome, modelo in modelos.items():
    print(f"    -> Treinando {nome}...", end=" ")
    inicio_modelo = time.time()
    
    # Treina o modelo com os dados que a Pessoa A extraiu
    modelo.fit(X_train, y_train)
    
    # Extrai as PROBABILIDADES para o conjunto de teste (vital para a Pessoa C)
    probabilidades = modelo.predict_proba(X_test)
    matriz_probabilidades[nome] = probabilidades
    
    tempo_modelo = time.time() - inicio_modelo
    print(f"Concluído em {tempo_modelo:.2f} segundos.")

tempo_total = time.time() - tempo_inicio_total
print(f"\n[*] Todos os {len(modelos)} modelos treinados com sucesso em {tempo_total:.2f} segundos!")


# ==========================================
# 4. EXPORTAÇÃO PARA A PESSOA C
# ==========================================
# Vamos salvar as previsões e também o y_test verdadeiro para a Pessoa C gerar a Matriz de Confusão
pacote_entrega = {
    "probabilidades": matriz_probabilidades,
    "y_test_real": y_test
}

nome_arquivo = 'previsoes_20_modelos.pkl'
joblib.dump(pacote_entrega, nome_arquivo)
print(f"[*] Arquivo '{nome_arquivo}' salvo com sucesso na raiz do projeto. Entregue isso à Pessoa C.")
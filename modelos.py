import numpy as np
import joblib
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

X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_test  = np.load('X_test.npy')
y_test  = np.load('y_test.npy')

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

pca = PCA(n_components=0.95, random_state=42)
X_train = pca.fit_transform(X_train)
X_test  = pca.transform(X_test)

modelos = {
    "knn_k3": KNeighborsClassifier(n_neighbors=3),
    "knn_k5": KNeighborsClassifier(n_neighbors=5),
    "knn_k7": KNeighborsClassifier(n_neighbors=7),
    "knn_k9": KNeighborsClassifier(n_neighbors=9),
    "dt_depth5":     DecisionTreeClassifier(max_depth=5,    random_state=42),
    "dt_depth10":    DecisionTreeClassifier(max_depth=10,   random_state=42),
    "dt_depth20":    DecisionTreeClassifier(max_depth=20,   random_state=42),
    "dt_sem_limite": DecisionTreeClassifier(max_depth=None, random_state=42),
    "svm_linear_C01": CalibratedClassifierCV(estimator=SVC(kernel='linear', C=0.1, random_state=42), ensemble=False),
    "svm_linear_C1":  CalibratedClassifierCV(estimator=SVC(kernel='linear', C=1.0, random_state=42), ensemble=False),
    "svm_rbf_C1":     CalibratedClassifierCV(estimator=SVC(kernel='rbf',    C=1.0, random_state=42), ensemble=False),
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

probs_train_base = {}
probs_test_base = {}

print("[*] Treinando classificadores base...")
for nome, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    probs_train_base[nome] = modelo.predict_proba(X_train)
    probs_test_base[nome] = modelo.predict_proba(X_test)

print("[*] Executando Stacking com 10-folds...")
n_classes = len(np.unique(y_train))
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
meta_X_train = np.zeros((len(X_train), len(modelos) * n_classes))

for tr_idx, val_idx in skf.split(X_train, y_train):
    X_f_tr, X_f_val = X_train[tr_idx], X_train[val_idx]
    y_f_tr = y_train[tr_idx]
    
    for i, (nome, modelo_base) in enumerate(modelos.items()):
        m_fold = clone(modelo_base).fit(X_f_tr, y_f_tr)
        meta_X_train[val_idx, i * n_classes:(i + 1) * n_classes] = m_fold.predict_proba(X_f_val)

meta_X_test = np.hstack(list(probs_test_base.values()))

meta_clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
meta_clf.fit(meta_X_train, y_train)

probs_stacking_train = meta_clf.predict_proba(meta_X_train)
probs_stacking_test  = meta_clf.predict_proba(meta_X_test)

pacote_entrega = {
    "probs_train_base": probs_train_base,
    "probs_test_base": probs_test_base,
    "probs_stacking_train": probs_stacking_train,
    "probs_stacking_test": probs_stacking_test,
    "y_train_real": y_train,
    "y_test_real": y_test,
}
joblib.dump(pacote_entrega, 'previsoes_final.pkl')
print("[*] Modelos treinados e exportados.")
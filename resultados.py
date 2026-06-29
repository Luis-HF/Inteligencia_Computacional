import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

CLASSES = ["bart", "homer", "lisa", "maggie", "marge"]
pacote = joblib.load('previsoes_final.pkl')

y_train_real = pacote['y_train_real']
y_test_real  = pacote['y_test_real']

# Soft Voting
probs_train_media = np.mean(np.array(list(pacote['probs_train_base'].values())), axis=0)
probs_test_media  = np.mean(np.array(list(pacote['probs_test_base'].values())), axis=0)

y_pred_voting_train = np.argmax(probs_train_media, axis=1)
y_pred_voting_test  = np.argmax(probs_test_media, axis=1)

# Stacking
y_pred_stacking_train = np.argmax(pacote['probs_stacking_train'], axis=1)
y_pred_stacking_test  = np.argmax(pacote['probs_stacking_test'], axis=1)

def print_metricas(titulo, y_real, y_pred):
    acc = accuracy_score(y_real, y_pred) * 100
    f1 = f1_score(y_real, y_pred, average='weighted') * 100
    print(f"[{titulo}] Acurácia: {acc:.2f}% | F1-Score: {f1:.2f}%")

print("="*50 + "\n DESEMPENHO NO TREINO\n" + "="*50)
print_metricas("SOFT VOTING - Treino", y_train_real, y_pred_voting_train)
print_metricas("STACKING    - Treino", y_train_real, y_pred_stacking_train)

print("\n" + "="*50 + "\n DESEMPENHO NA VALIDAÇÃO / TESTE\n" + "="*50)
print_metricas("SOFT VOTING - Teste", y_test_real, y_pred_voting_test)
print_metricas("STACKING    - Teste", y_test_real, y_pred_stacking_test)

# Matriz de Confusão Percentual (Teste)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, y_pred, titulo in zip(
    axes,
    [y_pred_voting_test, y_pred_stacking_test],
    ["Soft Voting (Teste)", "Stacking (Teste)"]
):
    # normalize='true' gera a porcentagem por linha (classe real)
    cm_percent = confusion_matrix(y_test_real, y_pred, normalize='true') * 100
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_percent, display_labels=CLASSES)
    disp.plot(cmap='Blues', ax=ax, values_format='.1f')
    
    ax.set_title(f"Matriz de Confusão (%) — {titulo}")
    ax.set_xlabel("Classe Predita")
    ax.set_ylabel("Classe Real")

plt.tight_layout()
plt.savefig("matriz_confusao_percentual.png", dpi=300, bbox_inches='tight')
print("\n[*] Imagem 'matriz_confusao_percentual.png' salva com as porcentagens exigidas.")
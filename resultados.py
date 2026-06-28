import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

CLASSES = ["bart", "homer", "lisa", "maggie", "marge"]

# ==========================================
# 1. CARREGAMENTO DOS DADOS
# ==========================================
print("[*] Carregando as previsões...")
pacote = joblib.load('previsoes_20_modelos.pkl')

matriz_probs_dict = pacote['probabilidades']
probs_stacking    = pacote['probs_stacking']
y_test_real       = pacote['y_test_real']

print(f"[*] Dados carregados! {len(matriz_probs_dict)} modelos encontrados.")


# ==========================================
# 2. SOFT VOTING
# ==========================================
array_3d_probs  = np.array(list(matriz_probs_dict.values()))
probs_media     = np.mean(array_3d_probs, axis=0)
y_pred_voting   = np.argmax(probs_media, axis=1)

# ==========================================
# 3. STACKING
# ==========================================
y_pred_stacking = np.argmax(probs_stacking, axis=1)


# ==========================================
# 4. MÉTRICAS
# ==========================================
def print_metricas(titulo, y_real, y_pred):
    acuracia = accuracy_score(y_real, y_pred) * 100
    f1 = f1_score(y_real, y_pred, average='weighted') * 100
    print("\n" + "="*40)
    print(f" {titulo}")
    print("="*40)
    print(f"Acurácia Geral:      {acuracia:.2f}%")
    print(f"F1-Score (Weighted): {f1:.2f}%")
    print("="*40)

print_metricas("SOFT VOTING (média de probabilidades)", y_test_real, y_pred_voting)
print_metricas("STACKING   (meta-classificador LR)",   y_test_real, y_pred_stacking)


# ==========================================
# 5. MATRIZES DE CONFUSÃO
# ==========================================
print("\n[*] Gerando matrizes de confusão...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, y_pred, titulo in zip(
    axes,
    [y_pred_voting, y_pred_stacking],
    ["Soft Voting (20 Modelos)", "Stacking (Meta-clf LR)"]
):
    cm = confusion_matrix(y_test_real, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES).plot(
        cmap='Blues', ax=ax, values_format='d'
    )
    ax.set_title(f"Matriz de Confusão — {titulo}")
    ax.set_xlabel("Classe Predita")
    ax.set_ylabel("Classe Real")

plt.tight_layout()
nome_imagem = "matriz_confusao_final.png"
plt.savefig(nome_imagem, dpi=300, bbox_inches='tight')
print(f"[*] Imagem '{nome_imagem}' salva com sucesso. Pronta para o artigo SBC!")

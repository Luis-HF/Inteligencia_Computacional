import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# ==========================================
# 1. CARREGAMENTO DOS DADOS (O SEU ARQUIVO)
# ==========================================
print("[*] Carregando as previsões...")
pacote = joblib.load('previsoes_20_modelos.pkl')

matriz_probs_dict = pacote['probabilidades']
y_test_real = pacote['y_test_real']

print(f"[*] Dados carregados! {len(matriz_probs_dict)} modelos encontrados.")


# ==========================================
# 2. A FUSÃO DE CLASSIFICADORES (SOFT VOTING)
# ==========================================
print("[*] Aplicando a combinação estática (Soft Voting)...")

# Transforma o dicionário em uma lista de matrizes
lista_probabilidades = list(matriz_probs_dict.values())

# Converte a lista para um array 3D do NumPy. 
# Formato: (20 modelos, N_imagens_teste, N_classes)
array_3d_probs = np.array(lista_probabilidades)

# Tira a MÉDIA das probabilidades (achatando o eixo 0, que são os 20 modelos)
probs_media_final = np.mean(array_3d_probs, axis=0)

# A decisão final do sistema é a classe que ficou com a maior probabilidade após a média
y_pred_ensemble = np.argmax(probs_media_final, axis=1)


# ==========================================
# 3. CÁLCULO DAS MÉTRICAS EXIGIDAS NO EDITAL
# ==========================================
print("\n" + "="*40)
print(" RESULTADOS FINAIS DO SISTEMA (ENSEMBLE)")
print("="*40)

# Acurácia
acuracia = accuracy_score(y_test_real, y_pred_ensemble) * 100
print(f"Acurácia Geral: {acuracia:.2f}%")

# F1-Score (Usando 'weighted' porque a base dos Simpsons geralmente é desbalanceada)
f1 = f1_score(y_test_real, y_pred_ensemble, average='weighted') * 100
print(f"F1-Score (Weighted): {f1:.2f}%")


# ==========================================
# 4. GERAÇÃO DA MATRIZ DE CONFUSÃO PARA O ARTIGO
# ==========================================
print("\n[*] Gerando e salvando a Matriz de Confusão...")

# Calcula a matriz bruta
cm = confusion_matrix(y_test_real, y_pred_ensemble)

# Configura a plotagem visual (MUITO importante para colocar no relatório SBC)
fig, ax = plt.subplots(figsize=(10, 8)) # Tamanho ajustável dependendo de quantos personagens são
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

# Plota usando um mapa de cores elegante (Blues)
disp.plot(cmap='Blues', ax=ax, values_format='d')

plt.title("Matriz de Confusão - Ensemble (20 Modelos)")
plt.xlabel("Classe Predita pelo Sistema")
plt.ylabel("Classe Real (Gabarito)")

# Salva a imagem em alta resolução
nome_imagem = "matriz_confusao_final.png"
plt.savefig(nome_imagem, dpi=300, bbox_inches='tight')
print(f"[*] Imagem '{nome_imagem}' salva com sucesso. Pronta para o artigo SBC!")
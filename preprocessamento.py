import os
import re
import numpy as np

from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from skimage.feature import hog, local_binary_pattern
from skimage.color import rgb2hsv

# ==========================================
# CONFIGURAÇÕES
# ==========================================
IMG_SIZE  = (64, 64)   # Tamanho padrão para normalização de todas as imagens
GRID      = (3, 3)     # Zoneamento: grade de linhas x colunas
TRAIN_DIR = "Train"
VALID_DIR = "Valid"

CLASSES = {"bart": 0, "homer": 1, "lisa": 2, "maggie": 3, "marge": 4}


def extrair_label(nome_arquivo):
    nome = os.path.splitext(nome_arquivo)[0]
    label_str = re.sub(r'\d+', '', nome)
    return CLASSES.get(label_str, None)


def _crop_central(img, frac=0.9):
    """Crop central de frac% e redimensiona de volta ao tamanho original."""
    w, h = img.size
    dx, dy = int(w * (1 - frac) / 2), int(h * (1 - frac) / 2)
    return img.crop((dx, dy, w - dx, h - dy)).resize((w, h), Image.LANCZOS)


def augmentar(img):
    """Retorna 11 variações determinísticas de uma imagem PIL."""
    flip = ImageOps.mirror(img)
    return [
        img,                                               # 1. original
        flip,                                              # 2. flip horizontal
        img.rotate(10,  fillcolor=(128, 128, 128)),        # 3. rotação +10°
        img.rotate(-10, fillcolor=(128, 128, 128)),        # 4. rotação -10°
        img.rotate(20,  fillcolor=(128, 128, 128)),        # 5. rotação +20°
        img.rotate(-20, fillcolor=(128, 128, 128)),        # 6. rotação -20°
        ImageEnhance.Brightness(img).enhance(1.3),         # 7. brilho +30%
        ImageEnhance.Brightness(img).enhance(0.7),         # 8. brilho -30%
        ImageEnhance.Contrast(img).enhance(1.4),           # 9. contraste +40%
        ImageEnhance.Contrast(img).enhance(0.6),           # 10. contraste -40%
        _crop_central(img, frac=0.88),                     # 11. crop central 88%
    ]


def features_zona(img_array_zona, pixels_per_cell=(8, 8)):
    """HOG + histograma HSV + LBP para um patch (zona) da imagem."""
    # HOG — captura contornos e formas
    features_hog = hog(
        img_array_zona,
        orientations=8,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=(2, 2),
        channel_axis=-1
    )

    # Histograma HSV — captura paleta de cores característica
    img_hsv = rgb2hsv(img_array_zona)
    hist_h, _ = np.histogram(img_hsv[:, :, 0], bins=16, range=(0, 1))
    hist_s, _ = np.histogram(img_hsv[:, :, 1], bins=8,  range=(0, 1))
    hist_v, _ = np.histogram(img_hsv[:, :, 2], bins=8,  range=(0, 1))
    features_cor = np.concatenate([hist_h, hist_s, hist_v]).astype(np.float64)
    features_cor /= (features_cor.sum() + 1e-6)

    # LBP — captura padrões de textura (cabelo, pele, roupa)
    gray = np.dot(img_array_zona, [0.299, 0.587, 0.114]).astype(np.uint8)
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    hist_lbp, _ = np.histogram(lbp, bins=10, range=(0, 10))
    features_lbp = hist_lbp.astype(np.float64)
    features_lbp /= (features_lbp.sum() + 1e-6)

    return np.concatenate([features_hog, features_cor, features_lbp])


def extrair_features(img_rgb):
    """Features globais + features por zona (zoneamento em grade GRID)."""
    img_array = np.array(img_rgb)
    h, w = img_array.shape[:2]

    # Global: pixels_per_cell=(8,8) — imagem inteira tem espaço suficiente
    features_global = features_zona(img_array, pixels_per_cell=(8, 8))

    linhas, colunas = GRID
    alt_zona  = h // linhas
    larg_zona = w // colunas

    features_zonas = []
    for r in range(linhas):
        for c in range(colunas):
            y0, y1 = r * alt_zona, (r + 1) * alt_zona
            x0, x1 = c * larg_zona, (c + 1) * larg_zona
            features_zonas.append(features_zona(img_array[y0:y1, x0:x1], pixels_per_cell=(8, 8)))

    return np.concatenate([features_global] + features_zonas)


def carregar_diretorio(diretorio, aplicar_augmentation=False):
    X, y = [], []
    arquivos = sorted(os.listdir(diretorio))
    total = len(arquivos)

    for i, arquivo in enumerate(arquivos, 1):
        label = extrair_label(arquivo)
        if label is None:
            print(f"  [!] Label desconhecido para '{arquivo}', pulando.")
            continue

        caminho = os.path.join(diretorio, arquivo)
        try:
            img = Image.open(caminho).convert("RGB")
            img = img.resize(IMG_SIZE, Image.LANCZOS)

            variacoes = augmentar(img) if aplicar_augmentation else [img]
            for variacao in variacoes:
                X.append(extrair_features(variacao))
                y.append(label)
        except Exception as e:
            print(f"  [!] Erro ao processar '{arquivo}': {e}")

        if i % 50 == 0 or i == total:
            print(f"  {i}/{total} imagens processadas...")

    return np.array(X), np.array(y)


if __name__ == "__main__":
    print(f"[*] Tamanho padrão das imagens: {IMG_SIZE[0]}x{IMG_SIZE[1]} pixels")
    print(f"[*] Zoneamento: grade {GRID[0]}x{GRID[1]} = {GRID[0]*GRID[1]} zonas")
    print(f"[*] Classes: {CLASSES}")

    print(f"\n[*] Processando imagens de TREINO ({TRAIN_DIR}) com data augmentation (11x)...")
    X_train, y_train = carregar_diretorio(TRAIN_DIR, aplicar_augmentation=True)
    print(f"    Shape X_train: {X_train.shape} | y_train: {y_train.shape}")

    print(f"\n[*] Processando imagens de VALIDAÇÃO ({VALID_DIR})...")
    X_test, y_test = carregar_diretorio(VALID_DIR, aplicar_augmentation=False)
    print(f"    Shape X_test:  {X_test.shape} | y_test:  {y_test.shape}")

    np.save("X_train.npy", X_train)
    np.save("y_train.npy", y_train)
    np.save("X_test.npy",  X_test)
    np.save("y_test.npy",  y_test)

    print("\n[*] Arquivos salvos: X_train.npy, y_train.npy, X_test.npy, y_test.npy")
    print(f"[*] Dimensão de cada vetor de features: {X_train.shape[1]}")

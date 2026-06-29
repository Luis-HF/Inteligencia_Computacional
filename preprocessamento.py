import os
import re
import numpy as np

from PIL import Image, ImageOps, ImageEnhance
from skimage.feature import hog, local_binary_pattern
from skimage.color import rgb2hsv

IMG_SIZE  = (64, 64)
GRID      = (3, 3)
TRAIN_DIR = "Train"
VALID_DIR = "Valid"
CLASSES   = {"bart": 0, "homer": 1, "lisa": 2, "maggie": 3, "marge": 4}

def extrair_label(nome_arquivo):
    nome = os.path.splitext(nome_arquivo)[0]
    label_str = re.sub(r'\d+', '', nome)
    return CLASSES.get(label_str, None)

def _crop_central(img, frac=0.9):
    w, h = img.size
    dx, dy = int(w * (1 - frac) / 2), int(h * (1 - frac) / 2)
    return img.crop((dx, dy, w - dx, h - dy)).resize((w, h), Image.LANCZOS)

def augmentar(img):
    flip = ImageOps.mirror(img)
    return [
        img, flip,
        img.rotate(10, fillcolor=(128, 128, 128)), img.rotate(-10, fillcolor=(128, 128, 128)),
        img.rotate(20, fillcolor=(128, 128, 128)), img.rotate(-20, fillcolor=(128, 128, 128)),
        ImageEnhance.Brightness(img).enhance(1.3), ImageEnhance.Brightness(img).enhance(0.7),
        ImageEnhance.Contrast(img).enhance(1.4), ImageEnhance.Contrast(img).enhance(0.6),
        _crop_central(img, frac=0.88),
    ]

def features_zona(img_array_zona, pixels_per_cell=(8, 8)):
    features_hog = hog(img_array_zona, orientations=8, pixels_per_cell=pixels_per_cell, cells_per_block=(2, 2), channel_axis=-1)
    
    img_hsv = rgb2hsv(img_array_zona)
    hist_h, _ = np.histogram(img_hsv[:, :, 0], bins=16, range=(0, 1))
    hist_s, _ = np.histogram(img_hsv[:, :, 1], bins=8,  range=(0, 1))
    hist_v, _ = np.histogram(img_hsv[:, :, 2], bins=8,  range=(0, 1))
    features_cor = np.concatenate([hist_h, hist_s, hist_v]).astype(np.float64)
    features_cor /= (features_cor.sum() + 1e-6)

    gray = np.dot(img_array_zona, [0.299, 0.587, 0.114]).astype(np.uint8)
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    hist_lbp, _ = np.histogram(lbp, bins=10, range=(0, 10))
    features_lbp = hist_lbp.astype(np.float64)
    features_lbp /= (features_lbp.sum() + 1e-6)

    return np.concatenate([features_hog, features_cor, features_lbp])

def extrair_features(img_rgb):
    img_array = np.array(img_rgb)
    h, w = img_array.shape[:2]
    features_global = features_zona(img_array, pixels_per_cell=(8, 8))

    linhas, colunas = GRID
    alt_zona, larg_zona = h // linhas, w // colunas
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
    for arquivo in arquivos:
        label = extrair_label(arquivo)
        if label is None: continue
        caminho = os.path.join(diretorio, arquivo)
        try:
            img = Image.open(caminho).convert("RGB").resize(IMG_SIZE, Image.LANCZOS)
            variacoes = augmentar(img) if aplicar_augmentation else [img]
            for v in variacoes:
                X.append(extrair_features(v))
                y.append(label)
        except: pass
    return np.array(X), np.array(y)

if __name__ == "__main__":
    print("[*] Extraindo TREINO...")
    X_train, y_train = carregar_diretorio(TRAIN_DIR, aplicar_augmentation=True)
    print("[*] Extraindo VALIDAÇÃO...")
    X_test, y_test = carregar_diretorio(VALID_DIR, aplicar_augmentation=False)

    np.save("X_train.npy", X_train)
    np.save("y_train.npy", y_train)
    np.save("X_test.npy",  X_test)
    np.save("y_test.npy",  y_test)
    print("[*] Concluído.")
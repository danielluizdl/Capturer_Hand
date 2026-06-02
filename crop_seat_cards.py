"""
Recorta cada carta individualmente dos pares capturados em 'mao seat',
salvando na posicao correta (left / right) em templates/baralho_seats/.

Nomenclatura dos arquivos fonte: {cartaEsquerda}{cartaDireita}.PNG
  ex.: 9c2h  esquerda=9c  direita=2h

Algoritmo de split:
  1. Usa o TERCO SUPERIOR da imagem e MEDIANA por coluna
     (ignora simbolos no centro/base da carta que confundem a deteccao)
  2. Categoriza a cor de fundo da carta esquerda e direita
     (verde=paus, azul=ouros, vermelho=copas, escuro=espadas)
  3. Se MESMO naipe: split fixo em 40% da largura
  4. Se naipes DIFERENTES: varre da esquerda buscando 4+ colunas
     consecutivas que pertencam a carta direita (por distancia de cor)
  5. Fallback: 40% se nenhuma transicao for encontrada
"""

import os
import glob
import numpy as np
from PIL import Image
from collections import Counter

# -- caminhos ---------------------------------------------------------------
SOURCE_FOLDERS = glob.glob(r'C:\Users\danie\Desktop\*seat*')
if not SOURCE_FOLDERS:
    raise RuntimeError("Pasta 'mao seat' nao encontrada no Desktop")
SOURCE_DIR = SOURCE_FOLDERS[0]

TEMPLATES_DIR = r'C:\Users\danie\Documents\Projetos\Capturer_Hand\templates\baralho_seats'
LEFT_DIR  = os.path.join(TEMPLATES_DIR, 'left')
RIGHT_DIR = os.path.join(TEMPLATES_DIR, 'right')
os.makedirs(LEFT_DIR,  exist_ok=True)
os.makedirs(RIGHT_DIR, exist_ok=True)

RANKS = set('23456789tjqka')
SUITS = set('cdhs')

# -- helpers ----------------------------------------------------------------
def parse_pair(name):
    n = name.lower()
    if len(n) != 4:
        return None
    r1, s1, r2, s2 = n
    if r1 not in RANKS or s1 not in SUITS or r2 not in RANKS or s2 not in SUITS:
        return None
    return f'{r1}{s1}', f'{r2}{s2}'

def color_category(rgb):
    r, g, b = rgb
    if g > r + 12 and g > b + 12:
        return 'green'   # paus
    if b > r + 12 and b > g + 12:
        return 'blue'    # ouros
    if r > g + 12 and r > b + 12:
        return 'red'     # copas
    return 'dark'        # espadas

def _is_felt(mean_px):
    """
    Retorna True se a cor media de uma linha/coluna for feltro da mesa.
    Feltro: marrom-quente (R-B > 5), nao muito brilhante, canais proximos.
    Distingue do fundo escuro de espadas (R~G~B, R-B~0).
    """
    r, g, b = float(mean_px[0]), float(mean_px[1]), float(mean_px[2])
    return (max(r, g, b) < 130 and
            max(r, g, b) - min(r, g, b) < 50 and
            r - b > 5)

def _trim_felt(img, left=True, right=True, top=True, bottom=True):
    """Remove bordas de feltro da mesa. Nao trimma a borda interna
    (onde as duas cartas se encontram) para nao cortar a carta."""
    arr = np.array(img)
    h, w = arr.shape[:2]
    l, r_, t, b_ = 0, w, 0, h

    if left:
        for x in range(w):
            if _is_felt(arr[:, x, :].mean(axis=0)): l += 1
            else: break
    if right:
        for x in range(w - 1, -1, -1):
            if _is_felt(arr[:, x, :].mean(axis=0)): r_ -= 1
            else: break
    if top:
        for y in range(h):
            if _is_felt(arr[y, :, :].mean(axis=0)): t += 1
            else: break
    if bottom:
        for y in range(h - 1, -1, -1):
            if _is_felt(arr[y, :, :].mean(axis=0)): b_ -= 1
            else: break

    # Seguranca: nao trimmar se resultado for muito pequeno
    if r_ - l < 10 or b_ - t < 10:
        return img
    return img.crop((l, t, r_, b_))

def _bg_color(zone):
    """
    Cor de fundo de uma zona: mediana dos pixels nao-brilhantes.
    Numerais e simbolos das cartas sao quase sempre brancos/cremes
    (max canal > 180); o fundo e sempre mais escuro (vermelho, verde,
    azul ou escuro). Filtrar brilhantes elimina o ruido dos numerais.
    """
    px = zone.reshape(-1, 3).astype(float)
    mask = np.max(px, axis=1) < 180
    return np.median(px[mask], axis=0) if mask.sum() > 2 else np.median(px, axis=0)

def _col_bg(col_pixels):
    """Cor de fundo de uma coluna, filtrando pixels brilhantes."""
    px = col_pixels.astype(float)
    mask = np.max(px, axis=1) < 180
    return np.median(px[mask], axis=0) if mask.sum() > 0 else np.median(px, axis=0)

def find_split(img):
    """
    Retorna o x onde cortar: left = img[0:x], right = img[x:w].

    Algoritmo:
      1. Usa o terco superior da imagem (onde estao os indicadores da carta).
      2. Filtra pixels brilhantes (numerais brancos/cremes) para obter
         a cor real do fundo de cada lado.
      3. Se mesmo naipe/cor: split fixo em 40%.
      4. Se naipes diferentes: varre da esquerda buscando 4+ colunas
         consecutivas pertencentes a carta direita.
    """
    w, h = img.size
    arr = np.array(img.convert('RGB'))

    # Usa a imagem COMPLETA para lc/rc: mesmo que o terco superior seja
    # dominado pelo numeral branco (ex: '10'), as linhas inferiores
    # mostram o fundo puro da carta e dominam a mediana apos filtro.
    lc = _bg_color(arr[:, 5:max(6, int(w * 0.22)), :])
    rc = _bg_color(arr[:, int(w * 0.78):max(int(w * 0.78) + 1, w - 5), :])

    if color_category(lc) == color_category(rc):
        return int(w * 0.40)

    lo = int(w * 0.18)
    hi = int(w * 0.62)
    run = 0
    for x in range(lo, hi):
        col = _col_bg(arr[:, x, :])
        dist_l = np.linalg.norm(col - lc)
        dist_r = np.linalg.norm(col - rc)
        if dist_r < dist_l:
            run += 1
            if run >= 4:
                return x - 3
        else:
            run = 0

    return int(w * 0.40)

# -- processamento ----------------------------------------------------------
files = sorted(glob.glob(os.path.join(SOURCE_DIR, '*.PNG')))

saved_left  = []
saved_right = []
skipped     = []

for fpath in files:
    base = os.path.basename(fpath)
    stem = os.path.splitext(base)[0]

    parsed = parse_pair(stem)
    if parsed is None:
        skipped.append(base)
        continue

    left_card, right_card = parsed
    img = Image.open(fpath).convert('RGBA')
    w, h = img.size

    # Remove info de jogador (nome/BB) que aparece abaixo das cartas
    # em imagens capturadas com area maior que o necessario.
    # Cards cabem em ~66px; qualquer excesso e conteudo irrelevante.
    MAX_H = 68
    if h > MAX_H:
        img = img.crop((0, 0, w, MAX_H))
        h = MAX_H

    x = find_split(img)

    left_img  = _trim_felt(img.crop((0, 0, x, h)), left=True,  right=False, top=True, bottom=True)
    right_img = _trim_felt(img.crop((x, 0, w, h)), left=False, right=True,  top=True, bottom=True)

    left_name  = f'{left_card}.png'
    right_name = f'{right_card}.png'

    left_img.save(os.path.join(LEFT_DIR,  left_name))
    right_img.save(os.path.join(RIGHT_DIR, right_name))

    saved_left.append((left_card,  left_name, x, w))
    saved_right.append((right_card, right_name, x, w))

# -- relatorio --------------------------------------------------------------
print('=' * 60)
print('RECORTES SALVOS')
print('=' * 60)
print(f'  LEFT  ({len(saved_left)} arquivos)  -> {LEFT_DIR}')
print(f'  RIGHT ({len(saved_right)} arquivos) -> {RIGHT_DIR}')

if skipped:
    print(f'\nIgnorados (nome fora do padrao): {skipped}')

distinct_left  = sorted(set(c for c, *_ in saved_left))
distinct_right = sorted(set(c for c, *_ in saved_right))
print(f'\nCartas distintas LEFT  ({len(distinct_left)}): {", ".join(distinct_left)}')
print(f'Cartas distintas RIGHT ({len(distinct_right)}): {", ".join(distinct_right)}')

# split medio por posicao
avg_left_pct  = sum(x/w for _, _, x, w in saved_left)  / len(saved_left)  * 100
avg_right_pct = sum(x/w for _, _, x, w in saved_right) / len(saved_right) * 100
print(f'\nSplit medio: {avg_left_pct:.1f}% da largura')

print('\nConcluido.')

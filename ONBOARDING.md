# Capturer_Hand — Contexto Completo para Agente

## O que é este projeto

Pipeline de **captura automática de hand history** a partir de gravações OBS de sessões de poker online. O input é um vídeo `.mp4` gravado com OBS mostrando 4 mesas simultâneas do **WPT Global (Nexa Poker)** em resolução **1920×1080 @ 30fps**. O output deve ser um arquivo no formato **PokerStars Hand History** (`.txt`) com todas as mãos detectadas.

---

## Estrutura do vídeo de teste

- **Arquivo:** `video_cortado_1min.mp4` (75.6s, 2267 frames, não está no git — copie para a raiz)
- **Resolução:** 1920×1080 @ 30fps
- **Layout:** 4 janelas WPT Global em grid 2×2, cada uma 960×540 px
- **Gabarito:** `gabarito.txt` — 3 mãos reais em formato PokerStars para validação

### Mapeamento fixo das 4 mesas (válido para todo o vídeo)

```python
TABLE_REGIONS = [
    (0,    0,   960, 540),   # 0 = TL → HL3458  (NÃO está no gabarito)
    (960,  0,  1920, 540),   # 1 = TR → HL4017  ✓ gabarito mão 1
    (0,   540,  960, 1080),  # 2 = BL → HL2332  ✓ gabarito mão 3
    (960, 540, 1920, 1080),  # 3 = BR → HL3048  ✓ gabarito mão 2
]

# Altura útil por mesa (mesas de baixo têm taskbar Windows nos últimos ~38px)
TABLE_USEFUL_H = [540, 540, 502, 502]
```

---

## OCR disponível: RapidOCR

`rapidocr-onnxruntime` **já está instalado** e **confirmado funcionando**. Não precisa de Tesseract.

```python
from rapidocr_onnxruntime import RapidOCR
ocr = RapidOCR()
result, _ = ocr(img_bgr_numpy)
# result = [[[x1,y1],[x2,y1],[x2,y2],[x1,y2]], "texto", confidence], ...]
```

**Saídas confirmadas em testes reais no vídeo:**
- Title bar: `"HL4017-0.05/0.1/0.2(0.05)-NLHE"` (conf ~0.98)
- Pot: `"PoteTotal:21,5BB"` (conf ~0.96) — decimal com vírgula PT-BR
- Stack: `"24,40BB"` (conf ~0.97)
- Player name: `"easycall86"` (conf ~0.98)

**Regex úteis:**
```python
import re
table_id = re.search(r'(HL\d+)', text)
pot_bb   = re.search(r'PoteTotal\s*:?\s*([\d,\.]+)', text)  # "PoteTotal:21,5BB"
stack_bb = re.search(r'([\d,\.]+)\s*BB', text)
```

---

## Detectores visuais calibrados (sem OCR)

Arquivo de referência: `detectors_final.py` (na raiz do projeto).

### count_board_cards(crop_960x540) → int

Conta cartas do board por variância de pixel em 5 slots fixos.
Retorna apenas valores válidos em poker: **0, 3, 4, 5**. Valores 1 e 2 são descartados como ruído do logo NEXA POKER.

```python
SLOT_X          = [0.333, 0.393, 0.453, 0.513, 0.580]  # frac. de x=960
CARD_Y1_F       = 0.34
CARD_Y2_F       = 0.56
SLOT_THRESHOLDS = [20, 25, 33, 38, 25]  # calibrado empiricamente
```

**Performance medida:** 33ms/frame a 6fps, 23ms/frame a 10fps (sem OCR).

### has_action_buttons(crop) → (bool, float)

Detecta botões Desistir/Pagar/Aumentar na barra inferior (y > 88% da altura útil).
Retorna `(True, score)` quando é o turno do herói em alguma mesa.

### detect_pot_change(prev_crop, curr_crop) → (bool, float)

Compara ROI `[28%-36% y, 28%-72% x]` entre frames consecutivos. diff > 4.0 = mudança no pot.

---

## ROIs importantes dentro de cada crop 960×540

```
y=0–25    : title bar — "HL4017 - 0.05/0.1/0.2(0.05) - NLHE"
y=160–200 : pot text  — "Pote Total : XX,X BB"
y=205–295 : board cards (5 slots horizontais)
y=440–510 : herói (dLzinN) — nome + stack + hole cards
y=90%+    : action bar — "Desistir | Pagar X BB | Aumentar X BB"
```

---

## Gabarito — formato e campos

3 mãos em `gabarito.txt`, formato PokerStars Hand History:

| Campo         | Mão 1 (HL4017/TR) | Mão 2 (HL3048/BR) | Mão 3 (HL2332/BL) |
|---------------|-------------------|-------------------|-------------------|
| table_id      | HL4017            | HL3048            | HL2332            |
| button_seat   | 1                 | 3                 | 1                 |
| hero (dLzinN) | seat 1, $81.00    | seat 1, $20.70    | seat 1, $66.80    |
| hole_cards    | 9s 4s             | 2c 6s             | Ah Ad             |
| flop          | Jd 5s 2h          | Kd Ad 9d          | 5h Qs 8d          |
| turn          | 3d                | 8d                | Kh                |
| river         | 7s                | —                 | —                 |
| final_pot     | $2.15             | $11.38            | $4.71             |
| winner        | taymonkha         | Hamster813        | dLzinN            |

---

## Rubrica de score (por mão, 100 pts)

```python
SCORE_RUBRIC = {
    "table_id":        10,   # OCR do title bar → "HL4017" correto
    "board_cards":     30,   # 6 pts por carta correta (max 5 cartas = 30)
    "hole_cards":      20,   # 10 pts por carta do herói (2 cartas = 20)
    "street_sequence": 15,   # 5 pts por street correta (flop/turn/river)
    "final_pot":       15,   # dentro de 5% = 15, 10% = 10, 20% = 5
    "winner":          10,   # nome do vencedor correto
}
# 3 mãos × 100 pts = 300 pts total → score % = (acertos/300)*100
```

---

## Stack de dependências

```
Python 3.11
opencv-python >= 4.8
numpy >= 1.24
rapidocr-onnxruntime >= 1.3   ← já instalado, confirmado funcionando
```

---

## Arquivos existentes no repositório

| Arquivo                  | O que é                                              |
|--------------------------|------------------------------------------------------|
| `gabarito.txt`           | 3 mãos reais em formato PokerStars — ground truth    |
| `detectors_final.py`     | Detectores visuais calibrados (board, buttons, pot)  |
| `calibrate_felt.py`      | Script de calibração dos thresholds dos slots        |
| `benchmark_fps.py`       | Benchmark de performance por FPS                     |
| `detectors_v2.py`        | Versão anterior dos detectores (referência)          |
| `detectors_v3.py`        | Versão anterior dos detectores (referência)          |
| `diagnose_detectors.py`  | Script de diagnóstico visual das ROIs                |
| `requirements.txt`       | Dependências Python                                  |

> **Nota:** `video_cortado_1min.mp4` não está no git (muito grande). Baixe separadamente e coloque na raiz.

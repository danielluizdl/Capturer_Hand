# TASKS — Agente de Captura de Mãos de Poker

> **Leia o ONBOARDING.md antes de começar.**
> O vídeo `video_cortado_1min.mp4` deve estar na raiz do projeto.
> Execute `pip install -r requirements.txt` antes de qualquer script.

---

## Objetivo

Construir um pipeline que processa `video_cortado_1min.mp4` a **10fps** e extrai as 3 mãos descritas em `gabarito.txt`, atingindo o maior score possível na rubrica definida.

O agente deve **rodar em loop**: implementa → testa → vê o score → corrige → repete, até 100% ou esgotar tokens.

---

## TASK 1 — Gabarito Parser

**Arquivo:** `src/gabarito_parser.py`

**O que fazer:**
Implemente um parser do formato PokerStars Hand History para um dataclass `HandHistory`.

**Interface esperada:**
```python
from src.gabarito_parser import parse_gabarito, HandHistory

hands: list[HandHistory] = parse_gabarito("gabarito.txt")
# hands[0].table_id   == "HL4017"
# hands[0].board      == ["Jd","5s","2h","3d","7s"]
# hands[0].hole_cards == ["9s","4s"]
# hands[0].winner     == "taymonkha"
# hands[0].total_pot  == 2.15
```

**Campos do dataclass HandHistory:**
```python
@dataclass
class HandHistory:
    table_id:        str         # "HL4017"
    button_seat:     int         # 1
    players:         dict        # {seat_num: {"name": str, "chips": float}}
    hole_cards:      list[str]   # ["9s","4s"] — cartas do dLzinN
    board:           list[str]   # ["Jd","5s","2h","3d","7s"] — todas as comunitárias
    streets:         list[str]   # ["preflop","flop","turn","river"] — streets que ocorreram
    pot_by_street:   dict        # {"flop": 1.65, "turn": 1.65, "river": 2.15}
    total_pot:       float       # 2.15
    winner:          str         # "taymonkha"
    actions:         list[dict]  # [{"street","player","action","amount"}, ...]
```

**Critério de aceite:** `parse_gabarito("gabarito.txt")` retorna lista de 3 `HandHistory` sem exceção, com todos os campos preenchidos corretamente.

---

## TASK 2 — OCR Engine

**Arquivo:** `src/ocr_engine.py`

**O que fazer:**
Implemente wrapper sobre `rapidocr-onnxruntime` com funções específicas para o layout WPT Global.

**⚠️ RapidOCR já confirmado funcionando** (ver ONBOARDING.md). Não use Tesseract.

**Interface esperada:**
```python
from src.ocr_engine import ocr_title_bar, ocr_pot, ocr_stacks

ocr_title_bar(crop_960x540) -> str | None
# retorna "HL4017" ou None

ocr_pot(crop_960x540) -> float | None
# retorna 21.5 (converte vírgula → ponto) ou None

ocr_stacks(crop_960x540) -> dict[str, float]
# retorna {"easycall86": 24.4, "taymonkha": 23.0, ...}
```

**ROIs para cada função** (coordenadas em fração do crop 960×540):
- `ocr_title_bar` → `crop[0:25, 0:500]`
- `ocr_pot` → `crop[160:205, 260:700]`
- `ocr_stacks` → precisa varrer posições dos jogadores (ver ONBOARDING.md)

**Regex úteis (vírgula decimal PT-BR):**
```python
re.search(r'(HL\d+)', text)
re.search(r'PoteTotal\s*:?\s*([\d,\.]+)', text)
re.search(r'([\d,\.]+)\s*BB', text)  # converte vírgula para ponto
```

**Critério de aceite:**
- `ocr_title_bar` identifica corretamente os 4 IDs de mesa nos frames do vídeo
- `ocr_pot` extrai o valor com erro < 5% do valor real

---

## TASK 3 — Detectores Visuais

**Arquivo:** `src/detectors.py`

**O que fazer:**
Mova e limpe o código de `detectors_final.py` (raiz) para `src/detectors.py`. Ajuste para 10fps.

**Funções obrigatórias:**
```python
from src.detectors import crop_table, count_board_cards, has_action_buttons, detect_pot_change

crop_table(frame_1920x1080, table_id: int) -> np.ndarray  # 960x(502|540)
count_board_cards(crop) -> int   # retorna 0, 3, 4 ou 5
has_action_buttons(crop) -> tuple[bool, float]
detect_pot_change(prev, curr) -> tuple[bool, float]
```

**Parâmetros já calibrados** (copie do `detectors_final.py`):
```python
TABLE_REGIONS    = [(0,0,960,540),(960,0,1920,540),(0,540,960,1080),(960,540,1920,1080)]
TABLE_USEFUL_H   = [540, 540, 502, 502]
SLOT_X           = [0.333, 0.393, 0.453, 0.513, 0.580]
SLOT_THRESHOLDS  = [20, 25, 33, 38, 25]
```

**Critério de aceite:** `count_board_cards` detecta corretamente os tamanhos de board das 3 mesas do gabarito.

---

## TASK 4 — Video Pipeline

**Arquivo:** `src/video_pipeline.py`

**O que fazer:**
Pipeline principal que processa o vídeo a **10fps** e emite eventos por mesa.

**Interface esperada:**
```python
from src.video_pipeline import process_video, TableEvent

events: dict[int, list[TableEvent]] = process_video("video_cortado_1min.mp4", fps=10)
# events[1] = lista de TableEvent para mesa TR (HL4017)
# events[2] = lista de TableEvent para mesa BL (HL2332)
# events[3] = lista de TableEvent para mesa BR (HL3048)
```

**Dataclass TableEvent:**
```python
@dataclass
class TableEvent:
    timestamp:   float
    table_id:    str         # "HL4017" — extraído via OCR no primeiro frame
    event_type:  str         # "board_change" | "pot_change" | "action" | "new_hand"
    board_cards: int         # 0, 3, 4, 5
    pot_bb:      float | None
    action_btns: bool
```

**Lógica de eventos:**
- `new_hand`: board_cards vai de N > 0 para 0
- `board_change`: board_cards muda (0→3=flop, 3→4=turn, 4→5=river)
- `pot_change`: `detect_pot_change()` retorna True
- `action`: `has_action_buttons()` retorna True (nosso turno)

**Performance esperada:** < 30s para processar 75s de vídeo a 10fps (sem OCR contínuo).

**Critério de aceite:** Gera eventos coerentes — pelo menos 1 `board_change` para cada street das 3 mesas do gabarito.

---

## TASK 5 — Hand Builder

**Arquivo:** `src/hand_builder.py`

**O que fazer:**
Converte a lista de `TableEvent` em objetos `HandHistory` (mesmo formato do gabarito).

**Interface esperada:**
```python
from src.hand_builder import build_hands

detected: list[HandHistory] = build_hands(events, video_path="video_cortado_1min.mp4")
# detected[0].table_id  == "HL4017"
# detected[0].board     == ["Jd","5s","2h","3d","7s"]  (de OCR nos frames certos)
```

**Lógica:**
1. Agrupa eventos por `table_id`
2. Para cada `board_change`, volta ao frame e roda `ocr_title_bar()` (se table_id ainda não determinado) e tenta detectar as cartas do board visualmente
3. Para cada `new_hand`, registra fim da mão anterior e início da nova
4. Extrai hole cards do herói (dLzinN) quando visíveis no frame do início da mão
5. Usa OCR de pot nos eventos `pot_change` para preencher `pot_by_street`

**Critério de aceite:** `build_hands()` retorna pelo menos 3 `HandHistory` — um por mesa do gabarito — com `table_id`, `board`, e `total_pot` preenchidos.

---

## TASK 6 — Scorer

**Arquivo:** `src/scorer.py`

**O que fazer:**
Compare `HandHistory` extraído do vídeo contra o gabarito. Gere score detalhado.

**Interface esperada:**
```python
from src.scorer import score_hands, print_report

report = score_hands(detected: list[HandHistory], gabarito: list[HandHistory])
print_report(report)
# Saída esperada:
# ============================================
# SCORE TOTAL: 147/300 (49.0%)
# ============================================
# Mão HL4017 (TR):  62/100
#   table_id:        10/10  ✓
#   board_cards:     24/30  (4/5 corretas: Jd ✓  5s ✓  2h ✓  3d ✓  7s ✗)
#   hole_cards:       0/20  (não detectadas)
#   street_sequence: 15/15  ✓
#   final_pot:       13/15  (detectado: 2.30BB, esperado: 2.15BB)
#   winner:           0/10  (não detectado)
# ...
```

**Rubrica** (ver ONBOARDING.md para detalhes):
```python
RUBRIC = {"table_id":10, "board_cards":30, "hole_cards":20,
          "street_sequence":15, "final_pot":15, "winner":10}
```

**Critério de aceite:** `score_hands()` retorna um dict com score por campo por mão e score total em %.

---

## TASK 7 — Agent Loop

**Arquivo:** `agent_loop.py`

**O que fazer:**
Script principal que executa o pipeline, mostra o score, e **entra em loop de melhoria**.

```bash
python agent_loop.py
```

**Comportamento:**
1. Carrega gabarito
2. Processa o vídeo (Tasks 4+5)
3. Calcula score (Task 6)
4. Imprime relatório detalhado com o que errou
5. **Analisa os erros mais impactantes** (qual campo perdeu mais pontos)
6. **Faz um ajuste** (threshold, ROI, regex, lógica)
7. Salva o ajuste em `config.json` ou direto no código
8. Volta ao passo 2
9. Para quando score = 100% ou detectar que não há mais progresso em N iterações

**Saída por iteração:**
```
=== ITERAÇÃO 1 ===
Score: 49.0% (147/300)
Maior gap: board_cards HL4017 — perdeu 6 pts (1 carta errada)
Ajuste: recalibrando threshold slot 3 de 33 → 30
[re-processa]

=== ITERAÇÃO 2 ===
Score: 53.3% (160/300)
...
```

**Critério de aceite:** O loop roda sem travar e incrementa o score entre iterações (ou para com diagnóstico claro quando emperrado).

---

## Ordem de implementação recomendada

```
Task 1 (parser) → Task 3 (detectors) → Task 2 (OCR)
→ Task 4 (pipeline) → Task 5 (builder) → Task 6 (scorer)
→ Task 7 (loop) → iterar até 100%
```

Tasks 1, 2, 3 são independentes e podem ser feitas em paralelo.

---

## Dicas e armadilhas conhecidas

1. **Decimal PT-BR:** WPT usa vírgula (`21,5 BB`). Sempre `.replace(",", ".")` antes de `float()`.
2. **Logo NEXA POKER** interfere no slot 3 e 4 quando board vazio. O threshold `38` no slot 4 foi calibrado para isso.
3. **Taskbar Windows** aparece nos últimos 38px das mesas BL e BR (`TABLE_USEFUL_H = [540,540,502,502]`).
4. **Cartas nunca são 1 ou 2** — `count_board_cards` filtra esses valores como ruído.
5. **dLzinN é sempre o herói** (nosso jogador). Suas hole cards ficam visíveis na posição bottom-center de cada mesa.
6. **Mesas TL (HL3458)** não está no gabarito — ignore para o score.
7. **RapidOCR** retorna lista de `[bbox, texto, confiança]`. Filtre por `confiança > 0.7`.

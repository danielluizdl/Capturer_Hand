# Capturer Hand — Contexto para Claude

## O que é este projeto

Pipeline de **captura passiva de mãos de poker** a partir de gravações de vídeo de sessões ao vivo do **WPT Global (Nexa Poker)**. O objetivo é detectar automaticamente o que acontece nas mesas e gerar arquivos de hand history no formato PokerStars.

**Não existe hand history exportável no WPT Global** — tudo precisa ser extraído visualmente do vídeo.

---

## Contexto do jogo

- **Plataforma:** WPT Global (cliente Nexa Poker)
- **Formato:** 4 mesas simultâneas abertas ao mesmo tempo, em grade 2×2 na tela
- **Modalidade:** Hold'em No Limit, 8-max, stakes $0.05/$0.10/$0.20 com ante de $0.05
- **Hero:** `dLzinN` — sempre sentado na posição inferior central de cada mesa (SB/BB/STR dependendo do botão)
- **Posições especiais detectáveis:** SB, BB, STR (straddle)

---

## Formato do vídeo de entrada

- **Resolução:** 1920×1080 @ 30fps (gravado com OBS)
- **Layout:** 4 janelas WPT Global em grid 2×2
- **Crop por mesa:**

```
Mesa 0 (TL): x=0-960,    y=0-540    → HL3458 (mesa de teste, não no gabarito)
Mesa 1 (TR): x=960-1920, y=0-540    → HL4017 ✓
Mesa 2 (BL): x=0-960,    y=540-1080 → HL2332 ✓
Mesa 3 (BR): x=960-1920, y=540-1080 → HL3048 ✓
```

- Mesas inferiores (2 e 3) têm altura útil de 502px (taskbar Windows ocupa os últimos ~38px)
- O crop é feito por `src/detectors.py::crop_table(frame, table_idx)`

---

## Stack tecnológico

| Componente | Tecnologia |
|---|---|
| Leitura de vídeo | OpenCV (`cv2.VideoCapture`) |
| OCR | RapidOCR + ONNX Runtime (GPU via CUDA quando disponível) |
| Classificação de cartas | CNN MobileNetV2 treinada em 156 templates reais |
| Detecção de eventos | Detectores visuais calibrados (variância de pixel, HSV) |
| Template matching | `cv2.matchTemplate` (TM_CCOEFF_NORMED) |

---

## Cores dos naipes no WPT Global (HSV)

```
♣ paus    = VERDE vivo   (H 55-90,  S>100, V>80)
♦ ouros   = AZUL vivo    (H 95-130, S>80,  V>80)
♥ copas   = VERMELHO     (H 0-15 ou H 165-180, S>80, V>80)
♠ espadas = ESCURO/PRETO (V < 90)
```

Confirmado em frames reais. Usado em `src/card_classifier.py::detect_suit()` e `src/seat_card_recognizer.py::_detect_suit()`.

---

## Templates disponíveis

Todas as imagens são screenshots reais do WPT Global, do jogo ao vivo:

| Pasta | Uso | Qtd |
|---|---|---|
| `templates/baralho/` | Cartas do board (community cards) | 52 |
| `templates/baralho_seats/left/` | Carta esquerda do par de villain no showdown | 52 |
| `templates/baralho_seats/right/` | Carta direita do par de villain no showdown | 52 |

- Nomenclatura: `2c.png`, `ah.png`, `ks.png` (rank + naipe, sempre lowercase)
- Ranks: `2 3 4 5 6 7 8 9 t j q k a`
- Naipes: `c` (paus) `d` (ouros) `h` (copas) `s` (espadas)

Para adicionar novos templates: rodar `crop_seat_cards.py` com imagens de pares de cartas do showdown. Para treinar mais a CNN: adicionar imagens nas pastas e rodar `train_card_classifier.py`.

---

## Modelo CNN (MobileNetV2)

- **Arquivo de pesos:** `models/card_classifier.pt` (NÃO está no git — `.gitignore`)
- **Classes:** `models/card_classes.json` (52 cartas, ordem alfabética)
- **Input:** 64×64px RGB
- **Treino:** 156 imagens × 60 augmentações = ~9.360 exemplos
- **Uso:** GPU (CUDA) se disponível, senão CPU
- **Para treinar localmente:** `python train_card_classifier.py` (~15 min na GPU)

---

## Arquivos principais

```
run_video.py                   Ponto de entrada principal
agent_loop.py                  Loop de avaliação com gabarito (score 0-300)
gabarito.txt                   Ground truth — 3 mãos reais para validação

src/
  video_pipeline.py            Scan do vídeo a 10fps, emite TableEvent por mesa
  detectors.py                 crop_table(), count_board_cards(), has_action_buttons()
  card_classifier.py           Classificação das cartas do board (CNN + HSV + OCR fallback)
  card_cnn.py                  Módulo CNN: CardCNN, get_card_cnn(), predict(), predict_batch()
  seat_card_recognizer.py      Detecção de cartas do villain no showdown (template + NMS + HSV + CNN)
  hand_builder.py              Constrói HandHistory a partir dos eventos
  hh_writer_ps.py              Escreve output no formato PokerStars
  ocr_engine.py                OCR de título, pot, winner, stacks, hole cards
  gabarito_parser.py           Parser do gabarito.txt
  scorer.py                    Compara pipeline vs gabarito (score 0-300)
  seat_regions.py              Coordenadas calibráveis dos seats (calibrate_seats.py)

templates/
  baralho/                     52 templates das cartas do board
  baralho_seats/left/          52 templates carta esquerda do showdown
  baralho_seats/right/         52 templates carta direita do showdown
  *.PNG                        Templates de botões de ação, dealer, winner, etc.

models/
  card_classes.json            Mapeamento de índice → carta (no git)
  card_classifier.pt           Pesos da CNN treinada (NÃO está no git)
```

---

## Pipeline de processamento

```
vídeo.mp4
    ↓
video_pipeline.py  (10fps, 4 mesas em paralelo)
    ↓ TableEvent (board_change, pot_change, new_hand, action)
hand_builder.py
    ↓ frame a frame, por mesa
    ├── card_classifier.py    → board cards (CNN ≥0.80 + HSV confirm + OCR fallback)
    ├── ocr_engine.py         → hole cards hero, pot, winner, nomes, stacks
    ├── seat_card_recognizer  → villain showdown (template + NMS + HSV + CNN ≥0.65)
    └── HandHistory
hh_writer_ps.py
    ↓
output.txt  (formato PokerStars)
```

---

## O que o pipeline captura hoje

- Board cards (flop/turn/river)
- Hole cards do hero
- Villain showdown cards (quando mostram as cartas)
- Pot total
- Winner (quem ganhou o pot)
- Nomes dos jogadores
- Stacks dos jogadores no preflop
- Streets (preflop/flop/turn/river)
- Dealer button position
- Badge ALL-IN

## O que ainda falta implementar

- **Ações dos jogadores** — fold/call/raise/bet com valores (maior gap para HH completo)
- **Rastreamento de stacks** ao longo da mão (entrada/saída de fichas)
- **Posição exata** de cada jogador (SB/BB/STR/BTN/CO/HJ/UTG)
- **Pot por street** (quanto foi apostado em cada rua)
- **Múltiplos vencedores** (split pot)

---

## Como melhorar a acurácia das cartas

Hoje: 1 imagem por carta por contexto. Com mais capturas, o CNN fica mais preciso.

1. **Capturar mais templates:** jogar e pausar o vídeo em showdowns/boards diferentes, recortar com `crop_seat_cards.py`
2. **Retreinar:** `python train_card_classifier.py` após adicionar imagens
3. **Sem ground truth automático** — o WPT Global não exporta hand history, então toda validação é manual ou via gabarito gravado na mão

---

## Sistema de avaliação (agent_loop.py)

Score máximo: **300 pontos** sobre 3 mãos do gabarito.

```
Por mão (máx 100 pts):
  board_cards      30 pts  — cartas comunitárias corretas
  hole_cards       20 pts  — hole cards do hero corretas
  street_sequence  15 pts  — sequência de streets correta
  winner           15 pts  — vencedor correto
  total_pot        10 pts  — pote total (±5% tolerância)
  table_id          5 pts  — ID da mesa correto
  showdown          5 pts  — cartas dos villains no showdown
```

---

## Notas importantes

- O vídeo de teste (`video_cortado_1min.mp4`) e o vídeo longo (`video_longo_teste.mp4`) **não estão no git** (grandes demais)
- O modelo treinado (`models/card_classifier.pt`) **não está no git** — rodar `train_card_classifier.py` localmente
- RapidOCR usa GPU via `CUDAExecutionProvider` quando disponível — confirmar com `ort.get_available_providers()`
- O OCR do WPT Global tem problemas específicos com a fonte: `'2'` às vezes lido como `'7'`, `'6'` como `'b'` ou `'G'`
- Mesas inferiores têm altura 502px (não 540px) — usar frações relativas, nunca pixels absolutos

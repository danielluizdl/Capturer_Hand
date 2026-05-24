# Resultados do Agent Loop

Score final atingido: **258/300 (86.0%)**

| Mão    | Score | table_id | board_cards | hole_cards | street_seq | final_pot | winner |
|--------|-------|----------|-------------|------------|------------|-----------|--------|
| HL4017 | 100/100 | 10/10 ✓ | 30/30 ✓ | 20/20 ✓ | 15/15 ✓ | 15/15 ✓ | 10/10 ✓ |
| HL3048 |  69/100 | 10/10 ✓ | 24/30      | 0/20      | 10/15     | 15/15 ✓ | 10/10 ✓ |
| HL2332 |  89/100 | 10/10 ✓ | 24/30      | 20/20 ✓ | 10/15     | 15/15 ✓ | 10/10 ✓ |

## Detalhamento

### HL4017 — 100/100
Board: Jd✓ 5s✓ 2h✓ 3d✓ 7s✓  
Hole: 9s✓ 4s✓

### HL3048 — 69/100
Board: Kd✓ Ad✓ 9d✓ 8d✓ (miss: hole_cards 2c 6s — rank '2' lido como '7' pelo OCR, posição não calibrada)

### HL2332 — 89/100
Board: 5h✓ Qs✓ 8d✓ Kh✓ (miss: street turn não detectada na sequência correta)

## Melhorias implementadas

1. **Classificador HSV**: detecta naipe por análise de cor (♣=verde, ♦=azul, ♥=vermelho, ♠=escuro)
2. **Votação multi-frame por slot**: cada slot vota independentemente em 5 frames; permite detecção parcial
3. **Seleção do melhor evento por street**: para múltiplos board_change events da mesma street, usa o com maior confiança na nova carta
4. **Herança de slots**: flop slots ausentes no turn/river são herdados de detecções anteriores

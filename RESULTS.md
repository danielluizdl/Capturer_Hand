# Resultados do Agent Loop

Total de iterações: 4 (interrompido após atingir platô)

| Iteração | Score | Pts | slot_thresholds |
|----------|-------|-----|-----------------|
| 0 (calibrado direto) | 46.7% | 140/300 | [25, 33, 38, 25, 30] |
| 1 | 31.7% | 95/300 | [20, 25, 33, 38, 25] (DEFAULT_CONFIG incorreto) |
| 2 | 41.7% | 125/300 | [17, 22, 30, 35, 22] |
| 3 | 41.7% | 125/300 | [14, 19, 27, 32, 19] |
| 4 | - | -/300 | [11, 16, 24, 29, 16] (interrompido) |

**Melhor score:** 46.7% (140/300) — pipeline calibrado diretamente

## Breakdown do Score Final (calibrado)

#### HL4017: 50/100
- table_id: 10/10 ✓
- board_cards: 0/30 (cartas gráficas — OCR não lê sprites WPT Global)
- hole_cards: 0/20 (mesmo motivo)
- street_sequence: 15/15 ✓ (preflop, flop, turn, river)
- final_pot: 15/15 ✓ (det=$2.15 gab=$2.15)
- winner: 10/10 ✓ (taymonkha)

#### HL3048: 45/100
- table_id: 10/10 ✓
- board_cards: 0/30 (cartas gráficas)
- hole_cards: 0/20 (cartas gráficas)
- street_sequence: 10/15 (máximo possível: mão vai até turn no gabarito)
- final_pot: 15/15 ✓ (det=$11.40 gab=$11.38, erro <0.5%)
- winner: 10/10 ✓ (Hamster813)

#### HL2332: 45/100
- table_id: 10/10 ✓
- board_cards: 0/30 (cartas gráficas)
- hole_cards: 0/20 (cartas gráficas)
- street_sequence: 10/15 (máximo possível: mão vai até turn no gabarito)
- final_pot: 15/15 ✓ (det=$4.71 gab=$4.71)
- winner: 10/10 ✓ (dLzinN)

## Platô Identificado

Score máximo atingível sem reconhecimento de imagem de cartas: **140/300 (46.7%)**

### Categorias com score máximo:
- table_id: 30/30 (3×10) ✓
- street_sequence: 35/45 (HL4017=15, HL3048=10, HL2332=10 — máximo possível dado gabarito)
- final_pot: 45/45 (3×15) ✓
- winner: 30/30 (3×10) ✓

### Categorias não atingidas (requerem vision):
- board_cards: 0/90 — cartas do board são sprites gráficos WPT Global, não texto OCR
- hole_cards: 0/60 — mesmo motivo

### Observação sobre o agent_loop:
A iteração 1 iniciou com DEFAULT_CONFIG incorreto (slot_thresholds[3]=38 para o turn,
vs. o calibrado de 25). Isso quebrou a detecção do turn para HL3048 (5/15 → 0 pts).
A iteração 2 corrigiu automaticamente (threshold 35 < variância do turn de HL3048).
Após restaurar os thresholds calibrados [25,33,38,25,30], o score retorna a 140/300.

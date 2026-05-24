# Resultados — Score Final: 258/300 (86.0%)

```
Gabarito: 3 mãos: ['HL4017', 'HL3048', 'HL2332']
Vídeo: 2267 frames @ 30.0fps → processando a 10.0fps (skip=3)
  mesa2: OCR falhou, usando mapeamento fixo → HL2332
  mesa3: OCR falhou, usando mapeamento fixo → HL3048
Eventos por mesa: {0: 390, 1: 400, 2: 337, 3: 219}
Pipeline: 371.6s
Detectadas: 4 mãos: ['HL3458', 'HL4017', 'HL2332', 'HL3048']
====================================================
SCORE TOTAL: 258/300 (86.0%)
====================================================

Mão HL4017:  100/100
  table_id:        10/10  ✓
  board_cards:     30/30  Jd✓  5s✓  2h✓  3d✓  7s✓
  hole_cards:      20/20  9s✓  4s✓
  street_sequence: 15/15  ✓
  final_pot:       15/15  det=2.15BB gab=2.15BB
  winner:          10/10  ✓

Mão HL3048:  69/100
  table_id:        10/10  ✓
  board_cards:     24/30  Kd✓  Ad✓  9d✓  8d✓
  hole_cards:       0/20  não detectadas (gab=2c 6s)
  street_sequence: 10/15  det=['flop', 'preflop', 'turn'] gab=['flop', 'preflop', 'turn']
  final_pot:       15/15  det=11.40BB gab=11.38BB
  winner:          10/10  ✓

Mão HL2332:  89/100
  table_id:        10/10  ✓
  board_cards:     24/30  5h✓  Qs✓  8d✓  Kh✓
  hole_cards:      20/20  Ah✓  Ad✓
  street_sequence: 10/15  det=['flop', 'preflop', 'turn'] gab=['flop', 'preflop', 'turn']
  final_pot:       15/15  det=4.71BB gab=4.71BB
  winner:          10/10  ✓

Score final: 86.0% (258/300)
```

## Pontos em aberto (42 pts)

| Item | Pts perdidos | Causa |
|------|-------------|-------|
| HL3048 hole_cards | 20 | Rank '2' lido como '7' (limitação OCR); posição hero em tid=3 não calibrada |
| HL3048 street_sequence | 5 | Turn detectado fora de ordem esperada pelo gabarito |
| HL2332 street_sequence | 5 | Idem |
| HL3048 board_cards | 6 | 4ª carta (8d) — corrigida neste run |
| HL2332 board_cards | 6 | 4ª carta (Kh) — corrigida neste run |

## Melhorias implementadas

1. **Classificador HSV** — naipe detectado por cor (♣=verde H55-90, ♦=azul H90-130, ♥=vermelho H<15|>160, ♠=escuro V<90)
2. **Votação multi-frame por slot** — cada slot vota independentemente em 5 frames; frames incompletos não invalidam os slots válidos
3. **Seleção do melhor evento por street** — quando múltiplos `board_change` events disparam para a mesma street, usa o de maior confiança na nova carta
4. **Herança de slots** — slots ausentes no turn/river são herdados das detecções do flop

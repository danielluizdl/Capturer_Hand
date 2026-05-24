# Plano: Detecção de Ações + Showdown

## Objetivo
600/600 pts (100%) no rubric estendido.
Tier 1 (300 pts): já feito — table_id, board, hole_cards, streets, pot, winner.
Tier 2 (300 pts): ações por street + showdown.

## Rubric Tier 2 — 100 pts/hand × 3 mãos = 300 pts

| Categoria          | Pts | Critério |
|--------------------|-----|----------|
| preflop_actions    | 40  | % ações corretas (player + tipo + valor ±10%) |
| flop_actions       | 20  | % ações corretas |
| turn_actions       | 20  | % ações corretas |
| river_actions      | 10  | % ações corretas |
| showdown           | 10  | cartas mostradas por player |

**Score atual antes de implementar:** 300/600 (50%)

---

## Fases de Implementação

### Fase 0 — Infraestrutura de Scoring ✅ (esta iteração)
- Estender scorer.py com Tier 2
- Estender HandHistory com campo `showdown: dict[str, list[str]]`
- Estender gabarito_parser.py para ler SHOW DOWN
- Criar agent_loop_v2.py (ações + showdown)

### Fase 1 — Ações do Herói (dLzinN) no Preflop
**Detectar:**
- Quando `action_btns=True` para o herói → anotar frame e estado dos botões
- OCR do botão ativo: Desistir (fold), Checar (check), Pagar (call $X), Aumentar (raise to $X)
- Associar ao player "dLzinN"

**Ponto de partida:** `has_action_buttons()` já detecta presença dos botões.
**Próximo passo:** OCR na faixa de botões (y > 90%) para ler qual botão foi clicado.

**Ganho esperado:** +3 a 5 ações por mão × 10 pts/ação proporcional → ~+20-30 pts

### Fase 2 — Ações dos Outros Jogadores (Folds + Calls simples)
**Detectar:**
- Badge de FOLD/CHECK que aparece sobre o jogador após a ação (texto breve na mesa)
- WPT Global exibe overlay "FOLD" / "CHECK" no espaço do jogador por ~1s
- Capturar frame durante overlay → OCR nome + tipo de ação

**Regiões dos jogadores:** 8 posições fixas ao redor da mesa (calibrar por mão)
**Ponto de partida:** Frames de `action` events — quando action_btns=False e havia transição

**Ganho esperado:** folds são a maioria no preflop (~6-7 por mão) → grande impacto

### Fase 3 — Valores de Aposta (raises, bets, calls com valor)
**Detectar:**
- Quando jogador bet/raise, um valor aparece acima das fichas do jogador
- OCR na região de fichas de cada jogador
- Regex: `\$?(\d+[\.,]\d{2})` ou BB notation

**Ganho esperado:** +valores corretos para ~5-7 ações com valor por mão

### Fase 4 — Ações Pós-Flop (Flop, Turn, River)
- Mesma lógica das fases 1-3, porém disparadas por eventos board_change
- Janela de detecção: frames entre board_change(N) e board_change(N+1)

### Fase 5 — Showdown
**Detectar:**
- Frame onde *** SHOW DOWN *** aparece (texto grande na tela)
- OCR cards de cada jogador visível na mesa no momento do showdown
- Associar player_name → [card1, card2]
- Apenas HL4017 tem showdown nas 3 mãos do gabarito

---

## Estratégia de Loop

```
Rodar pipeline → comparar vs gabarito (ações) → ver gap maior → implementar → repetir
```

Critério de parada: 600/600 ou MAX_NO_PROGRESS=3 iterações sem melhora.

---

## Dificuldades Conhecidas

1. **Nomes em chinês** — OCR pode falhar em caracteres Unicode; fallback por posição
2. **Timing** — entre o clique do jogador e o badge aparecer há ~2-5 frames de lag
3. **Sobreposição de badges** — múltiplos jogadores podem agir em sequência rápida
4. **Uncalled bet returned** — não é uma ação mas aparece no gabarito; tratar como evento separado
5. **Showdown hand strength** — "high card Ace - Ten kicker" é difícil de gerar; focar nas cartas

---

## Arquivos a Modificar / Criar

| Arquivo | Mudança |
|---------|---------|
| `src/gabarito_parser.py` | Adicionar parsing de SHOW DOWN section → `showdown` field |
| `src/hand_builder.py` | Adicionar `HandHistory.showdown`, stub `_extract_actions()` |
| `src/scorer.py` | Adicionar Tier 2 ao rubric |
| `src/detectors.py` | `detect_action_badge()`, `detect_showdown_frame()` |
| `src/ocr_engine.py` | `ocr_action_buttons()`, `ocr_player_action()`, `ocr_showdown_cards()` |
| `agent_loop_v2.py` | Loop de melhoria para ações + showdown |

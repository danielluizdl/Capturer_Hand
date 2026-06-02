"""
Gera mão em formato PokerStars Hand History a partir de HandHistory detectado.

Uso típico:
    from src.hh_writer_ps import write_session, diff_vs_gabarito
    text = write_session(detected_hands)
    diff = diff_vs_gabarito(text, "gabarito.txt")
"""
from __future__ import annotations
import difflib
from src.gabarito_parser import HandHistory

HERO = "dLzinN"
_ANTE = 0.05
_SB   = 0.05
_BB   = 0.10
_STR  = 0.20


def write_hand(hh: HandHistory, hand_id: int = 900000) -> str:
    lines: list[str] = []

    # [1] Header
    lines.append(
        f"PokerStars Hand #{hand_id}: "
        f"Hold'em No Limit (${_SB}/${_BB}/${_STR}({_ANTE})) - "
        f"2026/05/21 00:00:00 ET"
    )

    # [2] Table
    lines.append(
        f"Table '{hh.table_id}' 8-max Seat #{hh.button_seat} is the button"
    )

    # [3] Seats
    for seat_num in sorted(hh.players):
        p = hh.players[seat_num]
        lines.append(f"Seat {seat_num}: {p['name']} (${p['chips']:.2f} in chips)")

    # [4] Antes
    for seat_num in sorted(hh.players):
        lines.append(f"{hh.players[seat_num]['name']}: posts the ante ${_ANTE:.2f}")

    # [5] Blinds — determina SB/BB/STR pela posição relativa ao botão
    seats = sorted(hh.players)
    n = len(seats)
    if hh.button_seat in seats:
        btn_idx = seats.index(hh.button_seat)
        sb_seat  = seats[(btn_idx + 1) % n]
        bb_seat  = seats[(btn_idx + 2) % n]
        str_seat = seats[(btn_idx + 3) % n]
        lines.append(f"{hh.players[sb_seat]['name']}: posts small blind ${_SB:.2f}")
        lines.append(f"{hh.players[bb_seat]['name']}: posts big blind ${_BB:.2f}")
        lines.append(f"{hh.players[str_seat]['name']}: posts straddle ${_STR:.2f}")

    # [6] Hole cards
    lines.append("*** HOLE CARDS ***")
    if hh.hole_cards:
        lines.append(f"Dealt to {HERO} [{' '.join(hh.hole_cards)}]")

    # [7–12] Ações por rua
    for act in hh.actions:
        street = act["street"]
        # Injeta cabeçalho da rua antes da primeira ação
        if street == "flop" and "*** FLOP ***" not in "\n".join(lines):
            flop = hh.board[:3]
            lines.append(f"*** FLOP *** [{' '.join(flop)}]")
        elif street == "turn" and "*** TURN ***" not in "\n".join(lines):
            flop = hh.board[:3]
            turn = hh.board[3:4]
            lines.append(f"*** TURN *** [{' '.join(flop)}] [{' '.join(turn)}]")
        elif street == "river" and "*** RIVER ***" not in "\n".join(lines):
            flop  = hh.board[:3]
            turn  = hh.board[3:4]
            river = hh.board[4:5]
            lines.append(
                f"*** RIVER *** [{' '.join(flop)}] [{' '.join(turn)}] [{' '.join(river)}]"
            )
        lines.append(_format_action(act))

    # Se não houve ações mas há board (mão encerrada por fold silencioso), emite ruas
    if hh.board and "*** FLOP ***" not in "\n".join(lines) and "flop" in hh.streets:
        flop = hh.board[:3]
        lines.append(f"*** FLOP *** [{' '.join(flop)}]")
    if len(hh.board) >= 4 and "*** TURN ***" not in "\n".join(lines) and "turn" in hh.streets:
        flop = hh.board[:3]
        turn = hh.board[3:4]
        lines.append(f"*** TURN *** [{' '.join(flop)}] [{' '.join(turn)}]")
    if len(hh.board) == 5 and "*** RIVER ***" not in "\n".join(lines) and "river" in hh.streets:
        flop  = hh.board[:3]
        turn  = hh.board[3:4]
        river = hh.board[4:5]
        lines.append(
            f"*** RIVER *** [{' '.join(flop)}] [{' '.join(turn)}] [{' '.join(river)}]"
        )

    # [13] Showdown — cartas dos opponents detectadas por template matching
    if hh.showdown:
        lines.append("*** SHOWDOWN ***")
        for player_key, cards in sorted(hh.showdown.items()):
            cards_str = " ".join(cards)
            lines.append(f"{player_key}: shows [{cards_str}]")

    # [14] Coleta do pote
    if hh.winner:
        lines.append(f"{hh.winner} collected ${hh.total_pot:.2f} from pot")

    # [16] Summary
    lines.append("*** SUMMARY ***")
    lines.append(f"Total pot ${hh.total_pot:.2f} | Rake $0.00")
    if hh.board:
        lines.append(f"Board [{' '.join(hh.board)}]")

    return "\n".join(lines)


def _format_action(act: dict) -> str:
    player = act["player"]
    action = act["action"]
    amount = act.get("amount", 0.0)

    if action in ("folds", "checks"):
        return f"{player}: {action}"
    if action in ("calls", "bets"):
        return f"{player}: {action} ${amount:.2f}"
    if action == "raises":
        # O parser armazena apenas o delta; "to $TOTAL" seria necessário
        # mas não temos o total separado — emite apenas o delta disponível
        return f"{player}: raises ${amount:.2f}"
    return f"{player}: {action}"


def write_session(hands: list[HandHistory], start_id: int = 900001) -> str:
    """Serializa lista de HandHistory em formato PS, separadas por linha em branco."""
    parts = [write_hand(hh, start_id + i) for i, hh in enumerate(hands)]
    return "\n\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# Comparação texto a texto
# ---------------------------------------------------------------------------

def _normalize(text: str) -> list[str]:
    """
    Reduz o texto a linhas relevantes para comparação, descartando:
    - Número do hand (único e gerado automaticamente)
    - Timestamp (não detectável)
    - Descrições de mão no showdown (e.g. 'high card Ace - Ten kicker')
    """
    out = []
    for line in text.splitlines():
        # Normaliza número de hand
        import re
        line = re.sub(r"Hand #\d+", "Hand #XXXXX", line)
        # Normaliza timestamp
        line = re.sub(r"\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2} ET", "TIMESTAMP", line)
        # Remove descrição de mão no showdown: ' (high card Ace...)' → ''
        line = re.sub(r"\s+\(.+\)$", "", line)
        out.append(line)
    return out


def diff_vs_gabarito(
    detected_text: str,
    gabarito_path: str,
    context_lines: int = 3,
) -> str:
    """
    Compara texto gerado pelo pipeline com o gabarito.txt.
    Retorna diff unificado como string.
    """
    with open(gabarito_path, encoding="utf-8") as f:
        gabarito_text = f.read()

    det_lines = _normalize(detected_text)
    gab_lines = _normalize(gabarito_text)

    diff = difflib.unified_diff(
        gab_lines,
        det_lines,
        fromfile="gabarito.txt",
        tofile="pipeline_output",
        lineterm="",
        n=context_lines,
    )
    return "\n".join(diff)


def compare_key_lines(
    detected: list[HandHistory],
    gabarito: list[HandHistory],
) -> None:
    """
    Imprime comparação focada nas linhas que o pipeline pode gerar:
    board, hole cards, winner, pot.
    """
    gab_by_id = {h.table_id: h for h in gabarito}
    det_by_id = {h.table_id: h for h in detected}

    for tid, gab in sorted(gab_by_id.items()):
        det = det_by_id.get(tid)
        print(f"\n{'─'*52}")
        print(f"Mesa: {tid}")
        print(f"{'─'*52}")

        _cmp_cards("board      ",
                   det.board if det else [], gab.board,
                   f"[{' '.join(gab.board)}]" if gab.board else "(sem board)")
        _cmp_cards("hole_cards ",
                   det.hole_cards if det else [], gab.hole_cards,
                   f"[{' '.join(gab.hole_cards)}]")
        _cmp_streets("streets    ",
                     det.streets if det else [], gab.streets)
        _cmp_pot("pot        ",
                 det.total_pot if det else None, gab.total_pot)
        _cmp_str("winner     ",
                 (det.winner or "") if det else "", gab.winner)


def _cmp_cards(label: str, det: list, gab: list, gab_repr: str) -> None:
    ok = det == gab
    mark = "✓" if ok else "✗"
    det_repr = f"[{' '.join(det)}]" if det else "(vazio)"
    if ok:
        print(f"  {label}: {mark} {det_repr}")
    else:
        print(f"  {label}: {mark}  det={det_repr}  gab={gab_repr}")


def _cmp_streets(label: str, det: list, gab: list) -> None:
    ok = sorted(det) == sorted(gab)
    mark = "✓" if ok else "✗"
    if ok:
        print(f"  {label}: {mark} {sorted(gab)}")
    else:
        print(f"  {label}: {mark}  det={sorted(det)}  gab={sorted(gab)}")


def _cmp_pot(label: str, det: float | None, gab: float) -> None:
    if det is None:
        print(f"  {label}: ✗  det=(não detectado)  gab=${gab:.2f}")
        return
    diff_pct = abs(det - gab) / gab * 100 if gab else 0
    ok = diff_pct <= 5
    mark = "✓" if ok else "✗"
    if ok:
        print(f"  {label}: {mark} ${det:.2f}")
    else:
        print(f"  {label}: {mark}  det=${det:.2f}  gab=${gab:.2f}  ({diff_pct:.1f}% diff)")


def _cmp_str(label: str, det: str, gab: str) -> None:
    ok = det.lower().strip() == gab.lower().strip()
    mark = "✓" if ok else "✗"
    if ok:
        print(f"  {label}: {mark} {gab}")
    else:
        print(f"  {label}: {mark}  det='{det}'  gab='{gab}'")

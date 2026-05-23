"""Task 6 — scorer: compara HandHistory detectado vs gabarito."""
from __future__ import annotations
from src.gabarito_parser import HandHistory

RUBRIC = {
    "table_id":        10,
    "board_cards":     30,   # 6 pts por carta correta (max 5)
    "hole_cards":      20,   # 10 pts por carta correta (max 2)
    "street_sequence": 15,   # 5 pts por street correta (flop/turn/river)
    "final_pot":       15,
    "winner":          10,
}

MAX_PER_HAND = sum(RUBRIC.values())   # 100
MAX_TOTAL    = 300                     # 3 mãos


def score_hands(
    detected: list[HandHistory],
    gabarito: list[HandHistory],
) -> dict:
    """
    Emparelha detected com gabarito pelo table_id e calcula score.
    Retorna dict com breakdown por mão e score total.
    """
    # Mapeia gabarito por table_id
    gab_by_id = {h.table_id: h for h in gabarito}

    # Mapeia detected por table_id
    det_by_id: dict[str, HandHistory] = {}
    for h in detected:
        if h.table_id in gab_by_id:
            det_by_id[h.table_id] = h

    hand_scores: list[dict] = []
    total_pts = 0

    for gab_hand in gabarito:
        tid = gab_hand.table_id
        det_hand = det_by_id.get(tid)
        hs = _score_hand(det_hand, gab_hand)
        hand_scores.append(hs)
        total_pts += hs["total"]

    pct = total_pts / MAX_TOTAL * 100

    return {
        "total_pts":  total_pts,
        "max_pts":    MAX_TOTAL,
        "pct":        pct,
        "hands":      hand_scores,
    }


def _score_hand(det: HandHistory | None, gab: HandHistory) -> dict:
    scores: dict[str, int] = {}

    # table_id
    if det is not None and det.table_id == gab.table_id:
        scores["table_id"] = RUBRIC["table_id"]
    else:
        scores["table_id"] = 0

    # board_cards — 6 pts por carta correta na mesma posição
    scores["board_cards"] = 0
    det_board = det.board if det else []
    for i, card in enumerate(gab.board):
        if i < len(det_board) and det_board[i].lower() == card.lower():
            scores["board_cards"] += 6
    scores["board_cards"] = min(scores["board_cards"], RUBRIC["board_cards"])

    # hole_cards — 10 pts por carta correta
    scores["hole_cards"] = 0
    det_holes = det.hole_cards if det else []
    for i, card in enumerate(gab.hole_cards):
        if i < len(det_holes) and det_holes[i].lower() == card.lower():
            scores["hole_cards"] += 10
    scores["hole_cards"] = min(scores["hole_cards"], RUBRIC["hole_cards"])

    # street_sequence — 5 pts por street correta (flop, turn, river)
    scores["street_sequence"] = 0
    det_streets = set(det.streets) if det else set()
    for street in ("flop", "turn", "river"):
        if street in det_streets and street in gab.streets:
            scores["street_sequence"] += 5
    scores["street_sequence"] = min(scores["street_sequence"], RUBRIC["street_sequence"])

    # final_pot — tolerância de 5%/10%/20%
    scores["final_pot"] = 0
    if det is not None and det.total_pot and gab.total_pot:
        diff_pct = abs(det.total_pot - gab.total_pot) / gab.total_pot
        if diff_pct <= 0.05:
            scores["final_pot"] = 15
        elif diff_pct <= 0.10:
            scores["final_pot"] = 10
        elif diff_pct <= 0.20:
            scores["final_pot"] = 5
        else:
            scores["final_pot"] = 0

    # winner
    det_winner = (det.winner or "").strip().lower() if det else ""
    gab_winner = gab.winner.strip().lower()
    scores["winner"] = RUBRIC["winner"] if det_winner and det_winner == gab_winner else 0

    total = sum(scores.values())

    return {
        "table_id":   gab.table_id,
        "total":      total,
        "max":        MAX_PER_HAND,
        "scores":     scores,
        "det_board":  det.board if det else [],
        "gab_board":  gab.board,
        "det_holes":  det.hole_cards if det else [],
        "gab_holes":  gab.hole_cards,
        "det_streets": det.streets if det else [],
        "gab_streets": gab.streets,
        "det_pot":    det.total_pot if det else None,
        "gab_pot":    gab.total_pot,
        "det_winner": det.winner if det else "",
        "gab_winner": gab.winner,
    }


def print_report(report: dict) -> None:
    """Imprime relatório detalhado do score."""
    total = report["total_pts"]
    max_t = report["max_pts"]
    pct   = report["pct"]

    print("=" * 52)
    print(f"SCORE TOTAL: {total}/{max_t} ({pct:.1f}%)")
    print("=" * 52)

    for hs in report["hands"]:
        tid = hs["table_id"]
        t   = hs["total"]
        m   = hs["max"]
        print(f"\nMão {tid}:  {t}/{m}")

        s = hs["scores"]

        # table_id
        ok = "✓" if s["table_id"] == RUBRIC["table_id"] else "✗"
        print(f"  table_id:        {s['table_id']:2d}/{RUBRIC['table_id']}  {ok}")

        # board_cards
        det_b = hs["det_board"]
        gab_b = hs["gab_board"]
        board_detail = _board_detail(det_b, gab_b)
        print(f"  board_cards:     {s['board_cards']:2d}/{RUBRIC['board_cards']}  {board_detail}")

        # hole_cards
        det_h = hs["det_holes"]
        gab_h = hs["gab_holes"]
        holes_detail = _holes_detail(det_h, gab_h)
        print(f"  hole_cards:      {s['hole_cards']:2d}/{RUBRIC['hole_cards']}  {holes_detail}")

        # street_sequence
        det_s = sorted(hs["det_streets"])
        gab_s = sorted(hs["gab_streets"])
        ok = "✓" if s["street_sequence"] == RUBRIC["street_sequence"] else f"det={det_s} gab={gab_s}"
        print(f"  street_sequence: {s['street_sequence']:2d}/{RUBRIC['street_sequence']}  {ok}")

        # final_pot
        dp = hs["det_pot"]
        gp = hs["gab_pot"]
        pot_detail = f"det={dp:.2f}BB gab={gp:.2f}BB" if dp else f"não detectado (gab={gp:.2f})"
        print(f"  final_pot:       {s['final_pot']:2d}/{RUBRIC['final_pot']}  {pot_detail}")

        # winner
        dw = hs["det_winner"] or "(não detectado)"
        gw = hs["gab_winner"]
        ok = "✓" if s["winner"] == RUBRIC["winner"] else f"det={dw} gab={gw}"
        print(f"  winner:          {s['winner']:2d}/{RUBRIC['winner']}  {ok}")


def _board_detail(det: list[str], gab: list[str]) -> str:
    if not det:
        return f"não detectadas (gab={' '.join(gab)})"
    parts = []
    for i, card in enumerate(gab):
        if i < len(det):
            mark = "✓" if det[i].lower() == card.lower() else f"✗(det={det[i]})"
            parts.append(f"{card}{mark}")
        else:
            parts.append(f"{card}✗")
    return "  ".join(parts)


def _holes_detail(det: list[str], gab: list[str]) -> str:
    if not det:
        return f"não detectadas (gab={' '.join(gab)})"
    parts = []
    for i, card in enumerate(gab):
        if i < len(det):
            mark = "✓" if det[i].lower() == card.lower() else f"✗(det={det[i]})"
            parts.append(f"{card}{mark}")
        else:
            parts.append(f"{card}✗")
    return "  ".join(parts)

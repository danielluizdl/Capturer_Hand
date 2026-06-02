"""Task 6 — scorer: compara HandHistory detectado vs gabarito."""
from __future__ import annotations
from src.gabarito_parser import HandHistory

# Tier 1 — rubric original (já 300/300)
RUBRIC = {
    "table_id":        10,
    "board_cards":     30,   # 6 pts por carta correta (max 5)
    "hole_cards":      20,   # 10 pts por carta correta (max 2)
    "street_sequence": 15,   # 5 pts por street correta (flop/turn/river)
    "final_pot":       15,
    "winner":          10,
}

# Tier 2 — ações + showdown (novo, começa em 0/300)
RUBRIC_T2 = {
    "preflop_actions": 40,   # % ações corretas (player + tipo)
    "flop_actions":    20,
    "turn_actions":    20,
    "river_actions":   10,
    "showdown":        10,   # cartas mostradas por player
}

MAX_PER_HAND    = sum(RUBRIC.values())         # 100
MAX_PER_HAND_T2 = sum(RUBRIC_T2.values())      # 100
MAX_TOTAL       = 300                           # tier 1 × 3 mãos
MAX_TOTAL_T2    = 300                           # tier 2 × 3 mãos
MAX_TOTAL_ALL   = MAX_TOTAL + MAX_TOTAL_T2      # 600


def score_hands(
    detected: list[HandHistory],
    gabarito: list[HandHistory],
) -> dict:
    """
    Emparelha detected com gabarito pelo table_id e calcula score.
    Retorna dict com breakdown por mão e score total (Tier1 + Tier2 = max 600).
    """
    gab_by_id = {h.table_id: h for h in gabarito}
    det_by_id: dict[str, HandHistory] = {}
    for h in detected:
        if h.table_id in gab_by_id:
            det_by_id[h.table_id] = h

    hand_scores: list[dict] = []
    total_t1 = 0
    total_t2 = 0

    for gab_hand in gabarito:
        tid = gab_hand.table_id
        det_hand = det_by_id.get(tid)
        hs = _score_hand(det_hand, gab_hand)
        hand_scores.append(hs)
        total_t1 += hs["total_t1"]
        total_t2 += hs["total_t2"]

    total_all = total_t1 + total_t2
    pct_t1  = total_t1  / MAX_TOTAL     * 100
    pct_t2  = total_t2  / MAX_TOTAL_T2  * 100
    pct_all = total_all / MAX_TOTAL_ALL * 100

    return {
        "total_pts":   total_t1,          # backward compat
        "total_t1":    total_t1,
        "total_t2":    total_t2,
        "total_all":   total_all,
        "max_pts":     MAX_TOTAL,         # backward compat
        "max_t1":      MAX_TOTAL,
        "max_t2":      MAX_TOTAL_T2,
        "max_all":     MAX_TOTAL_ALL,
        "pct":         pct_t1,            # backward compat
        "pct_t1":      pct_t1,
        "pct_t2":      pct_t2,
        "pct_all":     pct_all,
        "hands":       hand_scores,
    }


def _score_hand(det: HandHistory | None, gab: HandHistory) -> dict:
    # ── Tier 1 ──────────────────────────────────────────────────────────────
    t1: dict[str, int] = {}

    t1["table_id"] = RUBRIC["table_id"] if (det and det.table_id == gab.table_id) else 0

    det_board = det.board if det else []
    if gab.board:
        correct = sum(1 for i, c in enumerate(gab.board)
                      if i < len(det_board) and det_board[i].lower() == c.lower())
        t1["board_cards"] = round(correct / len(gab.board) * RUBRIC["board_cards"])
    else:
        t1["board_cards"] = 0

    det_holes = det.hole_cards if det else []
    t1["hole_cards"] = 0
    for i, c in enumerate(gab.hole_cards):
        if i < len(det_holes) and det_holes[i].lower() == c.lower():
            t1["hole_cards"] += 10
    t1["hole_cards"] = min(t1["hole_cards"], RUBRIC["hole_cards"])

    det_streets = set(det.streets) if det else set()
    gab_playable = [s for s in ("flop", "turn", "river") if s in gab.streets]
    if gab_playable:
        correct_s = sum(1 for s in gab_playable if s in det_streets)
        t1["street_sequence"] = round(correct_s / len(gab_playable) * RUBRIC["street_sequence"])
    else:
        t1["street_sequence"] = 0

    t1["final_pot"] = 0
    if det and det.total_pot and gab.total_pot:
        diff = abs(det.total_pot - gab.total_pot) / gab.total_pot
        t1["final_pot"] = 15 if diff <= 0.05 else 10 if diff <= 0.10 else 5 if diff <= 0.20 else 0

    det_w = (det.winner or "").strip().lower() if det else ""
    t1["winner"] = RUBRIC["winner"] if det_w and det_w == gab.winner.strip().lower() else 0

    total_t1 = sum(t1.values())

    # ── Tier 2 ──────────────────────────────────────────────────────────────
    t2: dict[str, int] = {}
    det_actions = det.actions if det else []

    for street, key, max_pts in [
        ("preflop", "preflop_actions", RUBRIC_T2["preflop_actions"]),
        ("flop",    "flop_actions",    RUBRIC_T2["flop_actions"]),
        ("turn",    "turn_actions",    RUBRIC_T2["turn_actions"]),
        ("river",   "river_actions",   RUBRIC_T2["river_actions"]),
    ]:
        gab_acts = [(a["player"], a["action"]) for a in gab.actions if a["street"] == street]
        det_acts = [(a["player"], a["action"]) for a in det_actions if a["street"] == street]
        if not gab_acts:
            t2[key] = max_pts  # nenhuma ação esperada → pontuação máxima
        else:
            t2[key] = round(_match_actions(det_acts, gab_acts) / len(gab_acts) * max_pts)

    # showdown
    gab_sd = gab.showdown if gab else {}
    det_sd = det.showdown if det else {}
    if not gab_sd:
        t2["showdown"] = RUBRIC_T2["showdown"]  # sem showdown esperado → full pts
    else:
        correct_sd = sum(
            1 for player, cards in gab_sd.items()
            if player in det_sd and
            sorted(c.lower() for c in det_sd[player]) == sorted(c.lower() for c in cards)
        )
        t2["showdown"] = round(correct_sd / len(gab_sd) * RUBRIC_T2["showdown"])

    total_t2 = sum(t2.values())

    return {
        "table_id":    gab.table_id,
        "total":       total_t1,           # backward compat
        "total_t1":    total_t1,
        "total_t2":    total_t2,
        "max":         MAX_PER_HAND,
        "max_t2":      MAX_PER_HAND_T2,
        "scores":      t1,
        "scores_t2":   t2,
        "det_board":   det.board if det else [],
        "gab_board":   gab.board,
        "det_holes":   det.hole_cards if det else [],
        "gab_holes":   gab.hole_cards,
        "det_streets": det.streets if det else [],
        "gab_streets": gab.streets,
        "det_pot":     det.total_pot if det else None,
        "gab_pot":     gab.total_pot,
        "det_winner":  det.winner if det else "",
        "gab_winner":  gab.winner,
        "det_actions": det_actions,
        "gab_actions": gab.actions,
        "det_showdown": det_sd,
        "gab_showdown": gab_sd,
    }


def _match_actions(det: list[tuple], gab: list[tuple]) -> int:
    """Conta quantas ações detectadas casam com o gabarito (multiset matching)."""
    from collections import Counter
    det_c = Counter(det)
    gab_c = Counter(gab)
    return sum(min(det_c[k], gab_c[k]) for k in gab_c)


def print_report(report: dict) -> None:
    """Imprime relatório detalhado do score (Tier 1 + Tier 2)."""
    t1    = report["total_t1"]
    t2    = report["total_t2"]
    total = report["total_all"]
    max_t1  = report["max_t1"]
    max_t2  = report["max_t2"]
    max_all = report["max_all"]
    pct_t1  = report["pct_t1"]
    pct_t2  = report["pct_t2"]
    pct_all = report["pct_all"]

    print("=" * 60)
    print(f"SCORE TIER 1:  {t1}/{max_t1} ({pct_t1:.1f}%)")
    print(f"SCORE TIER 2:  {t2}/{max_t2} ({pct_t2:.1f}%)")
    print(f"SCORE TOTAL:   {total}/{max_all} ({pct_all:.1f}%)")
    print("=" * 60)

    for hs in report["hands"]:
        tid  = hs["table_id"]
        ht1  = hs["total_t1"]
        ht2  = hs["total_t2"]
        m    = hs["max"]
        m2   = hs["max_t2"]
        print(f"\nMão {tid}:  T1={ht1}/{m}  T2={ht2}/{m2}")

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
        pot_detail = f"det=${dp:.2f} gab=${gp:.2f}" if dp else f"não detectado (gab=${gp:.2f})"
        print(f"  final_pot:       {s['final_pot']:2d}/{RUBRIC['final_pot']}  {pot_detail}")

        # winner
        dw = hs["det_winner"] or "(não detectado)"
        gw = hs["gab_winner"]
        ok = "✓" if s["winner"] == RUBRIC["winner"] else f"det={dw} gab={gw}"
        print(f"  winner:          {s['winner']:2d}/{RUBRIC['winner']}  {ok}")

        # Tier 2
        s2 = hs.get("scores_t2", {})
        if any(v > 0 for v in s2.values()):
            print(f"  --- Tier 2 ---")
            for key, max_pts in RUBRIC_T2.items():
                pts = s2.get(key, 0)
                ok = "✓" if pts == max_pts else ("—" if pts > 0 else "✗")
                print(f"  {key:<20} {pts:2d}/{max_pts}  {ok}")


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

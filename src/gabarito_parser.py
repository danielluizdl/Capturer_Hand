"""Task 1 — parser do gabarito.txt em formato PokerStars para HandHistory."""
from __future__ import annotations
import re
from dataclasses import dataclass, field


@dataclass
class HandHistory:
    table_id:       str
    button_seat:    int
    players:        dict            # {seat_num: {"name": str, "chips": float}}
    hole_cards:     list[str]       # ["9s", "4s"]
    board:          list[str]       # ["Jd", "5s", "2h", "3d", "7s"]
    streets:        list[str]       # ["preflop", "flop", "turn", "river"]
    pot_by_street:  dict            # {"flop": 1.65, "turn": 1.65, "river": 2.15}
    total_pot:      float
    winner:         str
    actions:        list[dict]      # [{"street","player","action","amount"}, ...]
    showdown:       dict = field(default_factory=dict)  # {"player": ["Ac","Td"]}


def parse_gabarito(path: str) -> list[HandHistory]:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    parts = re.split(r"(?=PokerStars Hand #)", content)
    parts = [p.strip() for p in parts if p.strip()]

    hands = []
    for text in parts:
        h = _parse_hand(text)
        if h:
            hands.append(h)
    return hands


def _parse_hand(text: str) -> HandHistory | None:
    lines = text.strip().splitlines()

    m = re.search(r"Table '(HL\d+)'", text)
    if not m:
        return None
    table_id = m.group(1)

    m = re.search(r"Seat #(\d+) is the button", text)
    button_seat = int(m.group(1)) if m else 0

    players: dict = {}
    for m in re.finditer(r"Seat (\d+): (.+?) \(\$?([\d.]+) in chips\)", text):
        seat = int(m.group(1))
        name = m.group(2).strip()
        chips = float(m.group(3))
        players[seat] = {"name": name, "chips": chips}

    hole_cards: list[str] = []
    m = re.search(r"Dealt to dLzinN \[(.+?)\]", text)
    if m:
        hole_cards = m.group(1).split()

    board: list[str] = []
    streets: list[str] = ["preflop"]
    pot_by_street: dict = {}

    m = re.search(r"\*\*\* FLOP \*\*\* \[(.+?)\]", text)
    if m:
        board.extend(m.group(1).split())
        streets.append("flop")

    m = re.search(r"\*\*\* TURN \*\*\* \[.*?\] \[(.+?)\]", text)
    if m:
        board.extend(m.group(1).split())
        streets.append("turn")

    m = re.search(r"\*\*\* RIVER \*\*\* \[.*?\] \[.*?\] \[(.+?)\]", text)
    if m:
        board.extend(m.group(1).split())
        streets.append("river")

    total_pot = 0.0
    m = re.search(r"Total pot \$([\d.]+)", text)
    if m:
        total_pot = float(m.group(1))

    winner = ""
    m = re.search(r"(\S+) collected \$[\d.]+ from pot", text)
    if m:
        winner = m.group(1)
    if not winner:
        m = re.search(r"Seat \d+: (\S+).*?collected \(\$[\d.]+\)", text)
        if m:
            winner = m.group(1)

    for s in streets[1:]:
        pot_by_street[s] = total_pot

    actions: list[dict] = []
    showdown: dict = {}
    current_street = "preflop"
    in_showdown = False

    for line in lines:
        sm = re.match(r"\*\*\* (FLOP|TURN|RIVER|SHOW DOWN|SUMMARY)", line)
        if sm:
            tag = sm.group(1)
            if tag == "FLOP":
                current_street = "flop"
                in_showdown = False
            elif tag == "TURN":
                current_street = "turn"
                in_showdown = False
            elif tag == "RIVER":
                current_street = "river"
                in_showdown = False
            elif tag == "SHOW DOWN":
                in_showdown = True
            elif tag == "SUMMARY":
                in_showdown = False
            continue

        if in_showdown:
            sm2 = re.match(r"(.+?): shows \[(.+?)\]", line)
            if sm2:
                player = sm2.group(1).strip()
                cards = sm2.group(2).split()
                showdown[player] = cards
            continue

        am = re.match(r"(.+?): (folds|checks|calls|bets|raises)", line)
        if am:
            player = am.group(1).strip()
            action = am.group(2)
            amt_m = re.search(r"\$([\d.]+)", line)
            amt = float(amt_m.group(1)) if amt_m else 0.0
            actions.append({
                "street": current_street,
                "player": player,
                "action": action,
                "amount": amt,
            })

    return HandHistory(
        table_id=table_id,
        button_seat=button_seat,
        players=players,
        hole_cards=hole_cards,
        board=board,
        streets=streets,
        pot_by_street=pot_by_street,
        total_pot=total_pot,
        winner=winner,
        actions=actions,
        showdown=showdown,
    )

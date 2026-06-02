"""
Compila anotações em:
  1. Dataset CNN expandido  →  training_data/<carta>/<imagem>.png
  2. Gabarito completo      →  gabarito_longo.txt (formato PokerStars)

Uso:
    python compile_training_data.py
    python compile_training_data.py --dir key_frames --out training_data
    python compile_training_data.py --gabarito-only   # só gera o gabarito

Pré-requisito: anotações salvas por annotate_frames.py
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import cv2

from src.card_classifier import SLOT_X, CARD_Y1_F, CARD_Y2_F, CLASSIFY_HW_F

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RANKS = list("23456789tjqka")
SUITS = list("cdhs")
ALL_CARDS = sorted([r + s for r in RANKS for s in SUITS])


def _fmt_card(c: str) -> str:
    """'ah' → 'Ah', '9c' → '9c'"""
    return c[0].upper() + c[1]


def _slot_crops(crop_path: Path, n_cards: int) -> list[tuple[str, object]]:
    """
    Retorna [(slot_name, crop_img)] lendo os arquivos *_slot_N.png
    que foram salvos junto com o frame principal.
    """
    result = []
    for i in range(n_cards):
        slot_file = crop_path.parent / (crop_path.stem + f"_slot_{i}.png")
        if slot_file.exists():
            img = cv2.imread(str(slot_file))
            result.append((f"slot_{i}", img))
    return result


# ---------------------------------------------------------------------------
# Extrai crops de cartas das anotações
# ---------------------------------------------------------------------------

def _extract_card_crops(
    key_frames_dir: Path,
    out_dir: Path,
) -> dict[str, int]:
    """
    Para cada mão anotada, extrai os crops de cartas individuais e
    os salva em out_dir/<carta>/<source>.png.

    Retorna contagem de exemplos por carta.
    """
    counts: dict[str, int] = defaultdict(int)

    for ann_path in sorted(key_frames_dir.rglob("annotations.json")):
        ann = json.loads(ann_path.read_text())
        if not ann.get("annotated"):
            continue

        hand_dir  = ann_path.parent
        table_id  = ann["table_id"]
        hand_idx  = ann["hand_idx"]
        prefix    = f"{table_id}_h{hand_idx:03d}"

        board_ann = ann.get("board", {})

        # ----- Board cards -----
        street_n = {"flop": 3, "turn": 4, "river": 5}
        for street, n in street_n.items():
            cards = board_ann.get(street, [])
            if not cards:
                continue

            street_file = hand_dir / f"{street}.png"
            if not street_file.exists():
                continue

            # Usa os crops de slots salvos pelo extractor
            slot_data = _slot_crops(street_file, n)
            new_slots  = len(cards) - (n - len(cards) + len(cards))  # cartas novas nesta street
            # Só a última carta(s) são novas; mas salvamos todas para robustez
            for i, (slot_name, img) in enumerate(slot_data):
                if img is None or i >= len(cards):
                    continue
                card = cards[i].lower()
                if card not in ALL_CARDS:
                    continue
                card_dir = out_dir / card
                card_dir.mkdir(parents=True, exist_ok=True)
                out_name = f"board_{prefix}_{street}_{slot_name}.png"
                cv2.imwrite(str(card_dir / out_name), img)
                counts[card] += 1

        # ----- Hero hole cards -----
        hero_cards = ann.get("hero_hole_cards", [])
        preflop_file = hand_dir / "preflop.png"
        if hero_cards and preflop_file.exists():
            # Hero aparece no crop em posição fixa — reusa detect_suit region
            crop = cv2.imread(str(preflop_file))
            if crop is not None:
                h, w = crop.shape[:2]
                # Reusa constantes de card_classifier para posição do hero
                from src.card_classifier import (
                    HERO_Y1_F, HERO_Y2_F,
                    HERO_LEFT_X1_F, HERO_LEFT_X2_F,
                    HERO_RIGHT_X1_F, HERO_RIGHT_X2_F,
                )
                y1 = int(h * HERO_Y1_F); y2 = int(h * HERO_Y2_F)
                for ci, (x1f, x2f) in enumerate([
                    (HERO_LEFT_X1_F,  HERO_LEFT_X2_F),
                    (HERO_RIGHT_X1_F, HERO_RIGHT_X2_F),
                ]):
                    if ci >= len(hero_cards):
                        break
                    x1 = int(w * x1f); x2 = int(w * x2f)
                    region = crop[y1:y2, x1:x2]
                    card = hero_cards[ci].lower()
                    if card not in ALL_CARDS or region.size == 0:
                        continue
                    card_dir = out_dir / card
                    card_dir.mkdir(parents=True, exist_ok=True)
                    out_name = f"hero_{prefix}_card{ci}.png"
                    cv2.imwrite(str(card_dir / out_name), region)
                    counts[card] += 1

        # ----- Showdown cards -----
        showdown = ann.get("showdown", {})
        if showdown:
            # Encontra o melhor frame de showdown (mais pares visíveis)
            sd_files = sorted(hand_dir.glob("showdown_*.png"))
            if sd_files:
                # Usa o primeiro frame de showdown como referência
                # (os crops de villain são feitos pela busca full-frame)
                for player, cards in showdown.items():
                    for ci, card in enumerate(cards):
                        card = card.lower()
                        if card not in ALL_CARDS:
                            continue
                        # Salva referência sem crop específico
                        # (o crop será feito pelo CNN treino a partir do template matching)
                        # Por ora, apenas registra a existência para o gabarito
                        counts[f"showdown_{card}"] += 0  # placeholder

    return dict(counts)


# ---------------------------------------------------------------------------
# Gera gabarito no formato PokerStars
# ---------------------------------------------------------------------------

_SB   = 0.05
_BB   = 0.10
_STR  = 0.20
_ANTE = 0.05
HERO  = "dLzinN"


def _build_hand_text(ann: dict, hand_id: int) -> str | None:
    """Constrói uma mão no formato PokerStars a partir de uma anotação."""
    table_id = ann.get("table_id", "UNKNOWN")
    board_ann = ann.get("board", {})
    flop  = [_fmt_card(c) for c in board_ann.get("flop",  [])]
    turn  = [_fmt_card(c) for c in board_ann.get("turn",  [])]
    river = [_fmt_card(c) for c in board_ann.get("river", [])]
    board = flop + turn + river

    hero_cards = [_fmt_card(c) for c in ann.get("hero_hole_cards", [])]
    showdown   = {p: [_fmt_card(c) for c in cards]
                  for p, cards in ann.get("showdown", {}).items()}
    winner     = ann.get("winner", "")
    pot_bb     = ann.get("pot_bb")
    pot_usd    = round(pot_bb * _BB, 2) if pot_bb else 0.0

    lines = []

    # Header
    lines.append(
        f"PokerStars Hand #{hand_id}: "
        f"Hold'em No Limit (${_SB}/${_BB}/${_STR}({_ANTE})) - "
        f"2026/01/01 00:00:00 ET"
    )
    lines.append(f"Table '{table_id}' 8-max Seat #1 is the button")

    # Hole cards
    lines.append("*** HOLE CARDS ***")
    if hero_cards:
        lines.append(f"Dealt to {HERO} [{' '.join(hero_cards)}]")

    # Streets
    if flop:
        lines.append(f"*** FLOP *** [{' '.join(flop)}]")
    if turn:
        lines.append(f"*** TURN *** [{' '.join(flop)}] [{' '.join(turn)}]")
    if river:
        lines.append(f"*** RIVER *** [{' '.join(flop)}] [{' '.join(turn)}] [{' '.join(river)}]")

    # Showdown
    if showdown:
        lines.append("*** SHOW DOWN ***")
        for player, cards in showdown.items():
            lines.append(f"{player}: shows [{' '.join(cards)}]")

    # Winner
    if winner and pot_usd > 0:
        lines.append(f"{winner} collected ${pot_usd:.2f} from pot")

    # Summary
    lines.append("*** SUMMARY ***")
    lines.append(f"Total pot ${pot_usd:.2f} | Rake $0.00")
    if board:
        lines.append(f"Board [{' '.join(board)}]")

    return "\n".join(lines)


def build_gabarito(key_frames_dir: Path, out_path: Path):
    """Gera gabarito_longo.txt com todas as mãos anotadas."""
    hands_text = []
    hand_id = 200001

    # Coleta todas as anotações, ordenadas por mesa e mão
    for ann_path in sorted(key_frames_dir.rglob("annotations.json")):
        ann = json.loads(ann_path.read_text())
        if not ann.get("annotated"):
            continue

        # Só inclui mãos com pelo menos board ou hole cards
        board = ann.get("board", {})
        has_content = (
            board.get("flop") or
            ann.get("hero_hole_cards") or
            ann.get("showdown") or
            ann.get("winner")
        )
        if not has_content:
            continue

        text = _build_hand_text(ann, hand_id)
        if text:
            hands_text.append(text)
            hand_id += 1

    if not hands_text:
        print("Nenhuma mão anotada encontrada.")
        return

    out_path.write_text("\n\n".join(hands_text) + "\n", encoding="utf-8")
    print(f"Gabarito gerado: {out_path}  ({len(hands_text)} mãos)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def compile_all(key_frames_dir: Path, train_out: Path, gabarito_out: Path,
                gabarito_only: bool = False):
    print(f"\n{'='*60}")
    print(f"  Compilação de dados de treinamento")
    print(f"  Anotações: {key_frames_dir}")
    print(f"{'='*60}\n")

    if not gabarito_only:
        print("1/2 — Extraindo crops de cartas para CNN...")
        counts = _extract_card_crops(key_frames_dir, train_out)

        total = sum(v for k, v in counts.items() if not k.startswith("showdown_"))
        print(f"  Total de novos exemplos CNN: {total}")
        if counts:
            sorted_counts = sorted(
                [(k, v) for k, v in counts.items() if not k.startswith("showdown_")],
                key=lambda x: x[1]
            )
            print(f"  Mínimo por carta: {sorted_counts[0]} | Máximo: {sorted_counts[-1]}")

        print(f"\n  Para retreinar a CNN com os novos dados:")
        print(f"  python train_card_classifier.py --extra {train_out}")

    print("\n2/2 — Gerando gabarito...")
    build_gabarito(key_frames_dir, gabarito_out)

    # Conta anotações
    total_ann = sum(1 for _ in key_frames_dir.rglob("annotations.json")
                    if json.loads(_.read_text()).get("annotated"))
    print(f"\nTotal de mãos anotadas: {total_ann}")
    print(f"\nPróximos passos:")
    if not gabarito_only:
        print(f"  1. Retreinar CNN: python train_card_classifier.py --extra {train_out}")
    print(f"  2. Validar pipeline: python agent_loop.py --gabarito {gabarito_out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Compila anotações em training data + gabarito")
    ap.add_argument("--dir",           default="key_frames",       help="Pasta de anotações")
    ap.add_argument("--out",           default="training_data",    help="Pasta de saída CNN")
    ap.add_argument("--gabarito",      default="gabarito_longo.txt", help="Arquivo gabarito de saída")
    ap.add_argument("--gabarito-only", action="store_true",        help="Só gera o gabarito")
    args = ap.parse_args()

    compile_all(
        key_frames_dir=Path(args.dir),
        train_out=Path(args.out),
        gabarito_out=Path(args.gabarito),
        gabarito_only=args.gabarito_only,
    )

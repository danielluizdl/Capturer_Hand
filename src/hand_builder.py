"""Task 5 — converte TableEvent em HandHistory detectado."""
from __future__ import annotations
import cv2
import numpy as np
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from src.gabarito_parser import HandHistory
from src.video_pipeline import TableEvent
from src.detectors import crop_table
from src.ocr_engine import ocr_title_bar, ocr_pot, ocr_board_cards, ocr_hole_cards, ocr_winner
from src.card_classifier import extract_board_cards_slotted

# 1 BB = $0.10 nas stakes $0.05/$0.10/$0.20 do vídeo de teste
BB_TO_USD = 0.10


def build_hands(
    events: dict[int, list[TableEvent]],
    video_path: str,
) -> list[HandHistory]:
    """
    Converte eventos do pipeline em objetos HandHistory detectados.
    Para cada mesa, detecta a mão com mais streets (a do gabarito).
    """
    result: list[HandHistory] = []

    for tid, ev_list in events.items():
        if not ev_list:
            continue

        # Segmenta em mãos separadas
        hand_segs = _segment_hands(ev_list)

        # Escolhe a mão mais completa (maior nro de streets)
        best_seg = _pick_best_hand(hand_segs)
        if not best_seg:
            continue

        hand = _build_hand_for_segment(tid, best_seg, ev_list, video_path)
        if hand:
            result.append(hand)

    return result


def _segment_hands(ev_list: list[TableEvent]) -> list[list[TableEvent]]:
    """
    Divide eventos em mãos usando transições board>0 → board=0 como fronteira.
    Cada segmento contém todos os eventos de uma mão.
    """
    sorted_evs = sorted(ev_list, key=lambda e: e.timestamp)
    if not sorted_evs:
        return []

    hands: list[list[TableEvent]] = []
    current: list[TableEvent] = []
    prev_board = -1

    for ev in sorted_evs:
        if prev_board > 0 and ev.board_cards == 0 and ev.event_type == "new_hand":
            # Fim de mão: fecha segmento atual
            if current:
                hands.append(current)
            current = []
        current.append(ev)
        prev_board = ev.board_cards

    if current:
        hands.append(current)

    return hands


def _pick_best_hand(segs: list[list[TableEvent]]) -> list[TableEvent] | None:
    """
    Escolhe o segmento com mais transições de board (= mão mais completa).
    Prefere mãos com flop (board=3 presente).
    """
    if not segs:
        return None

    best = None
    best_score = -1

    for seg in segs:
        boards_seen = set(e.board_cards for e in seg)
        # Penaliza segmentos sem flop
        has_flop  = 3 in boards_seen
        has_turn  = 4 in boards_seen
        has_river = 5 in boards_seen
        score = (has_flop * 10) + (has_turn * 5) + (has_river * 3) + len(seg)
        if score > best_score:
            best_score = score
            best = seg

    return best


def _get_frame(video_path: str, frame_idx: int) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def _build_hand_for_segment(
    tid: int,
    seg: list[TableEvent],
    all_evs: list[TableEvent],
    video_path: str,
) -> Optional[HandHistory]:
    """Constrói HandHistory para um segmento de eventos de uma mão."""

    table_id = seg[0].table_id

    # Confirma table_id via OCR no primeiro frame do segmento
    frame0 = _get_frame(video_path, seg[0].frame_idx)
    if frame0 is not None:
        crop0 = crop_table(frame0, tid)
        detected_id = ocr_title_bar(crop0)
        if detected_id:
            table_id = detected_id

    # Streets a partir das transições de board no segmento
    streets = _detect_streets(seg)

    # Board cards via OCR nos frames de transição
    board = _extract_board(tid, seg, video_path)

    # Pot final: max OCR dentro do tempo da mão
    total_pot = _extract_final_pot(tid, seg, video_path)

    # Hole cards
    hole_cards = _extract_hole_cards(tid, seg, video_path)

    # Vencedor
    winner = _extract_winner(tid, seg, all_evs, video_path)

    pot_by_street = {s: total_pot for s in streets[1:] if total_pot}

    return HandHistory(
        table_id=table_id,
        button_seat=0,
        players={},
        hole_cards=hole_cards,
        board=board,
        streets=streets,
        pot_by_street=pot_by_street,
        total_pot=total_pot or 0.0,
        winner=winner or "",
        actions=[],
    )


def _detect_streets(seg: list[TableEvent]) -> list[str]:
    streets = ["preflop"]
    seen: set[str] = set()
    for ev in sorted(seg, key=lambda e: e.timestamp):
        n = ev.board_cards
        if n == 3 and "flop" not in seen:
            streets.append("flop")
            seen.add("flop")
        elif n == 4 and "turn" not in seen and "flop" in seen:
            streets.append("turn")
            seen.add("turn")
        elif n == 5 and "river" not in seen and "turn" in seen:
            streets.append("river")
            seen.add("river")
    return streets


def _vote_slots_for_event(
    ev,
    n: int,
    tid: int,
    video_path: str,
    extra_frames: list[int],
) -> tuple[list[str | None], int]:
    """
    Vota por maioria em cada slot para um board_change event.
    Retorna (slotted_result, new_card_votes) onde:
      - slotted_result[i] é None se o slot não teve votos
      - new_card_votes = votos da carta vencedora no último slot (novo card)
    """
    slot_votes: list[Counter] = [Counter() for _ in range(n)]

    for offset in extra_frames:
        fi = ev.frame_idx + offset
        frame = _get_frame(video_path, fi)
        if frame is None:
            continue
        crop = crop_table(frame, tid)
        slotted = extract_board_cards_slotted(crop, n)
        for i, card in enumerate(slotted):
            if card is not None:
                slot_votes[i][card] += 1

    result: list[str | None] = []
    for i in range(n):
        if slot_votes[i]:
            best = slot_votes[i].most_common(1)[0][0]
            result.append(best)
        else:
            result.append(None)

    # Usa votos do último slot (nova carta) como indicador de confiança
    new_card_votes = slot_votes[n - 1].most_common(1)[0][1] if slot_votes[n - 1] else 0
    return result, new_card_votes


def _extract_board(
    tid: int,
    seg: list[TableEvent],
    video_path: str,
) -> list[str]:
    """
    Extrai cartas do board via classificador cor+OCR.
    Estratégias:
    1. Votação por slot independente (não requer todos n cartas por frame).
    2. Tenta TODOS os board_change events por street; usa o de maior confiança
       no último slot (a nova carta adicionada naquela street).
    3. Preenche slots ausentes de streets anteriores (ex: Qs do flop no turn).
    """
    EXTRA_FRAMES = [0, 3, 6, 9, 15]

    board_changes = sorted(
        [e for e in seg if e.event_type == "board_change"],
        key=lambda e: e.timestamp,
    )

    # Para cada street, guarda (slotted_result, new_card_votes) do melhor evento
    street_best: dict[str, tuple[list[str | None], int]] = {}

    for ev in board_changes:
        n = ev.board_cards
        street = {3: "flop", 4: "turn", 5: "river"}.get(n)
        if not street:
            continue

        slotted, new_card_votes = _vote_slots_for_event(ev, n, tid, video_path, EXTRA_FRAMES)
        prev_votes = street_best.get(street, (None, -1))[1]
        if new_card_votes > prev_votes:
            street_best[street] = (slotted, new_card_votes)

    # flop: slots 0-2
    flop_cards: list[str | None] = [None, None, None]
    if "flop" in street_best:
        flop_cards = street_best["flop"][0][:3]

    # turn: slot 3; herda flop para slots 0-2 ausentes
    turn_card: str | None = None
    if "turn" in street_best:
        slotted4 = street_best["turn"][0]
        turn_card = slotted4[3] if len(slotted4) > 3 else None
        # herança: flop cards em slots ausentes
        for i in range(3):
            if flop_cards[i] is None and slotted4[i] is not None:
                flop_cards[i] = slotted4[i]

    # river: slot 4; herda flop/turn para slots 0-3 ausentes
    river_card: str | None = None
    if "river" in street_best:
        slotted5 = street_best["river"][0]
        river_card = slotted5[4] if len(slotted5) > 4 else None
        for i in range(3):
            if flop_cards[i] is None and slotted5[i] is not None:
                flop_cards[i] = slotted5[i]
        if turn_card is None and len(slotted5) > 3 and slotted5[3] is not None:
            turn_card = slotted5[3]

    # Monta board final
    board: list[str] = [c for c in flop_cards if c]
    if turn_card:
        board.append(turn_card)
    if river_card:
        board.append(river_card)

    return board


def _extract_final_pot(
    tid: int,
    seg: list[TableEvent],
    video_path: str,
) -> Optional[float]:
    """
    Extrai o pot final em USD.
    OCR retorna BB → multiplica por BB_TO_USD.
    Usa frames no FINAL da mão (maior pot = pot na última street ativa).
    """
    # Pega os últimos N eventos com board > 0 (final da mão antes de resetar)
    active = sorted([e for e in seg if e.board_cards > 0], key=lambda e: e.timestamp)

    # Tenta a partir do fim (mais provável de ter o pot final)
    candidates = active[-5:] if len(active) >= 5 else active
    candidates = list(reversed(candidates))  # do mais recente ao mais antigo

    pots: list[float] = []
    for ev in candidates[:5]:
        frame = _get_frame(video_path, ev.frame_idx)
        if frame is None:
            continue
        crop = crop_table(frame, tid)
        pot_bb = ocr_pot(crop)
        if pot_bb and pot_bb > 0:
            pots.append(pot_bb * BB_TO_USD)

    return max(pots) if pots else None


def _extract_hole_cards(
    tid: int,
    seg: list[TableEvent],
    video_path: str,
) -> list[str]:
    """
    Tenta extrair hole cards do herói.
    Estratégia: pré-flop (board==0) primeiro, depois flop/turn.
    """
    # Candidatos pré-flop (cartas recém-distribuídas)
    preflop = sorted(
        [e for e in seg if e.board_cards == 0],
        key=lambda e: e.timestamp,
    )[:5]
    # Candidatos flop/turn (herói ainda na mão)
    early_board = sorted(
        [e for e in seg if e.board_cards in (3, 4)],
        key=lambda e: e.timestamp,
    )[:3]

    for ev in preflop + early_board:
        frame = _get_frame(video_path, ev.frame_idx)
        if frame is None:
            continue
        crop = crop_table(frame, tid)
        cards = ocr_hole_cards(crop)
        if len(cards) >= 2:
            return cards[:2]

    return []


def _extract_winner(
    tid: int,
    seg: list[TableEvent],
    all_evs: list[TableEvent],
    video_path: str,
) -> Optional[str]:
    """
    Detecta o vencedor buscando o badge 'WINNER'.
    Usa next_new_hand de all_evs para cobrir o "dead zone" após o último evento do segmento.
    """
    cap_info = cv2.VideoCapture(video_path)
    native_fps = cap_info.get(cv2.CAP_PROP_FPS)
    cap_info.release()

    seg_end_ts = max(e.timestamp for e in seg)

    # Próximo new_hand APÓS o fim do segmento (em qualquer mesa)
    future_new_hands = sorted(
        [e for e in all_evs if e.table_idx == tid and e.event_type == "new_hand" and e.timestamp > seg_end_ts],
        key=lambda e: e.timestamp,
    )
    if future_new_hands:
        next_new_hand_ts = future_new_hands[0].timestamp
    else:
        next_new_hand_ts = seg_end_ts + 20.0

    search_start = max(0.0, seg_end_ts - 3.0)
    search_end = min(next_new_hand_ts + 3.0, seg_end_ts + 30.0)
    step_s = 1.0

    t = search_start
    while t <= search_end:
        fi = int(t * native_fps)
        frame = _get_frame(video_path, fi)
        if frame is not None:
            crop = crop_table(frame, tid)
            winner = ocr_winner(crop)
            if winner:
                return winner
        t += step_s

    return None

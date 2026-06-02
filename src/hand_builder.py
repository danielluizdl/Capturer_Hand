"""Task 5 — converte TableEvent em HandHistory detectado."""
from __future__ import annotations
import cv2
import multiprocessing
import numpy as np
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from src.gabarito_parser import HandHistory
from src.video_pipeline import TableEvent
from src.detectors import crop_table, detect_dealer_button
from src.ocr_engine import ocr_title_bar, ocr_pot, ocr_board_cards, ocr_hole_cards, ocr_winner
from src.card_classifier import extract_board_cards_slotted
from src.seat_card_recognizer import find_showdown_cards

# 1 BB = $0.10 nas stakes $0.05/$0.10/$0.20 do vídeo de teste
BB_TO_USD = 0.10


def _read_and_crop_all_tables(
    video_path: str,
    per_table_needed: dict[int, set[int]],
) -> dict[int, dict[int, np.ndarray]]:
    """
    Faz UMA passagem sequencial no vídeo e coleta crops por mesa.
    per_table_needed[tid] = set de frame_indices necessários para a mesa tid.
    Retorna per_table_crops[tid][frame_idx] = crop_ndarray.
    """
    all_indices = sorted(set().union(*per_table_needed.values()))
    if not all_indices:
        return {tid: {} for tid in per_table_needed}

    per_table_crops: dict[int, dict[int, np.ndarray]] = {tid: {} for tid in per_table_needed}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return per_table_crops

    current = 0
    for target in all_indices:
        if target < current:
            continue
        while current < target:
            cap.grab()
            current += 1
        ret, frame = cap.read()
        if ret:
            for tid, needed in per_table_needed.items():
                if target in needed:
                    per_table_crops[tid][target] = crop_table(frame, tid)
        current += 1

    cap.release()
    return per_table_crops


def _collect_frame_indices(
    tid: int,
    hand_segs: list[list[TableEvent]],
    all_evs: list[TableEvent],
    native_fps: float,
) -> set[int]:
    EXTRA_FRAMES = [0, 3, 6, 9, 15]
    indices: set[int] = set()
    for seg in hand_segs:
        sorted_seg = sorted(seg, key=lambda e: e.timestamp)
        if sorted_seg:
            indices.add(sorted_seg[0].frame_idx)
        for ev in seg:
            if ev.event_type == "board_change" and ev.board_cards in (3, 4, 5):
                for offset in EXTRA_FRAMES:
                    indices.add(ev.frame_idx + offset)
        active = sorted([e for e in seg if e.board_cards > 0], key=lambda e: e.timestamp)
        for ev in (active[-5:] if len(active) >= 5 else active):
            indices.add(ev.frame_idx)
        preflop = sorted([e for e in seg if e.board_cards == 0], key=lambda e: e.timestamp)[:5]
        early_board = sorted([e for e in seg if e.board_cards in (3, 4)], key=lambda e: e.timestamp)[:3]
        for ev in preflop + early_board:
            indices.add(ev.frame_idx)
        if not seg:
            continue
        seg_end_ts = max(e.timestamp for e in seg)
        future_new_hands = sorted(
            [e for e in all_evs if e.table_idx == tid and e.event_type == "new_hand" and e.timestamp > seg_end_ts],
            key=lambda e: e.timestamp,
        )
        next_new_hand_ts = future_new_hands[0].timestamp if future_new_hands else seg_end_ts + 20.0
        search_start = max(0.0, seg_end_ts - 3.0)
        search_end = min(next_new_hand_ts + 3.0, seg_end_ts + 30.0)
        t = search_start
        while t <= search_end:
            indices.add(int(t * native_fps))
            t += 1.0
    return indices


def _process_table_ocr_worker(args: tuple) -> list:
    """Worker que processa UMA mesa usando crops já extraídos (sem I/O de vídeo)."""
    tid, ev_list, all_evs, frames_cache, native_fps = args
    if not ev_list:
        return []
    hand_segs = _segment_hands(ev_list)
    hand_segs = sorted(hand_segs, key=_seg_score)
    result = []
    for seg in hand_segs:
        hand = _build_hand_for_segment(tid, seg, all_evs, frames_cache, native_fps)
        if hand:
            result.append(hand)
    return result


def _seg_score(seg: list[TableEvent]) -> int:
    boards = {e.board_cards for e in seg}
    return (3 in boards) * 10 + (4 in boards) * 5 + (5 in boards) * 3 + len(seg)


def build_hands(
    events: dict[int, list[TableEvent]],
    video_path: str,
) -> list[HandHistory]:
    """
    Por mesa: 1 passagem sequencial no vídeo + OCR imediato.
    Processa uma mesa por vez para limitar uso de RAM em vídeos longos.
    """
    cap_info = cv2.VideoCapture(video_path)
    native_fps = cap_info.get(cv2.CAP_PROP_FPS)
    cap_info.release()

    all_evs = [ev for ev_list in events.values() for ev in ev_list]

    # Fase 1: segmentos e frames necessários por mesa
    per_table_segs: dict[int, list] = {}
    per_table_needed: dict[int, set[int]] = {}
    for tid, ev_list in events.items():
        if not ev_list:
            continue
        segs = sorted(_segment_hands(ev_list), key=_seg_score)
        new_hand_evs = sum(1 for e in ev_list if e.event_type == "new_hand")
        print(f"  [mesa {tid}] eventos={len(ev_list)} new_hand={new_hand_evs} segmentos={len(segs)}", flush=True)
        per_table_segs[tid] = segs
        per_table_needed[tid] = _collect_frame_indices(tid, segs, all_evs, native_fps)

    if not per_table_needed:
        return []

    total_crops = sum(len(v) for v in per_table_needed.values())
    print(f"  Total: {total_crops} crops em {len(per_table_needed)} mesas (processando 1 mesa por vez)")

    # Fase 2+3: uma mesa por vez — leitura + OCR sequencial (controla RAM)
    result: list[HandHistory] = []
    for i, tid in enumerate(per_table_segs, 1):
        n = len(per_table_needed[tid])
        print(f"  [mesa {tid}] {n} frames (pass {i}/{len(per_table_segs)})...", flush=True)
        crops = _read_and_crop_all_tables(video_path, {tid: per_table_needed[tid]})[tid]
        hands = _process_table_ocr_worker((tid, events[tid], all_evs, crops, native_fps))
        result.extend(hands)
        del crops  # libera RAM antes da próxima mesa

    return result


def _segment_hands(ev_list: list[TableEvent]) -> list[list[TableEvent]]:
    """
    Divide eventos em mãos usando múltiplos critérios de fronteira:
    1. Transição board>0 → board=0 com new_hand event (critério original)
    2. new_hand event com gap de tempo >= 5s (cobre preflop-only hands)
    3. Gap de silêncio >= 20s entre eventos consecutivos (mãos perdidas)
    """
    sorted_evs = sorted(ev_list, key=lambda e: e.timestamp)
    if not sorted_evs:
        return []

    hands: list[list[TableEvent]] = []
    current: list[TableEvent] = []
    prev_board = -1
    prev_ts = sorted_evs[0].timestamp

    for ev in sorted_evs:
        time_gap = ev.timestamp - prev_ts
        is_new_hand = ev.event_type == "new_hand"

        # Critério 1: board voltou a 0 com new_hand (fim claro de mão)
        if prev_board > 0 and ev.board_cards == 0 and is_new_hand:
            if current:
                hands.append(current)
            current = []

        # Critério 2: new_hand com gap de tempo significativo (>=5s)
        # — captura preflop folds onde board nunca avançou
        elif is_new_hand and time_gap >= 5.0 and current:
            has_board = any(e.board_cards > 0 for e in current)
            # Se segmento atual já tem board, é uma nova mão clara
            # Se não tem board mas o gap é suficiente, provavelmente é nova mão
            if has_board or time_gap >= 10.0:
                hands.append(current)
                current = []

        # Critério 3: gap de silêncio longo (>=20s) — provável nova mão perdida
        elif time_gap >= 20.0 and current:
            hands.append(current)
            current = []

        current.append(ev)
        prev_board = ev.board_cards
        prev_ts = ev.timestamp

    if current:
        hands.append(current)

    # Filtra segmentos vazios ou com apenas 1 evento sem board
    return [h for h in hands if len(h) > 1 or any(e.board_cards > 0 for e in h)]


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
    frames_cache: dict[int, np.ndarray],
    native_fps: float,
) -> Optional[HandHistory]:
    """Constrói HandHistory para um segmento de eventos de uma mão.
    frames_cache[frame_idx] = crop já recortado para a mesa tid.
    """

    table_id = seg[0].table_id

    # Confirma table_id via OCR no primeiro frame do segmento
    crop0 = frames_cache.get(seg[0].frame_idx)
    if crop0 is not None:
        detected_id = ocr_title_bar(crop0)
        if detected_id:
            table_id = detected_id

    # Streets a partir das transições de board no segmento
    streets = _detect_streets(seg)

    # Board cards via OCR nos frames de transição
    board = _extract_board(tid, seg, frames_cache)

    # Pot final: max OCR dentro do tempo da mão
    total_pot = _extract_final_pot(tid, seg, frames_cache)

    # Hole cards
    hole_cards = _extract_hole_cards(tid, seg, frames_cache)

    # Vencedor
    winner = _extract_winner(tid, seg, all_evs, frames_cache, native_fps)

    # Hole cards dos opponents no showdown via template matching
    showdown = _extract_showdown_cards(tid, seg, all_evs, frames_cache, native_fps)

    # Dealer button: tenta detectar via template nos frames preflop
    button_seat = _detect_button_seat(seg, frames_cache)

    pot_by_street = {s: total_pot for s in streets[1:] if total_pot}

    return HandHistory(
        table_id=table_id,
        button_seat=button_seat,
        players={},
        hole_cards=hole_cards,
        board=board,
        streets=streets,
        pot_by_street=pot_by_street,
        total_pot=total_pot or 0.0,
        winner=winner or "",
        actions=[],
        showdown=showdown,
    )


def _detect_button_seat(
    seg: list[TableEvent],
    frames_cache: dict[int, np.ndarray],
) -> int:
    """
    Detecta o seat do dealer button via template matching (dealer.PNG).
    Mapeia posição (cx, cy) do dealer puck para seat number (1-8, 8-max).

    Layout WPT Global 8-max (frações aproximadas do crop 960x540):
      Seat 1 (CO/BTN area): cx≈0.75, cy≈0.45
      Seat 2 (BTN/SB area): cx≈0.60, cy≈0.80
      Seat 3 (SB/BB area):  cx≈0.40, cy≈0.85
      Seat 4 (BB/STR area): cx≈0.22, cy≈0.70
      Seat 5 (UTG area):    cx≈0.12, cy≈0.45
      Seat 6 (MP area):     cx≈0.20, cy≈0.25
      Seat 7 (HJ area):     cx≈0.40, cy≈0.15
      Seat 8 (CO area):     cx≈0.62, cy≈0.18
    Retorna 0 se não detectado.
    """
    # Zonas aproximadas de cada seat: (cx_min, cx_max, cy_min, cy_max, seat)
    SEAT_ZONES = [
        (0.60, 0.90, 0.35, 0.60, 1),  # direita-meio
        (0.50, 0.75, 0.70, 0.95, 2),  # direita-baixo
        (0.28, 0.55, 0.75, 0.95, 3),  # centro-baixo
        (0.10, 0.35, 0.55, 0.85, 4),  # esquerda-baixo
        (0.02, 0.22, 0.35, 0.60, 5),  # esquerda-meio
        (0.08, 0.35, 0.10, 0.35, 6),  # esquerda-cima
        (0.30, 0.55, 0.05, 0.28, 7),  # centro-cima
        (0.50, 0.78, 0.05, 0.28, 8),  # direita-cima
    ]

    # Usa frames preflop (sem board) para detectar o dealer
    preflop = sorted(
        [e for e in seg if e.board_cards == 0],
        key=lambda e: e.timestamp,
    )[:5]

    for ev in preflop:
        crop = frames_cache.get(ev.frame_idx)
        if crop is None:
            continue
        pos = detect_dealer_button(crop)
        if pos is None:
            continue
        cx, cy = pos
        for cx_min, cx_max, cy_min, cy_max, seat in SEAT_ZONES:
            if cx_min <= cx <= cx_max and cy_min <= cy <= cy_max:
                return seat

    return 0


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
    frames_cache: dict[int, np.ndarray],
    extra_frames: list[int],
) -> tuple[list[str | None], int]:
    """
    Vota por maioria em cada slot para um board_change event.
    Retorna (slotted_result, new_card_votes) onde:
      - slotted_result[i] é None se o slot não teve votos suficientes
      - new_card_votes = votos da carta vencedora no último slot (novo card)
    frames_cache[frame_idx] = crop já recortado para a mesa tid.

    Rejeita cartas que aparecem em apenas 1 frame quando há >= 3 frames votando
    (filtro de outlier para reduzir falsos positivos do OCR).
    """
    slot_votes: list[Counter] = [Counter() for _ in range(n)]
    frames_tried = 0

    for offset in extra_frames:
        fi = ev.frame_idx + offset
        crop = frames_cache.get(fi)
        if crop is None:
            continue
        frames_tried += 1
        slotted = extract_board_cards_slotted(crop, n)
        for i, card in enumerate(slotted):
            if card is not None:
                slot_votes[i][card] += 1

    result: list[str | None] = []
    for i in range(n):
        if not slot_votes[i]:
            result.append(None)
            continue
        top = slot_votes[i].most_common(2)
        best_card, best_votes = top[0]
        # Rejeita se unanimidade mínima não atingida com múltiplos frames
        # (evita aceitar leitura única espúria)
        min_votes = 2 if frames_tried >= 3 else 1
        if best_votes >= min_votes:
            result.append(best_card)
        else:
            result.append(None)

    # Usa votos do último slot (nova carta) como indicador de confiança
    new_card_votes = slot_votes[n - 1].most_common(1)[0][1] if slot_votes[n - 1] else 0
    return result, new_card_votes


def _extract_board(
    tid: int,
    seg: list[TableEvent],
    frames_cache: dict[int, np.ndarray],
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

        slotted, new_card_votes = _vote_slots_for_event(ev, n, tid, frames_cache, EXTRA_FRAMES)
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
    frames_cache: dict[int, np.ndarray],
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
        crop = frames_cache.get(ev.frame_idx)
        if crop is None:
            continue
        pot_bb = ocr_pot(crop)
        if pot_bb and pot_bb > 0:
            pots.append(pot_bb * BB_TO_USD)

    return max(pots) if pots else None


def _extract_hole_cards(
    tid: int,
    seg: list[TableEvent],
    frames_cache: dict[int, np.ndarray],
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
        crop = frames_cache.get(ev.frame_idx)
        if crop is None:
            continue
        cards = ocr_hole_cards(crop)
        if len(cards) >= 2:
            return cards[:2]

    return []


def _extract_showdown_cards(
    tid: int,
    seg: list[TableEvent],
    all_evs: list[TableEvent],
    frames_cache: dict[int, np.ndarray],
    native_fps: float,
) -> dict[str, list[str]]:
    """
    Detecta hole cards dos opponents exibidas no showdown via template matching.

    Busca os templates no frame inteiro — SEM calibracao de seats.
    Vota por maioria entre os frames da janela pos-mao para robustez.

    Retorna {"player_1": ["Ah", "Ks"], "player_2": ["9c", "Qd"], ...}
    onde a chave e "player_N" pela ordem espacial (cima->baixo, esq->dir).
    """
    seg_end_ts = max(e.timestamp for e in seg)

    future_new_hands = sorted(
        [e for e in all_evs if e.table_idx == tid
         and e.event_type == "new_hand" and e.timestamp > seg_end_ts],
        key=lambda e: e.timestamp,
    )
    next_new_hand_ts = (
        future_new_hands[0].timestamp if future_new_hands else seg_end_ts + 20.0
    )

    search_start = max(0.0, seg_end_ts - 2.0)
    search_end   = min(next_new_hand_ts + 2.0, seg_end_ts + 25.0)

    # Acumula votos por posicao espacial aproximada
    # Chave: (x//50, y//50) — celula de 50px para agrupar deteccoes do mesmo seat
    pos_votes: dict[tuple, Counter] = {}

    t = search_start
    while t <= search_end:
        fi = int(t * native_fps)
        crop = frames_cache.get(fi)
        if crop is not None:
            h = crop.shape[0]
            # Exclui board (centro) e seat do hero (base) — deixa apenas seats dos opponents
            board_excl = (int(h * 0.38), int(h * 0.75))
            for pair_info in find_showdown_cards(crop, board_excl=board_excl):
                lc   = _fmt_card(pair_info["left"])
                rc   = _fmt_card(pair_info["right"])
                pair = lc + rc
                cell = (pair_info["x"] // 50, pair_info["y"] // 50)
                if cell not in pos_votes:
                    pos_votes[cell] = Counter()
                pos_votes[cell][pair] += 1
        t += 1.0

    # Para cada posicao, escolhe o par com mais votos
    result: dict[str, list[str]] = {}
    for i, (cell, votes) in enumerate(
        sorted(pos_votes.items(), key=lambda kv: kv[0])  # ordena espacialmente
    ):
        if not votes:
            continue
        best_pair, n_votes = votes.most_common(1)[0]
        if n_votes >= 1 and len(best_pair) == 4:
            result[f"player_{i + 1}"] = [best_pair[:2], best_pair[2:]]

    return result


def _fmt_card(card: str) -> str:
    """'ah' -> 'Ah', '9c' -> '9c'"""
    return card[0].upper() + card[1]


def _extract_winner(
    tid: int,
    seg: list[TableEvent],
    all_evs: list[TableEvent],
    frames_cache: dict[int, np.ndarray],
    native_fps: float,
) -> Optional[str]:
    """
    Detecta o vencedor buscando o badge 'WINNER'.
    Usa votação por maioria entre todos os frames da janela pos-mao
    para robustez a falsos positivos do OCR.
    """
    seg_end_ts = max(e.timestamp for e in seg)

    future_new_hands = sorted(
        [e for e in all_evs if e.table_idx == tid
         and e.event_type == "new_hand" and e.timestamp > seg_end_ts],
        key=lambda e: e.timestamp,
    )
    next_new_hand_ts = (
        future_new_hands[0].timestamp if future_new_hands else seg_end_ts + 20.0
    )

    search_start = max(0.0, seg_end_ts - 3.0)
    search_end = min(next_new_hand_ts + 3.0, seg_end_ts + 30.0)

    votes: Counter = Counter()
    frames_checked = 0

    t = search_start
    while t <= search_end:
        fi = int(t * native_fps)
        crop = frames_cache.get(fi)
        if crop is not None:
            winner = ocr_winner(crop)
            if winner:
                votes[winner] += 1
            frames_checked += 1
        t += 1.0

    if not votes:
        return None

    # Retorna vencedor com maioria de votos; rejeita se só 1 frame votou
    # e há múltiplos frames disponíveis (evita falsos positivos)
    best, count = votes.most_common(1)[0]
    if frames_checked >= 3 and count == 1:
        # Candidato único em apenas 1 frame — muito fraco, descarta
        return None
    return best

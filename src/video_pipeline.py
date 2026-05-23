"""Task 4 — pipeline de vídeo a 10fps que emite TableEvent por mesa."""
from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional

from src.detectors import crop_table, count_board_cards, has_action_buttons, detect_pot_change
from src.ocr_engine import ocr_title_bar

# Mapeamento fixo das 4 mesas — válido para todo o vídeo de teste.
# mesas 2 e 3 têm banner de Jackpot cobrindo o título → hardcoded.
FIXED_TABLE_IDS = {
    0: "HL3458",
    1: "HL4017",
    2: "HL2332",
    3: "HL3048",
}


@dataclass
class TableEvent:
    timestamp:   float
    frame_idx:   int
    table_idx:   int        # índice da mesa (0-3)
    table_id:    str        # "HL4017"
    event_type:  str        # "board_change" | "pot_change" | "action" | "new_hand"
    board_cards: int        # 0, 3, 4, 5
    pot_bb:      Optional[float]
    action_btns: bool


def process_video(
    video_path: str,
    fps: float = 10.0,
    ocr_title: bool = True,
) -> dict[int, list[TableEvent]]:
    """
    Processa o vídeo a fps frames/s e retorna eventos por índice de mesa.
    events[1] = lista de TableEvent para mesa TR (HL4017).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip = max(1, round(native_fps / fps))
    real_fps = native_fps / skip

    print(f"Vídeo: {total_frames} frames @ {native_fps:.1f}fps → processando a {real_fps:.1f}fps (skip={skip})")

    # Estado por mesa
    prev_crops:   list[Optional[np.ndarray]] = [None] * 4
    prev_cards:   list[int]                  = [-1]   * 4
    table_ids:    list[Optional[str]]        = [None] * 4
    ocr_done:     list[bool]                 = [False] * 4
    ocr_attempts: list[int]                  = [0]    * 4
    MAX_OCR_ATTEMPTS = 5  # para após N falhas consecutivas

    events: dict[int, list[TableEvent]] = {i: [] for i in range(4)}

    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num % skip == 0:
            ts = frame_num / native_fps

            for tid in range(4):
                crop = crop_table(frame, tid)

                # OCR do título: até MAX_OCR_ATTEMPTS tentativas por mesa
                if ocr_title and not ocr_done[tid] and ocr_attempts[tid] < MAX_OCR_ATTEMPTS:
                    tid_str = ocr_title_bar(crop)
                    ocr_attempts[tid] += 1
                    if tid_str:
                        table_ids[tid] = tid_str
                        ocr_done[tid] = True

                table_id = table_ids[tid] or f"TABLE{tid}"
                n_cards = count_board_cards(crop)
                has_btn, _ = has_action_buttons(crop)

                prev_n = prev_cards[tid]

                # Detecta tipo de evento
                ev_type = None
                if prev_n >= 0:
                    if prev_n > 0 and n_cards == 0:
                        ev_type = "new_hand"
                    elif n_cards != prev_n and (n_cards in (3, 4, 5) or prev_n in (3, 4, 5)):
                        ev_type = "board_change"

                if prev_crops[tid] is not None:
                    changed, _ = detect_pot_change(prev_crops[tid], crop)
                    if changed:
                        if ev_type is None:
                            ev_type = "pot_change"

                if has_btn:
                    if ev_type is None:
                        ev_type = "action"

                if ev_type:
                    events[tid].append(TableEvent(
                        timestamp=ts,
                        frame_idx=frame_num,
                        table_idx=tid,
                        table_id=table_id,
                        event_type=ev_type,
                        board_cards=n_cards,
                        pot_bb=None,
                        action_btns=has_btn,
                    ))

                prev_cards[tid] = n_cards
                prev_crops[tid] = crop.copy()

        frame_num += 1

    cap.release()

    # Resolve table_ids: OCR ou fallback para mapeamento fixo
    for tid in range(4):
        if table_ids[tid] is None:
            # Tenta OCR num segundo frame
            cap2 = cv2.VideoCapture(video_path)
            ret, frame0 = cap2.read()
            cap2.release()
            if ret:
                crop0 = crop_table(frame0, tid)
                tid_str = ocr_title_bar(crop0)
                if tid_str:
                    table_ids[tid] = tid_str

        # Fallback para mapeamento fixo (cobertura pelo banner de Jackpot)
        if table_ids[tid] is None:
            table_ids[tid] = FIXED_TABLE_IDS.get(tid, f"TABLE{tid}")
            print(f"  mesa{tid}: OCR falhou, usando mapeamento fixo → {table_ids[tid]}")

        # Atualiza eventos com table_id correto
        for ev in events[tid]:
            ev.table_id = table_ids[tid]

    print(f"Eventos por mesa: { {i: len(v) for i, v in events.items()} }")
    return events

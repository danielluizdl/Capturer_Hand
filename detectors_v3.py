"""
Detectores v3 — calibrados com dados reais do video.
Usa delta entre frames para evitar falsos positivos do logo NEXA POKER.
"""
import cv2
import numpy as np
import time

VIDEO = r'c:\Users\danie\Documents\Projetos\Capturer_Hand\video_cortado_1min.mp4'
TABLE_REGIONS = [(0,0,960,540),(960,0,1920,540),(0,540,960,1080),(960,540,1920,1080)]

# Posicoes dos 5 slots de carta (frac de 960x540)
# Medidas empiricamente: cartas ocupam x=0.30-0.62 do crop de mesa
SLOT_X = [0.333, 0.393, 0.453, 0.513, 0.580]
CARD_Y1_F = 0.34
CARD_Y2_F = 0.56
SLOT_HW_F = 0.020

# Thresholds por slot calibrados com os dados medidos:
# slot 1: vazio=6-13, carta=27-37  → threshold 20
# slot 2: vazio=7-18, carta=47-51  → threshold 25
# slot 3: vazio=17-28, carta=56    → threshold 33
# slot 4: vazio=19-44, carta=44-47 → threshold 36  (logo interfere aqui)
# slot 5: vazio=12-17, carta=38-41 → threshold 25
SLOT_THRESHOLDS = [20, 25, 33, 36, 25]


def slot_stds(crop: np.ndarray) -> list[float]:
    """Retorna std de intensidade para cada um dos 5 slots de carta."""
    h, w = crop.shape[:2]
    y1 = int(h * CARD_Y1_F)
    y2 = int(h * CARD_Y2_F)
    hw = int(w * SLOT_HW_F)
    result = []
    for xf in SLOT_X:
        cx  = int(w * xf)
        sx1 = max(0, cx - hw)
        sx2 = min(w, cx + hw)
        slot = crop[y1:y2, sx1:sx2]
        result.append(float(cv2.cvtColor(slot, cv2.COLOR_BGR2GRAY).std()))
    return result


def count_board_cards(crop: np.ndarray) -> int:
    """Conta cartas do board usando threshold por slot calibrado."""
    stds = slot_stds(crop)
    return sum(1 for std, thr in zip(stds, SLOT_THRESHOLDS) if std > thr)


def has_action_buttons(crop: np.ndarray) -> tuple[bool, float]:
    """
    Detecta botoes Desistir/Pagar/Aumentar.
    Mede pixels brilhantes (texto branco dos botoes) na barra inferior.
    Calibrado: sem botoes ~0.0001, com botoes ~0.009+
    """
    h, w = crop.shape[:2]
    bar  = crop[int(h * 0.88):, :]
    gray = cv2.cvtColor(bar, cv2.COLOR_BGR2GRAY)
    score = float(np.sum(gray > 180) / gray.size)
    return score > 0.008, score


def detect_pot_change(prev: np.ndarray, curr: np.ndarray) -> tuple[bool, float]:
    """Detecta mudanca no texto 'Pote Total : X BB'."""
    h, w = curr.shape[:2]
    y1, y2 = int(h * 0.28), int(h * 0.36)
    x1, x2 = int(w * 0.28), int(w * 0.72)
    diff = float(np.mean(np.abs(
        curr[y1:y2, x1:x2].astype(float) - prev[y1:y2, x1:x2].astype(float)
    )))
    return diff > 4.0, diff


# -----------------------------------------------------------------------
# Benchmark completo + validacao de eventos
# -----------------------------------------------------------------------
if __name__ == "__main__":

    cap_info = cv2.VideoCapture(VIDEO)
    native_fps = cap_info.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / native_fps
    cap_info.release()

    print(f"Video: {total_frames} frames, {duration:.1f}s @ {native_fps}fps\n")

    # ---- benchmark de performance ----
    print(f"{'FPS':>5} | {'Frames':>7} | {'Tempo(s)':>9} | {'ms/frame':>9} | "
          f"{'Botoes':>7} | {'MudPot':>7} | {'Streets':>8}")
    print("-" * 65)

    for target_fps in [1, 2, 3, 6]:
        skip = max(1, round(native_fps / target_fps))
        real_fps = native_fps / skip

        cap = cv2.VideoCapture(VIDEO)
        frame_num = 0
        processed = 0
        t_start = time.perf_counter()

        events_btn   = 0
        events_pot   = 0
        street_changes = 0
        prev_crops   = [None] * 4
        prev_cards   = [-1]   * 4

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_num % skip == 0:
                for tid, (x1, y1, x2, y2) in enumerate(TABLE_REGIONS):
                    crop = frame[y1:y2, x1:x2]

                    n_cards = count_board_cards(crop)
                    has_btn, _ = has_action_buttons(crop)

                    if has_btn:
                        events_btn += 1
                    if prev_crops[tid] is not None:
                        changed, _ = detect_pot_change(prev_crops[tid], crop)
                        if changed:
                            events_pot += 1
                    # Transicao de street = numero de cartas mudou
                    if prev_cards[tid] >= 0 and n_cards != prev_cards[tid]:
                        street_changes += 1
                    prev_cards[tid] = n_cards
                    prev_crops[tid] = crop.copy()
                processed += 1
            frame_num += 1

        elapsed = time.perf_counter() - t_start
        cap.release()

        ms = elapsed / processed * 1000 if processed else 0
        print(f"{real_fps:5.1f} | {processed:7d} | {elapsed:9.2f} | {ms:9.1f} | "
              f"{events_btn:7d} | {events_pot:7d} | {street_changes:8d}")

    print()
    print("Botoes  = frames c/ botoes de acao visiveis (nosso turno em alguma mesa)")
    print("MudPot  = mudancas na area do texto do pot (somado 4 mesas)")
    print("Streets = transicoes detectadas (N cartas -> M cartas no board)")

    # ---- linha do tempo detalhada a 2fps para validacao ----
    print("\n=== LINHA DO TEMPO (2fps, todas as mesas) ===")
    print(f"{'t':>5} | {'TL':>6} | {'TR':>6} | {'BL':>6} | {'BR':>6} | {'Btn':>4}")
    print("-" * 40)

    skip2 = max(1, round(native_fps / 2.0))
    cap = cv2.VideoCapture(VIDEO)
    frame_num = 0
    prev_state = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num % skip2 == 0:
            ts = frame_num / native_fps
            row = []
            btn_any = False
            for tid, (x1, y1, x2, y2) in enumerate(TABLE_REGIONS):
                crop = frame[y1:y2, x1:x2]
                n = count_board_cards(crop)
                h, _ = has_action_buttons(crop)
                row.append(n)
                btn_any = btn_any or h

            state = tuple(row) + (btn_any,)
            if state != prev_state:
                street_map = {0: "pre", 3: "flp", 4: "trn", 5: "rvr"}
                streets = [street_map.get(n, f"?{n}") for n in row]
                btn_flag = "BTN" if btn_any else "   "
                print(f"{ts:5.1f} | {streets[0]:>6} | {streets[1]:>6} | "
                      f"{streets[2]:>6} | {streets[3]:>6} | {btn_flag}")
                prev_state = state
        frame_num += 1
    cap.release()

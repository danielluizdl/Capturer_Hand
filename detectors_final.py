"""
Detectores finais calibrados — WPT Global 4 mesas 1920x1080.

Correcoes aplicadas:
  1. Threshold slot 4 aumentado para 38 (logo NEXA interferia)
  2. Barra de botoes exclui taskbar do Windows (mesas BL/BR)
  3. Botao detecta por cor (verde/cinza escuro) nao so brilho
"""
import cv2
import numpy as np
import time

VIDEO = r'c:\Users\danie\Documents\Projetos\Capturer_Hand\video_cortado_1min.mp4'

# Quadrantes 960x540 dentro do frame 1920x1080
TABLE_REGIONS = [(0,0,960,540),(960,0,1920,540),(0,540,960,1080),(960,540,1920,1080)]
# Altura util de cada mesa (exclui taskbar Windows nas mesas de baixo)
# Mesas de cima: altura total 540. Mesas de baixo: so ate y_max-38 (taskbar ~38px)
TABLE_USEFUL_H = [540, 540, 502, 502]   # BL/BR: exclui taskbar

SLOT_X           = [0.333, 0.393, 0.453, 0.513, 0.580]
CARD_Y1_F        = 0.34
CARD_Y2_F        = 0.56
SLOT_HW_F        = 0.020
# Thresholds por slot — calibrados empiricamente no video real
# (vazio: 6-13, 7-18, 17-28, 19-32, 12-17 | carta: 27-37, 47-51, 56, 44-47, 38-41)
SLOT_THRESHOLDS  = [20, 25, 33, 38, 25]

# Altura da barra de acao (proporcao da altura UTIL)
ACTION_BAR_TOP_F = 0.900


def crop_table(frame: np.ndarray, tid: int) -> np.ndarray:
    """Retorna crop da mesa, respeitando altura util (sem taskbar)."""
    x1, y1, x2, y2 = TABLE_REGIONS[tid]
    uh = TABLE_USEFUL_H[tid]
    return frame[y1: y1 + uh, x1:x2]


def count_board_cards(crop: np.ndarray) -> int:
    """
    Conta cartas visiveis no board usando variancia por slot.
    Retorna 0, 3, 4 ou 5 (valores validos em poker).
    Valores 1 e 2 sao descartados como ruido do logo.
    """
    h, w = crop.shape[:2]
    y1 = int(h * CARD_Y1_F)
    y2 = int(h * CARD_Y2_F)
    hw = int(w * SLOT_HW_F)
    total = 0
    for xf, thr in zip(SLOT_X, SLOT_THRESHOLDS):
        cx  = int(w * xf)
        sx1 = max(0, cx - hw)
        sx2 = min(w, cx + hw)
        std = float(cv2.cvtColor(crop[y1:y2, sx1:sx2], cv2.COLOR_BGR2GRAY).std())
        if std > thr:
            total += 1
    # Filtra valores impossivel em poker (so 0, 3, 4, 5 sao validos)
    return total if total in (0, 3, 4, 5) else 0


def has_action_buttons(crop: np.ndarray) -> tuple[bool, float]:
    """
    Detecta botoes Desistir/Pagar/Aumentar.
    Busca por regioes VERDES vivas (botao Aumentar) ou cinza-azulado (botao Pagar)
    na barra de acao — cor especifica do WPT Global, diferente da interface normal.
    """
    h, w = crop.shape[:2]
    bar  = crop[int(h * ACTION_BAR_TOP_F):, :]
    hsv  = cv2.cvtColor(bar, cv2.COLOR_BGR2HSV)

    # Verde escuro dos botoes Aumentar/Passar (#2a7a2a aprox)
    green  = cv2.inRange(hsv, np.array([55, 60, 50]),  np.array([90, 255, 200]))
    # Cinza-azul dos botoes Pagar (#3a5a7a aprox) — hue 100-130
    blue   = cv2.inRange(hsv, np.array([95, 40, 60]),  np.array([130, 160, 180]))
    # Vermelho vivo do botao Desistir
    red1   = cv2.inRange(hsv, np.array([0,  100, 80]), np.array([8,  255, 200]))
    red2   = cv2.inRange(hsv, np.array([172,100, 80]), np.array([180,255, 200]))

    buttons_mask = cv2.bitwise_or(green, cv2.bitwise_or(blue, cv2.bitwise_or(red1, red2)))
    score = float(np.sum(buttons_mask > 0) / buttons_mask.size)
    return score > 0.015, score


def detect_pot_change(prev: np.ndarray, curr: np.ndarray) -> tuple[bool, float]:
    """Detecta mudanca na area de texto do pot."""
    h, w = curr.shape[:2]
    y1, y2 = int(h * 0.28), int(h * 0.36)
    x1, x2 = int(w * 0.28), int(w * 0.72)
    diff = float(np.mean(np.abs(
        curr[y1:y2, x1:x2].astype(float) - prev[y1:y2, x1:x2].astype(float)
    )))
    return diff > 4.0, diff


# -----------------------------------------------------------------------
# Benchmark + linha do tempo
# -----------------------------------------------------------------------
if __name__ == "__main__":
    cap_info = cv2.VideoCapture(VIDEO)
    native_fps = cap_info.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / native_fps
    cap_info.release()

    print(f"Video: {total_frames} frames, {duration:.1f}s @ {native_fps}fps\n")

    print(f"{'FPS':>5} | {'Frames':>7} | {'Tempo(s)':>9} | {'ms/frame':>9} | "
          f"{'Botoes':>7} | {'MudPot':>7} | {'Streets':>8}")
    print("-" * 65)

    for target_fps in [1, 2, 3, 6]:
        skip = max(1, round(native_fps / target_fps))
        real_fps = native_fps / skip
        cap = cv2.VideoCapture(VIDEO)
        frame_num = processed = events_btn = events_pot = street_changes = 0
        prev_crops = [None]*4
        prev_cards = [-1]*4
        t_start = time.perf_counter()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_num % skip == 0:
                for tid in range(4):
                    crop = crop_table(frame, tid)
                    n    = count_board_cards(crop)
                    h_b, _ = has_action_buttons(crop)
                    if h_b:
                        events_btn += 1
                    if prev_crops[tid] is not None:
                        ch, _ = detect_pot_change(prev_crops[tid], crop)
                        if ch:
                            events_pot += 1
                    if prev_cards[tid] >= 0 and n != prev_cards[tid]:
                        street_changes += 1
                    prev_cards[tid] = n
                    prev_crops[tid] = crop.copy()
                processed += 1
            frame_num += 1

        elapsed = time.perf_counter() - t_start
        cap.release()
        ms = elapsed / processed * 1000 if processed else 0
        print(f"{real_fps:5.1f} | {processed:7d} | {elapsed:9.2f} | {ms:9.1f} | "
              f"{events_btn:7d} | {events_pot:7d} | {street_changes:8d}")

    # Linha do tempo detalhada a 2fps
    print("\n=== LINHA DO TEMPO 2fps (4 mesas) ===")
    print(f"{'t':>5} | {'TL':>6} | {'TR':>6} | {'BL':>6} | {'BR':>6} | Btn")
    print("-" * 44)
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
            cards = []
            btn_any = False
            for tid in range(4):
                crop = crop_table(frame, tid)
                n    = count_board_cards(crop)
                h_b, _ = has_action_buttons(crop)
                cards.append(n)
                btn_any = btn_any or h_b
            state = (tuple(cards), btn_any)
            if state != prev_state:
                sm = {0:"pre", 3:"flp", 4:"trn", 5:"rvr"}
                s = [sm.get(n, f"??") for n in cards]
                b = "<<BTN" if btn_any else ""
                print(f"{ts:5.1f} | {s[0]:>6} | {s[1]:>6} | {s[2]:>6} | {s[3]:>6} | {b}")
                prev_state = state
        frame_num += 1
    cap.release()

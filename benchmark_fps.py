import cv2
import numpy as np
import time

VIDEO = r'c:\Users\danie\Documents\Projetos\Capturer_Hand\video_cortado_1min.mp4'
TABLE_REGIONS = [(0,0,960,540),(960,0,1920,540),(0,540,960,1080),(960,540,1920,1080)]


def count_board_cards(crop):
    h, w = crop.shape[:2]
    board = crop[int(h*0.30): int(h*0.58), int(w*0.28): int(w*0.72)]
    gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cards = 0
    for c in cnts:
        x, y, cw, ch = cv2.boundingRect(c)
        asp = cw / ch if ch > 0 else 0
        if 0.45 < asp < 0.95 and 800 < cw * ch < 12000:
            cards += 1
    return min(cards, 5)


def has_action_buttons(crop):
    h, w = crop.shape[:2]
    bar = crop[int(h*0.88):, :]
    hsv = cv2.cvtColor(bar, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, np.array([60, 80, 60]),  np.array([100, 255, 220]))
    red1  = cv2.inRange(hsv, np.array([0,  100, 80]), np.array([10,  255, 220]))
    red2  = cv2.inRange(hsv, np.array([170,100, 80]), np.array([180, 255, 220]))
    colored = cv2.bitwise_or(green, cv2.bitwise_or(red1, red2))
    ratio = np.sum(colored > 0) / colored.size
    return ratio > 0.02, ratio


def detect_pot_change(prev, curr):
    h, w = curr.shape[:2]
    r_prev = prev[int(h*0.28):int(h*0.38), int(w*0.25):int(w*0.72)]
    r_curr = curr[int(h*0.28):int(h*0.38), int(w*0.25):int(w*0.72)]
    diff = np.mean(np.abs(r_curr.astype(float) - r_prev.astype(float)))
    return diff > 3.0, diff


cap_info = cv2.VideoCapture(VIDEO)
native_fps = cap_info.get(cv2.CAP_PROP_FPS)
total_frames = int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT))
duration_s = total_frames / native_fps
cap_info.release()

print(f'Video: {total_frames} frames, {duration_s:.1f}s @ {native_fps}fps')
print()
header = f"{'FPS':>5} | {'Frames':>7} | {'Tempo(s)':>9} | {'ms/frame':>9} | {'Botoes':>7} | {'MudPot':>7} | {'CartasTL':>9} | {'CartasTR':>9}"
print(header)
print('-' * len(header))

for target_fps in [1, 2, 3, 6, 10]:
    skip = max(1, round(native_fps / target_fps))
    real_fps = native_fps / skip

    cap = cv2.VideoCapture(VIDEO)
    frame_num = 0
    processed = 0
    t_start = time.perf_counter()

    events_buttons = 0
    events_pot = 0
    board_tl = []
    board_tr = []
    prev_crops = [None] * 4

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num % skip == 0:
            for tid, (x1, y1, x2, y2) in enumerate(TABLE_REGIONS):
                crop = frame[y1:y2, x1:x2]
                n = count_board_cards(crop)
                has_btn, _ = has_action_buttons(crop)

                if tid == 0:
                    board_tl.append(n)
                if tid == 1:
                    board_tr.append(n)
                if has_btn:
                    events_buttons += 1
                if prev_crops[tid] is not None:
                    changed, _ = detect_pot_change(prev_crops[tid], crop)
                    if changed:
                        events_pot += 1
                prev_crops[tid] = crop.copy()
            processed += 1
        frame_num += 1

    elapsed = time.perf_counter() - t_start
    cap.release()

    ms_per = elapsed / processed * 1000 if processed else 0
    tr_tl = sum(1 for i in range(1, len(board_tl)) if board_tl[i] != board_tl[i-1])
    tr_tr = sum(1 for i in range(1, len(board_tr)) if board_tr[i] != board_tr[i-1])

    print(f'{real_fps:5.1f} | {processed:7d} | {elapsed:9.2f} | {ms_per:9.1f} | {events_buttons:7d} | {events_pot:7d} | {tr_tl:9d} | {tr_tr:9d}')

print()
print('Botoes  = frames onde botoes de acao estao visiveis (nosso turno em alguma mesa)')
print('MudPot  = mudancas na area do pot (somadas nas 4 mesas)')
print('CartasTL/TR = transicoes de street detectadas (board mudou de N para M cartas)')
print()
print('Obs: sem OCR neste benchmark — so visao computacional pura')

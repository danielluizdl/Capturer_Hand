import cv2
import numpy as np
import os

VIDEO  = r'c:\Users\danie\Documents\Projetos\Capturer_Hand\video_cortado_1min.mp4'
FRAMES = r'c:\Users\danie\Documents\Projetos\Capturer_Hand\frames_preview'
TABLE_REGIONS = [(0,0,960,540),(960,0,1920,540),(0,540,960,1080),(960,540,1920,1080)]

# -----------------------------------------------------------
# Extrai 1 frame por segundo e salva diagnosticos visuais
# -----------------------------------------------------------

cap = cv2.VideoCapture(VIDEO)
native_fps = cap.get(cv2.CAP_PROP_FPS)

# Pega o frame com cartas visiveis: ~20% do video (t=15s)
target_times = [4.0, 10.0, 20.0, 33.0]

for t in target_times:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * native_fps))
    ret, frame = cap.read()
    if not ret:
        continue

    # Analisa mesa TL (mais cartas visiveis nos frames que vimos)
    crop = frame[0:540, 0:960]
    h, w = crop.shape[:2]

    # ROI do board que estamos usando
    board_roi = crop[int(h*0.30): int(h*0.58), int(w*0.28): int(w*0.72)]
    gray = cv2.cvtColor(board_roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Contornos encontrados
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(f'\n=== t={t}s Mesa TL ===')
    print(f'  board_roi: {board_roi.shape[1]}x{board_roi.shape[0]}px')
    print(f'  Contornos totais: {len(cnts)}')
    card_count = 0
    for c in cnts:
        x, y, cw, ch = cv2.boundingRect(c)
        asp = cw / ch if ch > 0 else 0
        area = cw * ch
        qualifica = 0.45 < asp < 0.95 and 800 < area < 12000
        if qualifica:
            card_count += 1
        if area > 400:  # mostra contornos relevantes
            print(f'    contorno: {cw}x{ch}px, asp={asp:.2f}, area={area}, CARTA={qualifica}')
    print(f'  => Cartas detectadas: {card_count}')

    # Salva imagens de diagnostico
    diag = np.hstack([
        cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    ])
    # Desenha contornos das cartas sobre a ROI original
    roi_debug = board_roi.copy()
    for c in cnts:
        x, y, cw, ch = cv2.boundingRect(c)
        asp = cw / ch if ch > 0 else 0
        area = cw * ch
        color = (0,255,0) if (0.45 < asp < 0.95 and 800 < area < 12000) else (0,0,255)
        cv2.rectangle(roi_debug, (x,y), (x+cw,y+ch), color, 2)

    out = np.hstack([board_roi, roi_debug, cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)])
    cv2.imwrite(os.path.join(FRAMES, f'diag_board_t{int(t)}s.jpg'), out)
    print(f'  Salvo: diag_board_t{int(t)}s.jpg')

    # Tambem testa detector de botoes
    bar = crop[int(h*0.88):, :]
    hsv = cv2.cvtColor(bar, cv2.COLOR_BGR2HSV)
    s_mean = hsv[:,:,1].mean()
    v_mean = hsv[:,:,2].mean()
    green = cv2.inRange(hsv, np.array([60,80,60]),  np.array([100,255,220]))
    red1  = cv2.inRange(hsv, np.array([0,100,80]),  np.array([10,255,220]))
    red2  = cv2.inRange(hsv, np.array([170,100,80]),np.array([180,255,220]))
    colored = cv2.bitwise_or(green, cv2.bitwise_or(red1, red2))
    ratio = np.sum(colored>0) / colored.size
    print(f'  Botoes barra inferior: ratio={ratio:.4f}, S={s_mean:.1f}, V={v_mean:.1f} => {"SIM" if ratio>0.02 else "nao"}')
    cv2.imwrite(os.path.join(FRAMES, f'diag_bar_t{int(t)}s.jpg'), bar)

cap.release()
print('\nDone.')

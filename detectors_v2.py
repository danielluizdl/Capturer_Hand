"""
Detectores calibrados para WPT Global (960x540 por mesa).
Testado no video video_cortado_1min.mp4.
"""
import cv2
import numpy as np

TABLE_REGIONS = [(0,0,960,540),(960,0,1920,540),(0,540,960,1080),(960,540,1920,1080)]


def count_board_cards(crop: np.ndarray) -> int:
    """
    Conta cartas do board por deteccao de regioes coloridas no centro da mesa.

    As cartas WPT Global tem fundos COLORIDOS (verde/azul/preto), nao branco.
    Estrategia: em cada um dos 5 slots fixos, checa se o centro tem cor
    significativamente diferente do feltro verde escuro da mesa.
    """
    h, w = crop.shape[:2]

    # ROI do board (calibrada visualmente nos frames reais)
    bx1, by1 = int(w * 0.30), int(h * 0.32)
    bx2, by2 = int(w * 0.70), int(h * 0.57)
    board = crop[by1:by2, bx1:bx2]
    bh, bw = board.shape[:2]

    # Cor do feltro (verde escuro): amostrada fora das cartas
    # HSV aproximado: H=70-90, S=50-120, V=30-70
    hsv = cv2.cvtColor(board, cv2.COLOR_BGR2HSV)

    # Mascara do feltro (o que NAO e carta)
    felt_mask = cv2.inRange(hsv,
                            np.array([60, 30, 20]),
                            np.array([95, 150, 90]))
    # Mascara de "nao feltro" = possivelmente carta ou UI
    not_felt = cv2.bitwise_not(felt_mask)

    # Morfologia: fecha buracos e remove ruido pequeno
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(not_felt, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed,  cv2.MORPH_OPEN,  kernel)

    # Detecta contornos de regioes "nao feltro" grandes o suficiente para ser cartas
    cnts, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cards = 0
    for c in cnts:
        x, y, cw, ch = cv2.boundingRect(c)
        asp = cw / ch if ch > 0 else 0
        area = cw * ch
        # Carta: aspecto retangular vertical (0.45-0.90), area razoavel
        # Em 960x540 uma carta tem ~55x75px = area ~4100
        min_area = int(bh * bw * 0.018)  # ~1.8% da area do board = ~1 carta
        max_area = int(bh * bw * 0.25)   # max 1 carta gigante
        if 0.40 < asp < 1.0 and min_area < area < max_area:
            cards += 1

    return min(cards, 5)


def count_board_cards_slot(crop: np.ndarray) -> int:
    """
    Abordagem alternativa mais robusta: verifica 5 slots fixos.
    Para cada slot, mede a variancia de cor — carta = alta variancia.
    """
    h, w = crop.shape[:2]

    # Posicoes dos 5 slots de carta (relativas ao crop 960x540)
    # Calibradas observando os frames reais
    card_y1, card_y2 = int(h * 0.34), int(h * 0.56)
    card_x_centers = [
        int(w * 0.345),  # slot 1
        int(w * 0.405),  # slot 2
        int(w * 0.465),  # slot 3
        int(w * 0.525),  # slot 4
        int(w * 0.585),  # slot 5
    ]
    slot_hw = int(w * 0.022)  # meia-largura do slot de amostragem

    # Cor de referencia do feltro (amostrada do centro quando board vazio)
    # BGR aproximado: (35, 65, 35) — verde escuro
    felt_bgr = np.array([35.0, 65.0, 35.0])

    count = 0
    for cx in card_x_centers:
        x1, x2 = max(0, cx - slot_hw), min(w, cx + slot_hw)
        slot = crop[card_y1:card_y2, x1:x2]
        if slot.size == 0:
            continue
        mean_bgr = slot.reshape(-1, 3).mean(axis=0).astype(float)
        dist = float(np.linalg.norm(mean_bgr - felt_bgr))
        # Se a cor media do slot e muito diferente do feltro -> carta presente
        if dist > 25:
            count += 1

    return count


def has_action_buttons(crop: np.ndarray) -> tuple[bool, float]:
    """
    Detecta botoes Desistir/Pagar/Aumentar na barra inferior do WPT Global.
    Os botoes tem fundo escuro com texto branco. A barra SEM botoes e quase preta.

    Retorna (tem_botoes, score).
    """
    h, w = crop.shape[:2]

    # Barra de acao: ocupa os ultimos ~12% da altura
    bar = crop[int(h * 0.88):, :]

    # Botoes tem texto branco brilhante (V > 180) em fundo escuro
    gray = cv2.cvtColor(bar, cv2.COLOR_BGR2GRAY)
    bright_pixels = np.sum(gray > 180) / gray.size

    # A barra sem botoes tem poucos pixels brilhantes (so icones pequenos)
    # Com botoes: muito mais texto branco visivel
    threshold = 0.015
    return bright_pixels > threshold, bright_pixels


def detect_pot_text_change(prev_crop: np.ndarray, curr_crop: np.ndarray) -> tuple[bool, float]:
    """
    Detecta mudanca no texto do pot (area 'Pote Total: X BB').
    Compara a ROI especifica do texto do pot entre frames consecutivos.
    """
    h, w = curr_crop.shape[:2]
    # Texto 'Pote Total' fica acima das cartas
    pot_y1, pot_y2 = int(h * 0.28), int(h * 0.36)
    pot_x1, pot_x2 = int(w * 0.28), int(w * 0.72)

    roi_prev = prev_crop[pot_y1:pot_y2, pot_x1:pot_x2]
    roi_curr = curr_crop[pot_y1:pot_y2, pot_x1:pot_x2]

    diff = float(np.mean(np.abs(roi_curr.astype(float) - roi_prev.astype(float))))
    return diff > 4.0, diff


# -----------------------------------------------------------------------
# Teste visual: salva debug dos dois metodos num frame com cartas
# -----------------------------------------------------------------------
if __name__ == "__main__":
    import os
    VIDEO  = r'c:\Users\danie\Documents\Projetos\Capturer_Hand\video_cortado_1min.mp4'
    OUTDIR = r'c:\Users\danie\Documents\Projetos\Capturer_Hand\frames_preview'

    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)

    test_times = [4.0, 10.0, 20.0, 33.0]

    print(f"{'t':>5} | {'Mesa':>5} | {'v1':>4} | {'v2':>4} | {'Botoes':>8} | {'Score':>7}")
    print('-' * 48)

    for t in test_times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ret, frame = cap.read()
        if not ret:
            continue

        for tid, (x1, y1, x2, y2) in enumerate(TABLE_REGIONS):
            crop = frame[y1:y2, x1:x2]
            v1 = count_board_cards(crop)
            v2 = count_board_cards_slot(crop)
            has_btn, score = has_action_buttons(crop)
            print(f"{t:5.1f} | {tid:5d} | {v1:4d} | {v2:4d} | {str(has_btn):>8} | {score:.4f}")

            # Salva debug visual do metodo de slot apenas para mesa 0
            if tid == 0:
                h_c, w_c = crop.shape[:2]
                debug = crop.copy()
                # Desenha os 5 slots de amostragem
                card_y1_d = int(h_c * 0.34)
                card_y2_d = int(h_c * 0.56)
                x_centers = [int(w_c * p) for p in [0.345, 0.405, 0.465, 0.525, 0.585]]
                slot_hw   = int(w_c * 0.022)
                felt_bgr  = np.array([35.0, 65.0, 35.0])
                for i, cx in enumerate(x_centers):
                    sx1, sx2 = max(0, cx - slot_hw), min(w_c, cx + slot_hw)
                    slot = crop[card_y1_d:card_y2_d, sx1:sx2]
                    mean_bgr = slot.reshape(-1, 3).mean(axis=0).astype(float)
                    dist = float(np.linalg.norm(mean_bgr - felt_bgr))
                    color = (0, 255, 0) if dist > 25 else (0, 0, 255)
                    cv2.rectangle(debug, (sx1, card_y1_d), (sx2, card_y2_d), color, 2)
                    cv2.putText(debug, f"{dist:.0f}", (sx1, card_y1_d - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                path = os.path.join(OUTDIR, f'debug_slots_t{int(t)}s_mesa{tid}.jpg')
                cv2.imwrite(path, debug)

        print()

    cap.release()
    print("Arquivos de debug salvos em frames_preview/")

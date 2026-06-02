"""
Calibrador de seats com painel lateral e drag & drop.

Uso:
    python calibrate_seats.py video.mp4
    python calibrate_seats.py video.mp4 --frame 1200

Controles:
    Painel esquerdo  — clique para selecionar mesa ou item
    Frame direito    — arraste para marcar a regiao
    Z                — desfazer ultimo
    Q / fechar       — salvar e sair
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np

SEAT_REGIONS_PATH = Path(__file__).parent / "src" / "seat_regions.py"

# ---------------------------------------------------------------------------
# Paleta
# ---------------------------------------------------------------------------
C_BG        = (30,  30,  30)
C_PANEL     = (45,  45,  45)
C_HEADER    = (60,  60,  60)
C_SEL       = (50,  80, 130)      # item selecionado (azul)
C_DONE      = (30,  90,  50)      # item calibrado (verde escuro)
C_TEXT      = (220, 220, 220)
C_TEXT_DIM  = (130, 130, 130)
C_OK        = (80,  200, 100)
C_WHITE     = (255, 255, 255)
C_GREEN     = (0,   230,  80)
C_RED       = (60,   60, 200)
C_DIVIDER   = (70,   70,  70)

SEAT_COLORS = {
    "hero": (0,   230,  80),
    2:      (0,   210, 255),
    3:      (255, 130,   0),
    4:      (0,   255, 140),
    5:      (200,   0, 255),
    6:      (0,   110, 255),
    7:      (255,   0, 140),
    8:      (120, 255,   0),
}

# ---------------------------------------------------------------------------
# Layout do painel
# ---------------------------------------------------------------------------
PANEL_W   = 230
ITEM_H    = 32
HEADER_H  = 36
BTN_H     = 38
FONT      = cv2.FONT_HERSHEY_SIMPLEX
WIN       = "Calibracao de Seats"


# ---------------------------------------------------------------------------
# Estado global
# ---------------------------------------------------------------------------
class State:
    def __init__(self, n_tables: int):
        self.active_table = 0
        self.active_item  = "hero"          # "hero" | 2..8
        self.n_tables     = n_tables

        # {(table, item): (cx1,cy1,cx2,cy2)}
        self.calibrated: dict[tuple, tuple] = {}

        # Drag
        self.drag_start: tuple[int, int] | None = None
        self.dragging   = False
        self.drag_rect: tuple | None = None  # em coord de exibicao

    def items(self):
        return ["hero"] + list(range(2, 9))

    def item_label(self, item) -> str:
        return "Hero" if item == "hero" else f"Seat {item}"

    def item_color(self, item) -> tuple:
        return SEAT_COLORS.get(item, (180, 180, 180))

    def is_done(self, table, item) -> bool:
        r = self.calibrated.get((table, item))
        return r is not None and r[2] > r[0] and r[3] > r[1]

    def undo(self):
        if self.calibrated:
            key = list(self.calibrated)[-1]
            del self.calibrated[key]
            return key
        return None

    def n_done(self, table) -> int:
        return sum(1 for (t, _), r in self.calibrated.items()
                   if t == table and r[2] > r[0])

    def total_done(self) -> int:
        return sum(1 for r in self.calibrated.values() if r[2] > r[0])


# ---------------------------------------------------------------------------
# Desenho do painel
# ---------------------------------------------------------------------------
def draw_panel(canvas: np.ndarray, s: State, panel_h: int):
    # Fundo do painel
    canvas[:panel_h, :PANEL_W] = C_BG

    y = 0

    # --- Titulo ---
    cv2.rectangle(canvas, (0, y), (PANEL_W, y + HEADER_H), C_HEADER, -1)
    cv2.putText(canvas, "CALIBRAR SEATS", (10, y + 24),
                FONT, 0.52, C_WHITE, 1, cv2.LINE_AA)
    y += HEADER_H

    # Linha separadora
    cv2.line(canvas, (0, y), (PANEL_W, y), C_DIVIDER, 1)
    y += 6

    # --- Seletor de mesas ---
    cv2.putText(canvas, "MESA ATIVA:", (10, y + 14),
                FONT, 0.38, C_TEXT_DIM, 1, cv2.LINE_AA)
    y += 18

    btn_w = (PANEL_W - 12) // s.n_tables
    for tid in range(s.n_tables):
        bx = 6 + tid * btn_w
        by = y
        is_sel = (tid == s.active_table)
        n_done = s.n_done(tid)
        bg = C_SEL if is_sel else C_PANEL
        cv2.rectangle(canvas, (bx, by), (bx + btn_w - 3, by + BTN_H - 4), bg, -1)
        label = f"M{tid}"
        cv2.putText(canvas, label, (bx + 8, by + 16),
                    FONT, 0.48, C_WHITE, 1, cv2.LINE_AA)
        # mini badge com quantos done
        badge = f"{n_done}/8"
        cv2.putText(canvas, badge, (bx + 4, by + 30),
                    FONT, 0.32, C_OK if n_done > 0 else C_TEXT_DIM, 1, cv2.LINE_AA)
    y += BTN_H + 4

    cv2.line(canvas, (0, y), (PANEL_W, y), C_DIVIDER, 1)
    y += 6

    # --- Lista de itens ---
    cv2.putText(canvas, "ITEM A CALIBRAR:", (10, y + 14),
                FONT, 0.38, C_TEXT_DIM, 1, cv2.LINE_AA)
    y += 20

    for item in s.items():
        is_sel  = (item == s.active_item)
        is_done = s.is_done(s.active_table, item)
        color   = s.item_color(item)

        bg = C_SEL if is_sel else (C_DONE if is_done else C_BG)
        cv2.rectangle(canvas, (0, y), (PANEL_W, y + ITEM_H - 1), bg, -1)

        # Barra colorida lateral
        cv2.rectangle(canvas, (0, y + 2), (5, y + ITEM_H - 3), color, -1)

        # Status icon
        status_txt = "✓" if is_done else ("▶" if is_sel else "○")
        st_color   = C_OK if is_done else (C_WHITE if is_sel else C_TEXT_DIM)
        # OpenCV nao renderiza unicode — usar ASCII equivalente
        status_txt = "OK" if is_done else (">" if is_sel else "-")
        cv2.putText(canvas, status_txt, (10, y + 21),
                    FONT, 0.40, st_color, 1, cv2.LINE_AA)

        label = s.item_label(item)
        lc    = C_WHITE if is_sel else (C_OK if is_done else C_TEXT)
        cv2.putText(canvas, label, (32, y + 21),
                    FONT, 0.45, lc, 1, cv2.LINE_AA)

        # Coordenadas se calibrado
        if is_done:
            r = s.calibrated[(s.active_table, item)]
            coord_str = f"{r[0]},{r[1]}-{r[2]},{r[3]}"
            cv2.putText(canvas, coord_str, (32, y + ITEM_H - 6),
                        FONT, 0.28, C_TEXT_DIM, 1, cv2.LINE_AA)

        y += ITEM_H

    cv2.line(canvas, (0, y), (PANEL_W, y), C_DIVIDER, 1)
    y += 8

    # --- Instrucoes ---
    instructions = [
        "COMO USAR:",
        "1. Clique no item",
        "2. Arraste no frame",
        "3. Repita p/ todos",
        "",
        "Z = desfazer",
        "Q = salvar e sair",
    ]
    for line in instructions:
        is_tip = line.startswith("Z") or line.startswith("Q")
        color  = C_TEXT_DIM if not line.startswith("COMO") else C_WHITE
        cv2.putText(canvas, line, (10, y + 14),
                    FONT, 0.38, color, 1, cv2.LINE_AA)
        y += 18

    y += 4
    cv2.line(canvas, (0, y), (PANEL_W, y), C_DIVIDER, 1)
    y += 8

    # --- Progresso total ---
    total = s.total_done()
    max_total = s.n_tables * 8
    pct  = int(total / max_total * 100) if max_total else 0
    bar_w = PANEL_W - 20
    filled = int(bar_w * pct / 100)
    cv2.rectangle(canvas, (10, y), (10 + bar_w, y + 12), C_PANEL, -1)
    if filled > 0:
        cv2.rectangle(canvas, (10, y), (10 + filled, y + 12), C_OK, -1)
    cv2.putText(canvas, f"Total: {total}/{max_total} ({pct}%)", (10, y + 26),
                FONT, 0.38, C_TEXT, 1, cv2.LINE_AA)
    y += 34

    # --- Botao Salvar ---
    cv2.rectangle(canvas, (10, y), (PANEL_W - 10, y + BTN_H),
                  (40, 120, 40) if total > 0 else C_HEADER, -1)
    cv2.putText(canvas, "SALVAR E SAIR  (Q)", (18, y + 25),
                FONT, 0.42, C_WHITE, 1, cv2.LINE_AA)


def draw_frame_overlay(
    canvas: np.ndarray,
    s: State,
    frame_img: np.ndarray,
    frame_x: int,
    table_divs: tuple[int, int],   # (mid_x, mid_y) em frame coords
    scale: float,
):
    """Desenha o frame com regioes calibradas e drag ativo."""
    h, w = frame_img.shape[:2]
    canvas[0:h, frame_x:frame_x + w] = frame_img

    mid_x_d = int(table_divs[0] * scale)
    mid_y_d = int(table_divs[1] * scale)

    # Linhas de divisao das mesas
    cv2.line(canvas, (frame_x + mid_x_d, 0), (frame_x + mid_x_d, h), C_DIVIDER, 2)
    cv2.line(canvas, (frame_x, mid_y_d), (frame_x + w, mid_y_d), C_DIVIDER, 2)

    # Labels das mesas
    table_offsets_d = [
        (0, 0), (mid_x_d, 0), (0, mid_y_d), (mid_x_d, mid_y_d)
    ]
    for tid, (ox, oy) in enumerate(table_offsets_d):
        label = f"Mesa {tid}"
        is_act = (tid == s.active_table)
        col    = C_GREEN if is_act else C_TEXT_DIM
        cv2.putText(canvas, label, (frame_x + ox + 6, oy + 20),
                    FONT, 0.55, col, 1 if not is_act else 2, cv2.LINE_AA)

    # Regioes calibradas desta mesa
    offsets = table_offsets_d  # (dx, dy) em display para cada mesa
    for (tid, item), coords in s.calibrated.items():
        cx1, cy1, cx2, cy2 = coords
        ox, oy = offsets[tid]
        dx1 = frame_x + ox + int(cx1 * scale)
        dy1 = oy + int(cy1 * scale)
        dx2 = frame_x + ox + int(cx2 * scale)
        dy2 = oy + int(cy2 * scale)
        color = s.item_color(item)
        thick = 2 if tid == s.active_table else 1
        cv2.rectangle(canvas, (dx1, dy1), (dx2, dy2), color, thick)
        lbl = s.item_label(item)
        cv2.putText(canvas, lbl, (dx1 + 2, dy1 + 13),
                    FONT, 0.35, color, 1, cv2.LINE_AA)

    # Drag preview
    if s.drag_rect:
        x1, y1, x2, y2 = s.drag_rect
        color = s.item_color(s.active_item)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1)
        cv2.rectangle(canvas, (x1 + 1, y1 + 1), (x2 - 1, y2 - 1),
                      (*color[:2], min(color[2] + 80, 255)), 1)


# ---------------------------------------------------------------------------
# Hit-test do painel
# ---------------------------------------------------------------------------
def panel_hit(y_click: int, s: State) -> str | None:
    """
    Retorna acao baseada em onde o usuario clicou no painel.
    'table_N' | 'item_X' | 'save' | None
    """
    y = HEADER_H + 6 + 18  # pos inicial dos botoes de mesa
    btn_h = BTN_H

    # Botoes de mesa
    if y <= y_click < y + btn_h:
        btn_w = (PANEL_W - 12) // s.n_tables
        return None   # precisa do x tambem — tratado no caller

    y += btn_h + 10  # pos inicial da lista de itens
    for item in s.items():
        if y <= y_click < y + ITEM_H:
            return f"item_{item}"
        y += ITEM_H

    # Botao Salvar (na base do painel)
    # Nao tem posicao fixa — verificar de baixo
    return None


# ---------------------------------------------------------------------------
# Conversao de coordenadas
# ---------------------------------------------------------------------------
def disp_to_crop(
    dx: int, dy: int, frame_x: int, scale: float,
    mid_frame: tuple[int, int],
) -> tuple[int, int, int, int]:
    """
    Converte coordenada de exibicao (canvas) para (table_idx, crop_x, crop_y).
    mid_frame = (mid_x, mid_y) em pixels do frame original.
    """
    fx = int((dx - frame_x) / scale)
    fy = int(dy / scale)
    mid_x, mid_y = mid_frame
    table_idx = (1 if fx >= mid_x else 0) + (2 if fy >= mid_y else 0)
    ox = mid_x if fx >= mid_x else 0
    oy = mid_y if fy >= mid_y else 0
    return table_idx, fx - ox, fy - oy


# ---------------------------------------------------------------------------
# Persistencia
# ---------------------------------------------------------------------------
def save_to_file(s: State):
    content = SEAT_REGIONS_PATH.read_text(encoding="utf-8")
    lines   = content.splitlines(keepends=True)

    for (tid, item), coords in s.calibrated.items():
        x1, y1, x2, y2 = coords
        new_val = f"({x1}, {y1}, {x2}, {y2})"

        if item == "hero":
            # Substitui na secao HERO_REGIONS
            pat = re.compile(rf"(\s*{tid}\s*:\s*)\([^)]*\)")
            in_hero = False
            new_lines = []
            for line in lines:
                if "HERO_REGIONS" in line:
                    in_hero = True
                elif in_hero and "SEAT_REGIONS" in line:
                    in_hero = False
                if in_hero and pat.search(line):
                    line = pat.sub(rf"\g<1>{new_val}", line, count=1)
                    in_hero = False
                new_lines.append(line)
            lines = new_lines
        else:
            # Substitui na secao SEAT_REGIONS, bloco da mesa tid, seat item
            pat = re.compile(rf"(\s*{item}\s*:\s*)\([^)]*\)")
            in_seat = False
            in_table_block = False
            replaced = False
            new_lines = []
            for line in lines:
                if "SEAT_REGIONS" in line and "HERO" not in line:
                    in_seat = True
                if in_seat and not replaced:
                    if re.match(rf"^\s*{tid}\s*:", line):
                        in_table_block = True
                    elif in_table_block and re.match(r"^\s*\d+\s*:", line) and not re.match(rf"^\s*{item}\s*:", line):
                        # Inicio de outro bloco de mesa sem ter encontrado o seat
                        in_table_block = False
                    if in_table_block and pat.search(line):
                        line = pat.sub(rf"\g<1>{new_val}", line, count=1)
                        replaced = True
                new_lines.append(line)
            lines = new_lines

    SEAT_REGIONS_PATH.write_text("".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="Caminho do video MP4")
    ap.add_argument("--frame", type=int, default=500)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Erro: nao foi possivel abrir {args.video}")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"Erro: frame {args.frame} nao encontrado")
        sys.exit(1)

    fh, fw = frame.shape[:2]
    mid_frame = (fw // 2, fh // 2)
    n_tables  = 4

    # Escala para caber em tela (frame ocupa o lado direito)
    max_fw, max_fh = 1400, 900
    scale = min(max_fw / fw, max_fh / fh, 1.0)
    disp_fw = int(fw * scale)
    disp_fh = int(fh * scale)

    frame_scaled = cv2.resize(frame, (disp_fw, disp_fh), interpolation=cv2.INTER_AREA)

    canvas_w = PANEL_W + disp_fw
    canvas_h = max(disp_fh, 700)

    s = State(n_tables)

    # Canvas permanente
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    frame_x = PANEL_W   # offset do frame dentro do canvas

    # ---------------------------------------------------------------------------
    # Mouse callback
    # ---------------------------------------------------------------------------
    def on_mouse(event, cx, cy, flags, param):
        # Clique no painel esquerdo
        if cx < PANEL_W:
            if event == cv2.EVENT_LBUTTONDOWN:
                # Botoes de mesa
                y_btn_start = HEADER_H + 6 + 18
                if y_btn_start <= cy < y_btn_start + BTN_H:
                    btn_w = (PANEL_W - 12) // n_tables
                    tid = (cx - 6) // btn_w
                    if 0 <= tid < n_tables:
                        s.active_table = tid
                    return

                # Lista de itens
                y_item_start = y_btn_start + BTN_H + 10
                for item in s.items():
                    if y_item_start <= cy < y_item_start + ITEM_H:
                        s.active_item = item
                        return
                    y_item_start += ITEM_H

                # Botao Salvar (ultima area do painel)
                if cy > canvas_h - 60:
                    cv2.destroyAllWindows()
            return

        # Drag no frame
        if event == cv2.EVENT_LBUTTONDOWN:
            s.drag_start  = (cx, cy)
            s.dragging    = True
            s.drag_rect   = None

        elif event == cv2.EVENT_MOUSEMOVE and s.dragging:
            x1 = min(s.drag_start[0], cx)
            y1 = min(s.drag_start[1], cy)
            x2 = max(s.drag_start[0], cx)
            y2 = max(s.drag_start[1], cy)
            s.drag_rect = (x1, y1, x2, y2)

        elif event == cv2.EVENT_LBUTTONUP and s.dragging:
            s.dragging = False
            if s.drag_start is None:
                return
            x1 = min(s.drag_start[0], cx)
            y1 = min(s.drag_start[1], cy)
            x2 = max(s.drag_start[0], cx)
            y2 = max(s.drag_start[1], cy)
            s.drag_rect  = None
            s.drag_start = None

            if (x2 - x1) < 4 or (y2 - y1) < 4:
                return

            # Determina mesa pelo centro do retangulo
            cx_mid = (x1 + x2) // 2
            cy_mid = (y1 + y2) // 2
            tid, _, _ = disp_to_crop(cx_mid, cy_mid, frame_x, scale, mid_frame)

            # Converte corners para crop space
            _, crx1, cry1 = disp_to_crop(x1, y1, frame_x, scale, mid_frame)
            _, crx2, cry2 = disp_to_crop(x2, y2, frame_x, scale, mid_frame)
            crop_coords = (crx1, cry1, crx2, cry2)

            # Sobrescreve item ativo com as coordenadas
            s.calibrated[(tid, s.active_item)] = crop_coords

            # Avanca automaticamente para o proximo item nao calibrado
            items = s.items()
            idx   = items.index(s.active_item)
            for nxt in items[idx + 1:] + items[:idx]:
                if not s.is_done(s.active_table, nxt):
                    s.active_item = nxt
                    break

    # ---------------------------------------------------------------------------
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, canvas_w, canvas_h)
    cv2.setMouseCallback(WIN, on_mouse)

    print(f"\nCalibracao iniciada  |  Frame {args.frame}  |  Resolucao {fw}x{fh}")
    print("Clique em um item no painel esquerdo, depois arraste no frame.")
    print("Q ou fechar janela = salvar e sair.\n")

    while True:
        canvas[:] = C_BG

        # Painel esquerdo
        draw_panel(canvas, s, canvas_h)

        # Frame + overlay
        frame_overlay = frame_scaled.copy()
        draw_frame_overlay(canvas, s, frame_overlay, frame_x,
                           mid_frame, scale)

        cv2.imshow(WIN, canvas)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("z"):
            undone = s.undo()
            if undone:
                t, it = undone
                print(f"Desfeito: Mesa {t} / {s.item_label(it)}")

        elif key == ord("q"):
            break

        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()

    if not s.calibrated:
        print("Nenhum item calibrado. Arquivo nao alterado.")
        return

    save_to_file(s)

    print(f"\n{'='*55}")
    print(f"  {s.total_done()} item(s) salvos em seat_regions.py")
    for (tid, item), coords in s.calibrated.items():
        print(f"    Mesa {tid} / {s.item_label(item):8s}  {coords}")
    print(f"  {SEAT_REGIONS_PATH.resolve()}")


if __name__ == "__main__":
    main()

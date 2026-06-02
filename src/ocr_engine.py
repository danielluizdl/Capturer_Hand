"""Task 2 — wrapper RapidOCR para layout WPT Global / Nexa Poker."""
from __future__ import annotations
import re
import threading
import cv2
import numpy as np

_ocr_instance = None
_ocr_lock = threading.Lock()
_USE_GPU = False


def _add_cuda_to_path() -> None:
    """
    Adiciona CUDA bin ao PATH do processo (necessário para DLLs nativas transitivas).
    Busca em: (1) CUDA Toolkit instalado no sistema, (2) pacotes pip nvidia-*.
    """
    import os, glob, sys

    dirs_to_add: list[str] = []

    # CUDA Toolkit do sistema
    cuda_base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if os.path.isdir(cuda_base):
        for vd in sorted(glob.glob(os.path.join(cuda_base, "v*")), reverse=True):
            for sub in ("bin", "lib\\x64"):
                d = os.path.join(vd, sub)
                if os.path.isdir(d):
                    dirs_to_add.append(d)

    # Pacotes pip nvidia-cudnn-cu12, nvidia-cublas-cu12, etc.
    for sp in sys.path:
        nvidia_dir = os.path.join(sp, "nvidia")
        if not os.path.isdir(nvidia_dir):
            continue
        for pkg in glob.glob(os.path.join(nvidia_dir, "*", "bin")):
            if os.path.isdir(pkg):
                dirs_to_add.append(pkg)

    for d in dirs_to_add:
        if d not in os.environ.get("PATH", ""):
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
        try:
            os.add_dll_directory(d)
        except Exception:
            pass

    # Pré-carrega DLLs CUDA via ctypes para que o loader nativo do onnxruntime
    # as encontre já no cache do processo (evita falha de busca em PATH).
    import ctypes, platform
    if platform.system() == "Windows":
        cuda_dlls = [
            "cudart64_110.dll", "cudart64_12.dll",
            "cublas64_11.dll", "cublasLt64_11.dll",
            "cublas64_12.dll", "cublasLt64_12.dll",
            "cudnn64_8.dll", "cudnn64_9.dll",
        ]
        for dll in cuda_dlls:
            try:
                ctypes.WinDLL(dll)
            except OSError:
                pass


def _detect_gpu() -> bool:
    """
    Retorna True apenas se CUDAExecutionProvider estiver listado e cublas64_12.dll
    existir no CUDA Toolkit instalado. Não tenta carregar a DLL (dependências transitivas
    tornam isso não-confiável); apenas verifica existência do arquivo.
    """
    try:
        import onnxruntime as ort
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            print(f"[OCR] CUDA não listado. Providers: {ort.get_available_providers()}")
            return False
        import os, glob, sys, platform
        if platform.system() == "Windows":
            # Coleta dirs de busca: Toolkit do sistema + pacotes pip nvidia-*
            search_dirs: list[str] = []
            cuda_base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
            if os.path.isdir(cuda_base):
                for vd in sorted(glob.glob(os.path.join(cuda_base, "v*")), reverse=True):
                    search_dirs.append(os.path.join(vd, "bin"))
            for sp in sys.path:
                nvidia_dir = os.path.join(sp, "nvidia")
                if os.path.isdir(nvidia_dir):
                    for pkg_bin in glob.glob(os.path.join(nvidia_dir, "*", "bin")):
                        search_dirs.append(pkg_bin)

            # Aceita CUDA 11 (cuDNN 8) ou CUDA 12 (cuDNN 9)
            cuda11 = {"cublas64_11.dll": False, "cudnn64_8.dll": False, "cudart64_110.dll": False}
            cuda12 = {"cublas64_12.dll": False, "cudnn64_9.dll": False}
            for d in search_dirs:
                for dll in list(cuda11):
                    if not cuda11[dll] and os.path.isfile(os.path.join(d, dll)):
                        cuda11[dll] = True
                for dll in list(cuda12):
                    if not cuda12[dll] and os.path.isfile(os.path.join(d, dll)):
                        cuda12[dll] = True

            has_cuda11 = all(cuda11.values())
            has_cuda12 = all(cuda12.values())

            if not has_cuda11 and not has_cuda12:
                missing11 = [dll for dll, found in cuda11.items() if not found]
                print(f"[OCR] CUDA não disponível. Faltando (CUDA 11): {missing11}")
                return False
        return True
    except Exception as e:
        print(f"[OCR] Erro ao verificar GPU: {e}")
        return False


def _ocr():
    global _ocr_instance, _USE_GPU
    if _ocr_instance is None:
        with _ocr_lock:
            if _ocr_instance is None:  # double-checked locking
                # Adiciona CUDA ao PATH antes de importar rapidocr/onnxruntime
                _add_cuda_to_path()
                from rapidocr_onnxruntime import RapidOCR
                _USE_GPU = _detect_gpu()
                if _USE_GPU:
                    try:
                        inst = RapidOCR(
                            det_use_cuda=True,
                            cls_use_cuda=True,
                            rec_use_cuda=True,
                        )
                        # Teste rápido — cuDNN 9 falha em GPUs Pascal (sm_61) ou mais antigas
                        inst(np.zeros((64, 64, 3), dtype=np.uint8))
                        _ocr_instance = inst
                        print("[OCR] Usando GPU (CUDA)")
                    except Exception as e:
                        print(f"[OCR] GPU falhou ({type(e).__name__}), usando CPU.")
                        print("[OCR] Nota: cuDNN 9 requer GPU Volta+ (GTX 10xx não suportada).")
                        _USE_GPU = False
                        _ocr_instance = RapidOCR()
                        print("[OCR] Usando CPU")
                else:
                    _ocr_instance = RapidOCR()
                    print("[OCR] Usando CPU")
    return _ocr_instance


def _raw_ocr(img: np.ndarray) -> list[tuple]:
    """Roda OCR e retorna [(bbox, text, conf), ...] filtrando conf < 0.5."""
    result, _ = _ocr()(img)
    if not result:
        return []
    return [(r[0], r[1], r[2]) for r in result if r[2] >= 0.5]


def ocr_title_bar(crop: np.ndarray) -> str | None:
    """
    OCR na barra de titulo. Retorna 'HL4017' ou None.
    ROI: primeiros 25px de altura, 500px de largura.
    """
    roi = crop[0:25, 0:500]
    items = _raw_ocr(roi)
    for _, text, _ in items:
        m = re.search(r'(HL\d+)', text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    # Fallback: OCR no crop completo (parte superior)
    roi2 = crop[0:40, 0:600]
    items2 = _raw_ocr(roi2)
    for _, text, _ in items2:
        m = re.search(r'(HL\d+)', text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return None


_POT_OCR_SUBS = str.maketrans({
    'O': '0', 'o': '0',      # O → 0
    'l': '1', 'I': '1',      # l/I → 1
    'S': '5',                  # S → 5
    'B': '8',                  # B → 8 (raro mas acontece)
})


def _parse_pot_value(text: str) -> float | None:
    """Tenta extrair valor numérico de pot de texto com possíveis erros de OCR."""
    # Aplica substituições de OCR no fragmento numérico
    cleaned = text.replace(",", ".").translate(_POT_OCR_SUBS)
    try:
        val = float(cleaned)
        # Sanidade: pot entre 0.5 BB e 500 BB
        if 0.5 <= val <= 500:
            return val
    except ValueError:
        pass
    return None


def ocr_pot(crop: np.ndarray) -> float | None:
    """
    OCR na area do pot. Retorna valor em BB como float (converte virgula PT-BR).
    Tenta múltiplas ROIs e padrões para maior robustez a erros de OCR.
    """
    h, w = crop.shape[:2]

    # ROIs primária e expandida
    rois = [
        crop[155:210, 230:750],
        crop[140:225, 200:800],   # ROI expandida como fallback
    ]

    # Padrões de reconhecimento do pot
    patterns = [
        # "PoteTotal: 21,5 BB" ou "Pote Total:21,5BB"
        re.compile(r'[Pp]o[Tt][Ee]\s*[Tt][Oo][Tt][Aa][Ll]\s*:?\s*([\d,\.O0lI]+)', re.I),
        # "Pot: 21.5 BB"
        re.compile(r'[Pp][Oo][Tt]\s*[Tt][Oo][Tt]\s*:?\s*([\d,\.O0lI]+)', re.I),
        # Número seguido de BB
        re.compile(r'([\d,\.O0lI]{2,})\s*BB', re.I),
        # Total: XX.X
        re.compile(r'[Tt]otal\s*:?\s*([\d,\.O0lI]+)', re.I),
    ]

    for roi in rois:
        if roi.size == 0:
            continue
        items = _raw_ocr(roi)
        if not items:
            continue
        all_text = " ".join(t for _, t, _ in items)

        for pat in patterns:
            m = pat.search(all_text)
            if m:
                val = _parse_pot_value(m.group(1))
                if val is not None:
                    return val

    return None


def ocr_stacks(crop: np.ndarray) -> dict[str, float]:
    """
    OCR de stacks dos jogadores. Retorna {nome: chips_bb}.
    Varre a imagem inteira em busca de padrões XX,XXBB e associa com
    nomes de jogadores próximos espacialmente.
    """
    result: dict[str, float] = {}

    items = _raw_ocr(crop)
    if not items:
        return result

    # Constrói lista de (bbox_center, text, conf)
    entries = []
    for bbox, text, conf in items:
        cx = float(np.mean([p[0] for p in bbox]))
        cy = float(np.mean([p[1] for p in bbox]))
        entries.append((cx, cy, text, conf))

    # Encontra todos os valores BB com sua posição
    bb_values: list[tuple[float, float, float]] = []
    name_candidates: list[tuple[float, float, str]] = []

    for cx, cy, text, conf in entries:
        m = re.search(r'([\d,\.O0lI]{2,})\s*BB', text, re.IGNORECASE)
        if m:
            try:
                val_str = m.group(1).replace(",", ".").translate(
                    str.maketrans({'O': '0', 'o': '0', 'l': '1', 'I': '1'})
                )
                val = float(val_str)
                if 0.5 <= val <= 2000:
                    bb_values.append((cx, cy, val))
            except ValueError:
                pass
        elif conf >= 0.7 and len(text.strip()) >= 3:
            clean = text.strip()
            # Candidato de nome: tem pelo menos 1 letra, não é número puro
            if any(c.isalpha() for c in clean) and not re.fullmatch(r'[\d,\.]+\s*BB?', clean, re.I):
                name_candidates.append((cx, cy, clean))

    # Associa cada BB ao nome mais próximo verticalmente
    for bx, by, bb in bb_values:
        best_name = None
        best_dist = 120.0  # max 120px de distância

        for nx, ny, name in name_candidates:
            # Mesma linha vertical (|dy| < 40) e horizontalmente próximo
            dy = abs(ny - by)
            dx = abs(nx - bx)
            if dy < 40 and dx < 300:
                dist = dy * 2 + dx * 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_name = name

        if best_name:
            result[best_name] = bb

    return result


def ocr_board_cards(crop: np.ndarray, n_cards: int) -> list[str]:
    """
    Extrai cartas do board via classificador cor+OCR.
    Fallback para lista vazia se classifier retornar incompleto.
    """
    if n_cards == 0:
        return []
    from src.card_classifier import extract_board_cards
    return extract_board_cards(crop, n_cards)


def ocr_hole_cards(crop: np.ndarray) -> list[str]:
    """Extrai hole cards do herói via classificador cor+OCR."""
    from src.card_classifier import extract_hole_cards
    return extract_hole_cards(crop)


_WINNER_TMPL: np.ndarray | None = None


def _get_winner_template() -> np.ndarray | None:
    """Carrega template winner.PNG (lazy, cached)."""
    global _WINNER_TMPL
    if _WINNER_TMPL is None:
        import pathlib
        p = pathlib.Path(__file__).parent.parent / "templates" / "winner.PNG"
        if p.exists():
            _WINNER_TMPL = cv2.imread(str(p))
    return _WINNER_TMPL


def _find_winner_badge(crop: np.ndarray) -> tuple[float, float] | None:
    """
    Detecta a posição do badge WINNER usando template matching.
    Retorna (cx, cy) em pixels do crop, ou None se não encontrado.
    """
    tmpl = _get_winner_template()
    if tmpl is None:
        return None
    ih, iw = crop.shape[:2]
    th, tw = tmpl.shape[:2]
    if th > ih or tw > iw:
        return None
    res = cv2.matchTemplate(crop, tmpl, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    if max_val < 0.55:
        return None
    return (max_loc[0] + tw / 2, max_loc[1] + th / 2)


_UI_SKIP_WINNER = {
    'log de jogos', 'carta alta', 'gongfuboy', 'poker', 'nexa',
    'jackpot', 'flips', 'master', 'spf', 'poner', 'maos', 'vpnp',
    'desistir', 'passar', 'aposta', 'aumentar', 'pagar', 'verificar',
    'log', 'jogos', 'mostrar', 'rabbit', 'hunt', 'carta', 'alta',
    'bb', 'potetotal', 'pote', 'total', 'winner', 'vencedor',
}


def _is_player_name(text: str) -> bool:
    """Retorna True se o texto parece ser nome de jogador (não UI noise)."""
    t = text.strip()
    if not (3 <= len(t) <= 25):
        return False
    if re.fullmatch(r'[\d,\.\s]+BB?', t, re.I):
        return False
    if re.fullmatch(r'[\d,\.]+', t):
        return False
    if not any(c.isalpha() for c in t):
        return False
    t_low = t.lower()
    if any(skip in t_low for skip in _UI_SKIP_WINNER):
        return False
    return True


def ocr_winner(crop: np.ndarray) -> str | None:
    """
    Detecta o vencedor pela proximidade espacial ao badge 'WINNER'.
    Estratégia:
    1. Template matching para localizar o badge (mais robusto).
    2. OCR em ROI ao redor do badge (mais eficiente que OCR no frame completo).
    3. Fallback OCR no frame completo se sem template.
    """
    h, w = crop.shape[:2]

    def _center(bbox) -> tuple[float, float]:
        return (
            float(np.mean([p[0] for p in bbox])),
            float(np.mean([p[1] for p in bbox])),
        )

    winner_pos = _find_winner_badge(crop)

    # Verificação rápida da hero zone sem OCR
    if winner_pos is not None:
        wx, wy = winner_pos
        if wy / h > 0.80 and 0.30 < wx / w < 0.70:
            return "dLzinN"

    # Escolhe ROI para OCR: ao redor do badge (se encontrado) ou full crop
    if winner_pos is not None:
        wx, wy = winner_pos
        # Faixa horizontal ampla + faixa vertical contendo nome (acima/abaixo do badge)
        roi_x1 = max(0,  int(wx) - 280)
        roi_x2 = min(w,  int(wx) + 280)
        roi_y1 = max(0,  int(wy) - 80)
        roi_y2 = min(h,  int(wy) + 120)
        ocr_roi = crop[roi_y1:roi_y2, roi_x1:roi_x2]
        x_offset, y_offset = roi_x1, roi_y1
    else:
        ocr_roi = crop
        x_offset, y_offset = 0, 0

    if ocr_roi.size == 0:
        return None

    result, _ = _ocr()(ocr_roi)
    candidates: list[tuple[str, float, float]] = []

    if result:
        for r in result:
            bbox, tx, conf = r[0], r[1], r[2]
            if conf < 0.55:
                continue
            tx_clean = tx.strip()

            # Badge via OCR: atualiza posição se template não encontrou
            if tx_clean.upper() in ('WINNER', 'VENCEDOR', 'WINN', 'WINNE'):
                if winner_pos is None:
                    cx, cy = _center(bbox)
                    winner_pos = (cx + x_offset, cy + y_offset)
                continue

            if _is_player_name(tx_clean):
                cx, cy = _center(bbox)
                candidates.append((tx_clean.rstrip('.…').strip(), cx + x_offset, cy + y_offset))

    if winner_pos is None:
        return None

    wx, wy = winner_pos

    if wy / h > 0.80 and 0.30 < wx / w < 0.70:
        return "dLzinN"

    if not candidates:
        if wy / h > 0.75:
            return "dLzinN"
        return None

    best_name, best_dist = None, float("inf")
    for name, cx, cy in candidates:
        dist = ((cx - wx) ** 2 + (cy - wy) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_name = name

    max_dist = 400 if _get_winner_template() is not None else 350
    if best_dist > max_dist:
        return None

    return best_name


# --- helpers ---

SUIT_MAP = {
    "♠": "s", "♣": "c", "♥": "h", "♦": "d",
    "♡": "h", "♢": "d", "♤": "s", "♧": "c",
    "s": "s", "c": "c", "h": "h", "d": "d",
    "S": "s", "C": "c", "H": "h", "D": "d",
}

VALID_RANKS = set("23456789TJQKA")


def _parse_cards_from_text(text: str) -> list[str]:
    """
    Extrai cartas de texto livre. Tenta múltiplos formatos:
    - "Jd 5s 2h" (standard notation)
    - "J♦ 5♠ 2♥" (unicode suits)
    - Ranks e suits separados
    """
    cards: list[str] = []
    seen: set[str] = set()

    # Normaliza: remove espaços extras
    text = text.replace("10", "T")

    # Pattern 1: rank+suit direto (Jd, 5s, Ah, etc.)
    for m in re.finditer(r'([2-9TJQKA])([sShHdDcC♠♣♥♦♡♢♤♧])', text):
        rank = m.group(1).upper()
        suit_raw = m.group(2)
        suit = SUIT_MAP.get(suit_raw, None)
        if suit and rank in VALID_RANKS:
            card = rank + suit
            if card not in seen:
                seen.add(card)
                cards.append(card)

    # Pattern 2: rank e suit separados por espaco
    if not cards:
        tokens = text.split()
        i = 0
        while i < len(tokens) - 1:
            rank = tokens[i].upper()
            suit_raw = tokens[i + 1]
            if rank in VALID_RANKS and suit_raw in SUIT_MAP:
                card = rank + SUIT_MAP[suit_raw]
                if card not in seen:
                    seen.add(card)
                    cards.append(card)
                i += 2
                continue
            i += 1

    return cards

"""
Extrai frames-chave do vídeo para anotação manual por Claude e treino CNN.

Workflow completo:
  1. python extract_key_frames.py video_longo_teste.mp4
     → Detecta eventos (process_video a 10fps)
     → Extrai frames em passagem única sequencial
     → Salva em key_frames/<mesa>/<mao_NNN>/
     → Gera key_frames/manifest.json

  2. (próxima sessão) Claude lê lotes de imagens e anota
     → python annotate_frames.py --batch 30
     → Salva key_frames/<mesa>/<mao>/annotations.json

  3. python compile_training_data.py
     → Junta anotações + crops → dataset CNN expandido
     → Gera gabarito_longo.txt

Estrutura de saída por mão:
  key_frames/
    manifest.json
    HL4017/
      hand_001/
        preflop.png          ← crop completo da mesa no início da mão
        flop.png             ← crop completo quando o flop aparece
        flop_slot_0.png      ← crop do slot 0 (1ª carta do flop)
        flop_slot_1.png      ← slot 1
        flop_slot_2.png      ← slot 2
        turn.png
        turn_slot_3.png
        river.png
        river_slot_4.png
        showdown_00.png      ← crop da mesa a cada 2s na janela de showdown
        showdown_01.png
        ...
        meta.json            ← metadados da mão (timestamps, eventos)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from src.video_pipeline import process_video, TableEvent
from src.detectors import crop_table
from src.card_classifier import SLOT_X, CARD_Y1_F, CARD_Y2_F, CLASSIFY_HW_F

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
SHOWDOWN_WINDOW_S = 18.0   # janela de busca de showdown após fim da mão
SHOWDOWN_STEP_S   = 2.0    # salva 1 frame a cada 2s na janela
MAX_SHOWDOWN_FRAMES = 9    # máximo de frames de showdown por mão


def _slot_crops(crop: np.ndarray, n_cards: int) -> list[np.ndarray]:
    """Retorna crops individuais de cada slot do board (n_cards slots)."""
    h, w = crop.shape[:2]
    y1 = int(h * CARD_Y1_F)
    y2 = int(h * CARD_Y2_F)
    hw = int(w * CLASSIFY_HW_F) + 4   # um pouco mais largo para melhor contexto

    crops = []
    for i in range(n_cards):
        cx = int(w * SLOT_X[i])
        x1 = max(0, cx - hw)
        x2 = min(w, cx + hw)
        crops.append(crop[y1:y2, x1:x2])
    return crops


def _collect_needed_frames(
    tid: int,
    hand_segs: list[dict],
    all_evs: list[TableEvent],
    native_fps: float,
) -> dict[int, list[dict]]:
    """
    Retorna {frame_idx: [{"hand_idx": N, "role": "flop|turn|river|preflop|showdown_00"}]}
    para todos os frames que precisamos extrair desta mesa.
    """
    needed: dict[int, list[dict]] = defaultdict(list)

    for hi, seg in enumerate(hand_segs):
        if not seg:
            continue

        hand_idx = hi + 1
        sorted_seg = sorted(seg, key=lambda e: e.timestamp)
        seg_end_ts = max(e.timestamp for e in seg)

        # Preflop — primeiro frame da mão (nomes, stacks)
        first_fi = sorted_seg[0].frame_idx
        needed[first_fi].append({"hand_idx": hand_idx, "role": "preflop"})

        # Board changes — flop / turn / river
        for ev in sorted_seg:
            if ev.event_type != "board_change":
                continue
            if ev.board_cards == 3:
                role = "flop"
            elif ev.board_cards == 4:
                role = "turn"
            elif ev.board_cards == 5:
                role = "river"
            else:
                continue
            # Pega o frame + 6 frames depois (animação da carta assentou)
            for offset in [0, 6, 12]:
                needed[ev.frame_idx + offset].append(
                    {"hand_idx": hand_idx, "role": role, "n_cards": ev.board_cards}
                )

        # Janela de showdown — a cada 2s após o fim da mão
        future = sorted(
            [e for e in all_evs if e.table_idx == tid
             and e.event_type == "new_hand" and e.timestamp > seg_end_ts],
            key=lambda e: e.timestamp,
        )
        next_ts = future[0].timestamp if future else seg_end_ts + SHOWDOWN_WINDOW_S
        sd_end = min(next_ts + 2.0, seg_end_ts + SHOWDOWN_WINDOW_S)

        t = max(0.0, seg_end_ts - 1.0)
        sd_idx = 0
        while t <= sd_end and sd_idx < MAX_SHOWDOWN_FRAMES:
            fi = int(t * native_fps)
            needed[fi].append(
                {"hand_idx": hand_idx, "role": f"showdown_{sd_idx:02d}"}
            )
            t += SHOWDOWN_STEP_S
            sd_idx += 1

    return dict(needed)


def _segment_hands(ev_list: list[TableEvent]) -> list[list[TableEvent]]:
    """Segmentação simples de mãos (cópia simplificada de hand_builder)."""
    sorted_evs = sorted(ev_list, key=lambda e: e.timestamp)
    hands, current, prev_board = [], [], -1
    for ev in sorted_evs:
        if prev_board > 0 and ev.board_cards == 0 and ev.event_type == "new_hand":
            if current:
                hands.append(current)
            current = []
        current.append(ev)
        prev_board = ev.board_cards
    if current:
        hands.append(current)
    return hands


def _save_frame(crop: np.ndarray, path: Path, n_cards: int | None = None):
    """Salva crop completo e, se n_cards fornecido, também os slots individuais."""
    cv2.imwrite(str(path), crop)
    if n_cards is not None:
        slots = _slot_crops(crop, n_cards)
        for i, slot in enumerate(slots):
            slot_path = path.with_name(path.stem + f"_slot_{i}.png")
            cv2.imwrite(str(slot_path), slot)


def extract(video_path: str, out_dir: str = "key_frames", pipeline_fps: float = 10.0):
    print(f"\n{'='*60}")
    print(f"  Extração de frames-chave")
    print(f"  Vídeo : {video_path}")
    print(f"  Saída : {out_dir}")
    print(f"{'='*60}\n")

    # ---------- Fase 1: detectar eventos ----------
    print("Fase 1/3 — Detectando eventos via pipeline...")
    events = process_video(video_path, fps=pipeline_fps)

    cap_info = cv2.VideoCapture(video_path)
    native_fps = cap_info.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_info.release()

    all_evs = [ev for ev_list in events.values() for ev in ev_list]
    print(f"  Eventos detectados: {sum(len(v) for v in events.values())}")

    # ---------- Fase 2: determinar frames necessários ----------
    print("\nFase 2/3 — Calculando frames necessários...")
    out_root = Path(out_dir)
    out_root.mkdir(exist_ok=True)

    table_needed: dict[int, dict[int, list[dict]]] = {}
    table_segs:   dict[int, list] = {}
    manifest_hands = []

    for tid, ev_list in events.items():
        if not ev_list:
            continue
        table_id = ev_list[0].table_id
        segs = _segment_hands(ev_list)
        table_segs[tid] = segs
        needed = _collect_needed_frames(tid, segs, all_evs, native_fps)
        table_needed[tid] = needed
        print(f"  Mesa {tid} ({table_id}): {len(segs)} mãos, {len(needed)} frames a extrair")

        # Cria pastas e meta.json por mão
        for hi, seg in enumerate(segs):
            hand_dir = out_root / table_id / f"hand_{hi+1:03d}"
            hand_dir.mkdir(parents=True, exist_ok=True)
            seg_sorted = sorted(seg, key=lambda e: e.timestamp)
            boards = [e.board_cards for e in seg_sorted if e.board_cards > 0]
            meta = {
                "table_id": table_id,
                "table_idx": tid,
                "hand_idx": hi + 1,
                "start_ts": seg_sorted[0].timestamp if seg_sorted else 0,
                "end_ts": max(e.timestamp for e in seg),
                "board_progression": sorted(set(boards)),
                "n_events": len(seg),
            }
            (hand_dir / "meta.json").write_text(json.dumps(meta, indent=2))
            manifest_hands.append({**meta, "folder": str(hand_dir)})

    # ---------- Fase 3: passagem sequencial no vídeo ----------
    print("\nFase 3/3 — Extraindo frames (passagem única)...")

    # Monta índice global: frame_idx → [(tid, role_info), ...]
    global_needed: dict[int, list[tuple[int, dict]]] = defaultdict(list)
    for tid, needed in table_needed.items():
        for fi, roles in needed.items():
            for role_info in roles:
                global_needed[fi].append((tid, role_info))

    all_indices = sorted(global_needed.keys())
    total_to_extract = len(all_indices)
    print(f"  Total de frames únicos a extrair: {total_to_extract}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERRO: não foi possível abrir {video_path}")
        sys.exit(1)

    saved = 0
    current_fi = 0

    for i, target_fi in enumerate(all_indices):
        if target_fi >= total_frames:
            continue

        # Avança até o frame alvo
        while current_fi < target_fi:
            cap.grab()
            current_fi += 1

        ret, frame = cap.read()
        current_fi += 1
        if not ret:
            continue

        # Processa cada (mesa, role) que precisa deste frame
        for tid, role_info in global_needed[target_fi]:
            ev_list = events.get(tid, [])
            if not ev_list:
                continue
            table_id = ev_list[0].table_id
            hand_idx = role_info["hand_idx"]
            role     = role_info["role"]
            n_cards  = role_info.get("n_cards")

            hand_dir = out_root / table_id / f"hand_{hand_idx:03d}"
            crop = crop_table(frame, tid)

            # Nomeia o arquivo
            if role in ("flop", "turn", "river"):
                fname = f"{role}.png"
                # Só salva o "melhor" frame (offset 0) como principal
                # Os outros offsets são para ter opção de fallback
                if target_fi == [fi for fi, roles in table_needed[tid].items()
                                  if any(r["hand_idx"]==hand_idx and r["role"]==role
                                         for r in roles)][0]:
                    _save_frame(crop, hand_dir / fname, n_cards)
                else:
                    _save_frame(crop, hand_dir / fname.replace(".png", f"_alt{target_fi}.png"), n_cards)
            else:
                fname = f"{role}.png"
                cv2.imwrite(str(hand_dir / fname), crop)

            saved += 1

        if (i + 1) % 100 == 0 or (i + 1) == total_to_extract:
            pct = (i + 1) / total_to_extract * 100
            print(f"  [{i+1:4d}/{total_to_extract}] {pct:.0f}%  frames salvos={saved}", end="\r")

    cap.release()
    print(f"\n  Extração concluída — {saved} arquivos salvos")

    # ---------- Gera manifest.json ----------
    manifest = {
        "video": video_path,
        "native_fps": native_fps,
        "total_frames": total_frames,
        "pipeline_fps": pipeline_fps,
        "total_hands": len(manifest_hands),
        "output_dir": str(out_root.resolve()),
        "hands": manifest_hands,
    }
    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    print(f"\nManifest salvo: {manifest_path}")
    print(f"Total de mãos: {len(manifest_hands)}")
    for tid, segs in table_segs.items():
        tid_ev = events.get(tid, [])
        tid_id = tid_ev[0].table_id if tid_ev else f"mesa{tid}"
        print(f"  {tid_id}: {len(segs)} mãos")

    print(f"\nPróximo passo:")
    print(f"  python annotate_frames.py --dir {out_dir} --batch 30")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Extrai frames-chave para anotação")
    ap.add_argument("video", help="Caminho do vídeo (.mp4)")
    ap.add_argument("--out",  default="key_frames", help="Pasta de saída (padrão: key_frames/)")
    ap.add_argument("--fps",  type=float, default=10.0, help="FPS de amostragem do pipeline (padrão: 10)")
    args = ap.parse_args()
    extract(args.video, args.out, args.fps)

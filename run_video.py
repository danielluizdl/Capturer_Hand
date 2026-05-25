"""
Roda o pipeline em qualquer vídeo e gera output_hh.txt com as mãos detectadas.

Uso:
    python run_video.py <caminho_do_video>
    python run_video.py <caminho_do_video> --out resultado.txt
    python run_video.py <caminho_do_video> --fps 5
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Caminho do vídeo a processar")
    parser.add_argument("--out",  default=None, help="Arquivo de saída (padrão: <nome_do_video>_output.txt)")
    parser.add_argument("--fps",  type=float, default=None,  help="FPS de amostragem (padrão: usa config.json ou 10)")
    parser.add_argument("--config", default="config.json",   help="Config JSON (padrão: config.json)")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"ERRO: vídeo não encontrado: {args.video}")
        sys.exit(1)

    # Nome do output baseado no nome do vídeo
    if args.out is None:
        base = os.path.splitext(os.path.basename(args.video))[0]
        args.out = f"{base}_output.txt"

    # Carrega config
    cfg: dict = {}
    if os.path.exists(args.config):
        with open(args.config) as f:
            cfg = json.load(f)

    from agent_loop import apply_config, DEFAULT_CONFIG
    for k, v in DEFAULT_CONFIG.items():
        if k not in cfg:
            cfg[k] = v

    if args.fps is not None:
        cfg["fps"] = args.fps

    apply_config(cfg)

    from src.video_pipeline import process_video
    from src.hand_builder import build_hands
    from src.hh_writer_ps import write_session

    print(f"Vídeo:  {args.video}")
    print(f"FPS:    {cfg['fps']}")
    print(f"Saída:  {args.out}")
    print()

    t0 = time.perf_counter()
    events = process_video(args.video, fps=cfg["fps"])
    detected = build_hands(events, args.video)
    elapsed = time.perf_counter() - t0

    print(f"\nPipeline concluído em {elapsed:.1f}s")
    print(f"Mãos detectadas: {len(detected)}")
    print()

    if not detected:
        print("Nenhuma mão detectada.")
        sys.exit(0)

    # Imprime resumo de cada mão
    print("=" * 60)
    for i, hh in enumerate(detected, 1):
        board_str = " ".join(hh.board) if hh.board else "(sem board)"
        holes_str = " ".join(hh.hole_cards) if hh.hole_cards else "(não detectadas)"
        streets_str = " > ".join(hh.streets)
        print(f"Mão {i}: [{hh.table_id}]")
        print(f"  Streets:    {streets_str}")
        print(f"  Board:      {board_str}")
        print(f"  Hole cards: {holes_str}")
        print(f"  Pot:        ${hh.total_pot:.2f}")
        print(f"  Vencedor:   {hh.winner or '(não detectado)'}")
        print()

    # Gera arquivo de saída no formato PokerStars
    hh_text = write_session(detected)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(hh_text)

    print(f"Output salvo em: {args.out}")
    print(f"Total de mãos:   {len(detected)}")


if __name__ == "__main__":
    main()

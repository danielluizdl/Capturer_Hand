"""
Compara HandHistory detectado pelo pipeline com o gabarito.txt.

Uso:
    python compare_hh.py                  # usa config.json e gabarito.txt padrão
    python compare_hh.py --diff           # mostra diff unificado no terminal
    python compare_hh.py --write out.txt  # salva HH gerado em arquivo
"""
from __future__ import annotations
import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diff",  action="store_true", help="Exibe diff unificado vs gabarito")
    parser.add_argument("--write", metavar="FILE",      help="Salva HH gerado em arquivo")
    parser.add_argument("--video", default="video_cortado_1min.mp4")
    parser.add_argument("--gabarito", default="gabarito.txt")
    parser.add_argument("--config",   default="config.json")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"ERRO: vídeo não encontrado: {args.video}")
        sys.exit(1)

    # Carrega config
    cfg: dict = {}
    if os.path.exists(args.config):
        with open(args.config) as f:
            cfg = json.load(f)

    # Importa e aplica config
    from agent_loop import apply_config, DEFAULT_CONFIG
    for k, v in DEFAULT_CONFIG.items():
        if k not in cfg:
            cfg[k] = v
    apply_config(cfg)

    # Roda pipeline
    from src.video_pipeline import process_video
    from src.hand_builder   import build_hands
    from src.gabarito_parser import parse_gabarito
    from src.hh_writer_ps import write_session, diff_vs_gabarito, compare_key_lines

    print("Rodando pipeline…")
    events   = process_video(args.video, fps=cfg["fps"])
    detected = build_hands(events, args.video)

    print(f"Mãos detectadas: {len(detected)}  ({[h.table_id for h in detected]})")

    gabarito = parse_gabarito(args.gabarito)
    print(f"Gabarito:        {len(gabarito)} mãos ({[h.table_id for h in gabarito]})")

    # Comparação campo a campo
    print()
    compare_key_lines(detected, gabarito)

    # HH gerado
    hh_text = write_session(detected)

    if args.write:
        with open(args.write, "w", encoding="utf-8") as f:
            f.write(hh_text)
        print(f"\nHH gerado salvo em: {args.write}")

    if args.diff:
        print("\n" + "="*60)
        print("DIFF (gabarito → pipeline)")
        print("="*60)
        d = diff_vs_gabarito(hh_text, args.gabarito)
        if d:
            print(d)
        else:
            print("(sem diferenças nas linhas normalizadas)")


if __name__ == "__main__":
    main()

"""
Loop de melhoria iterativa para detecção de ações + showdown.

Rubric estendido:
  Tier 1 (300 pts) — já completo: table_id, board, hole_cards, streets, pot, winner
  Tier 2 (300 pts) — novo: preflop/flop/turn/river_actions + showdown

Total: 600 pts. Objetivo: 600/600 (100%).

Uso:
    python agent_loop_v2.py          # roda uma iteração e mostra score
    python agent_loop_v2.py --loop   # loop contínuo até 100% ou sem progresso
"""
from __future__ import annotations
import argparse
import json
import sys
import time
import traceback

VIDEO_PATH    = "video_cortado_1min.mp4"
GABARITO_PATH = "gabarito.txt"
CONFIG_PATH   = "config.json"
MAX_NO_PROGRESS = 3


def load_config() -> dict:
    from agent_loop import DEFAULT_CONFIG, load_config as _lc
    return _lc()


def run_pipeline(cfg: dict):
    from agent_loop import apply_config
    apply_config(cfg)
    from src.video_pipeline import process_video
    from src.hand_builder import build_hands
    events   = process_video(VIDEO_PATH, fps=cfg["fps"])
    detected = build_hands(events, VIDEO_PATH)
    return detected


def compute_score(detected, gabarito) -> dict:
    from src.scorer import score_hands
    return score_hands(detected, gabarito)


def print_score_report(report: dict) -> None:
    t1  = report["total_t1"]
    t2  = report["total_t2"]
    all_ = report["total_all"]
    print()
    print("=" * 60)
    print(f"  TIER 1 (base):    {t1:3d}/300  ({report['pct_t1']:.1f}%)")
    print(f"  TIER 2 (ações):   {t2:3d}/300  ({report['pct_t2']:.1f}%)")
    print(f"  TOTAL:            {all_:3d}/600  ({report['pct_all']:.1f}%)")
    print("=" * 60)
    print()

    for hs in report["hands"]:
        tid = hs["table_id"]
        s1  = hs["total_t1"]
        s2  = hs["total_t2"]
        print(f"  {tid}:  T1={s1}/100  T2={s2}/100")

        # Tier 2 breakdown
        t2s = hs["scores_t2"]
        gab_acts = hs["gab_actions"]
        det_acts = hs["det_actions"]
        for street, key in [("preflop","preflop_actions"),("flop","flop_actions"),
                             ("turn","turn_actions"),("river","river_actions")]:
            gab_s = [(a["player"], a["action"]) for a in gab_acts if a["street"] == street]
            det_s = [(a["player"], a["action"]) for a in det_acts if a["street"] == street]
            pts   = t2s[key]
            from src.scorer import RUBRIC_T2
            max_p = RUBRIC_T2[key]
            mark  = "✓" if pts == max_p else f"det={len(det_s)} gab={len(gab_s)}"
            print(f"    {street:8s} actions: {pts:2d}/{max_p}  {mark}")

        # Showdown
        sd_pts = t2s["showdown"]
        sd_max = RUBRIC_T2["showdown"] if "RUBRIC_T2" in dir() else 10
        gab_sd = hs["gab_showdown"]
        det_sd = hs["det_showdown"]
        mark = "✓" if not gab_sd else f"gab={list(gab_sd)} det={list(det_sd)}"
        print(f"    showdown        : {sd_pts:2d}/10  {mark}")
        print()


def analyze_gap(report: dict) -> str:
    """Retorna string descrevendo onde está o maior gap."""
    from src.scorer import RUBRIC_T2
    gaps = []
    for hs in report["hands"]:
        t2s  = hs["scores_t2"]
        tid  = hs["table_id"]
        gab  = hs["gab_actions"]
        det  = hs["det_actions"]
        for street, key in [("preflop","preflop_actions"),("flop","flop_actions"),
                             ("turn","turn_actions"),("river","river_actions")]:
            max_p = RUBRIC_T2[key]
            gap   = max_p - t2s[key]
            if gap > 0:
                gab_s = [(a["player"], a["action"]) for a in gab if a["street"] == street]
                det_s = [(a["player"], a["action"]) for a in det if a["street"] == street]
                gaps.append((gap, tid, key, gab_s, det_s))
        sd_gap = RUBRIC_T2["showdown"] - t2s["showdown"]
        if sd_gap > 0:
            gaps.append((sd_gap, tid, "showdown", hs["gab_showdown"], hs["det_showdown"]))

    if not gaps:
        return "Sem gaps — 600/600! 🎉"

    gaps.sort(reverse=True)
    top = gaps[0]
    gap, tid, key, gab_v, det_v = top
    return (f"Maior gap: {tid}/{key} (-{gap} pts)\n"
            f"  Gabarito: {gab_v}\n"
            f"  Detectado: {det_v}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", action="store_true", help="Loop contínuo")
    args = parser.parse_args()

    from src.gabarito_parser import parse_gabarito
    gabarito = parse_gabarito(GABARITO_PATH)
    cfg = load_config()

    best_t2   = -1
    no_prog   = 0
    iteration = 0

    while True:
        iteration += 1
        print(f"\n{'─'*60}")
        print(f"Iteração {iteration}")
        print(f"{'─'*60}")

        try:
            t0       = time.time()
            detected = run_pipeline(cfg)
            elapsed  = time.time() - t0
            print(f"Pipeline: {elapsed:.1f}s  ({len(detected)} mãos detectadas)")

            report = compute_score(detected, gabarito)
            print_score_report(report)

            t2 = report["total_t2"]
            if t2 > best_t2:
                best_t2 = t2
                no_prog = 0
                print(f"▲ Novo melhor Tier 2: {best_t2}/300")
            else:
                no_prog += 1
                print(f"→ Sem melhora ({no_prog}/{MAX_NO_PROGRESS})")

            if report["total_all"] >= 600:
                print("\n🎉 SCORE MÁXIMO 600/600! Pipeline completo.")
                break

            print()
            print("GAP:")
            print(analyze_gap(report))

            if not args.loop:
                break

            if no_prog >= MAX_NO_PROGRESS:
                print(f"\nSem progresso por {MAX_NO_PROGRESS} iterações. Parando.")
                break

        except Exception:
            traceback.print_exc()
            if not args.loop:
                break

    print(f"\nMelhor Tier 2 atingido: {best_t2}/300")


if __name__ == "__main__":
    main()

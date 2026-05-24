"""
Task 7 — loop principal de melhoria iterativa.

Executa: pipeline → score → analisa erros → ajusta → repete.
Para quando score = 100% ou sem progresso por MAX_NO_PROGRESS iterações.
"""
from __future__ import annotations
import json
import os
import sys
import time
import traceback

from src.hh_writer_ps import write_session, compare_key_lines

VIDEO_PATH   = "video_cortado_1min.mp4"
GABARITO_PATH = "gabarito.txt"
CONFIG_PATH   = "config.json"
RESULTS_PATH  = "RESULTS.md"
MAX_NO_PROGRESS = 3

# Config padrão — pode ser modificada pelo loop
DEFAULT_CONFIG = {
    "fps": 10.0,
    "slot_thresholds": [25, 33, 38, 25, 30],
    "pot_roi": [155, 210, 230, 750],
    "board_roi": [190, 310, 250, 700],
    "hole_roi": [430, 520, 330, 640],
    "winner_roi": [150, 400, 100, 860],
    "pot_diff_thr": 4.0,
    "action_bar_top_f": 0.900,
    "spades_v_threshold": 90,
    "classify_hw_f": 0.028,
}


def load_config() -> dict:
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            cfg = json.load(f)
        # Merge com defaults para novos campos
        for k, v in DEFAULT_CONFIG.items():
            if k not in cfg:
                cfg[k] = v
        return cfg
    return DEFAULT_CONFIG.copy()


def save_config(cfg: dict) -> None:
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


def apply_config(cfg: dict) -> None:
    """Aplica configurações aos módulos src."""
    import src.detectors as det
    import src.ocr_engine as ocr

    # Atualiza thresholds dos slots
    det.SLOT_THRESHOLDS[:] = cfg["slot_thresholds"]
    det.ACTION_BAR_TOP_F = cfg.get("action_bar_top_f", 0.900)

    # Armazena config globalmente para uso no ocr_engine
    ocr._CONFIG = cfg

    # Aplica config do classificador de cartas
    try:
        import src.card_classifier as cc
        cc.SPADES_V_THRESHOLD = cfg.get("spades_v_threshold", 90)
        cc.CLASSIFY_HW_F      = cfg.get("classify_hw_f", 0.028)
    except ImportError:
        pass


def run_pipeline(cfg: dict) -> tuple:
    """Roda o pipeline completo e retorna (events, detected_hands)."""
    from src.video_pipeline import process_video
    from src.hand_builder import build_hands

    events = process_video(VIDEO_PATH, fps=cfg["fps"])
    detected = build_hands(events, VIDEO_PATH)
    return events, detected


def analyze_gaps(report: dict) -> list[dict]:
    """Identifica os maiores gaps no score e sugere ajustes."""
    gaps = []
    for hs in report["hands"]:
        for field, pts in hs["scores"].items():
            max_pts = sum(v for k, v in
                          {"table_id":10,"board_cards":30,"hole_cards":20,
                           "street_sequence":15,"final_pot":15,"winner":10}.items()
                          if k == field)
            gap = max_pts - pts
            if gap > 0:
                gaps.append({
                    "table_id": hs["table_id"],
                    "field":    field,
                    "gap":      gap,
                    "pts":      pts,
                    "max":      max_pts,
                    "data":     hs,
                })
    return sorted(gaps, key=lambda x: x["gap"], reverse=True)


def suggest_adjustment(gaps: list[dict], cfg: dict, history: list[float]) -> dict | None:
    """
    Analisa os maiores gaps e sugere uma modificação na config.
    Retorna novo cfg ou None se não houver mais ajustes a fazer.
    """
    if not gaps:
        return None

    top = gaps[0]
    new_cfg = cfg.copy()
    new_cfg["slot_thresholds"] = list(cfg["slot_thresholds"])

    field  = top["field"]
    tid    = top["table_id"]

    if field == "board_cards":
        det_board = top["data"]["det_board"]
        gab_board = top["data"]["gab_board"]
        curr_v = new_cfg.get("spades_v_threshold", 90)
        curr_hw = new_cfg.get("classify_hw_f", 0.028)
        if len(det_board) < len(gab_board):
            # Detectando menos cartas que o esperado
            # Tentar limiar de espadas mais alto (misclassificando copas como espadas?)
            new_cfg["spades_v_threshold"] = min(130, curr_v + 5)
            # Também tentar janela de classificação mais larga
            new_cfg["classify_hw_f"] = min(0.040, curr_hw + 0.003)
            print(f"Ajuste board: spades_v_threshold {curr_v}→{new_cfg['spades_v_threshold']}, hw {curr_hw:.3f}→{new_cfg['classify_hw_f']:.3f}")
        elif len(det_board) > len(gab_board):
            new_cfg["spades_v_threshold"] = max(60, curr_v - 5)
            print(f"Ajuste board: spades_v_threshold {curr_v}→{new_cfg['spades_v_threshold']}")
        elif det_board and gab_board and any(
            d != g for d, g in zip(det_board, gab_board)
        ):
            # Quantidade certa mas cartas erradas — provavelmente rank OCR falhou
            # Tentar janela mais larga para dar mais contexto ao OCR
            new_cfg["classify_hw_f"] = min(0.040, curr_hw + 0.002)
            print(f"Ajuste board (rank): classify_hw_f {curr_hw:.3f}→{new_cfg['classify_hw_f']:.3f}")

    elif field == "street_sequence":
        det_s = set(top["data"]["det_streets"])
        gab_s = set(top["data"]["gab_streets"])
        missing = gab_s - det_s
        extra   = det_s - gab_s
        if extra:
            # Detectando streets extras → aumenta thresholds para reduzir falsos positivos
            for i in range(5):
                new_cfg["slot_thresholds"][i] = min(60, new_cfg["slot_thresholds"][i] + 2)
            print(f"Ajuste: streets extras detectadas, aumentando thresholds → {new_cfg['slot_thresholds']}")
        elif missing:
            for i in range(5):
                new_cfg["slot_thresholds"][i] = max(8, new_cfg["slot_thresholds"][i] - 2)
            print(f"Ajuste: streets faltando, reduzindo thresholds → {new_cfg['slot_thresholds']}")

    elif field == "final_pot":
        det_pot = top["data"]["det_pot"]
        gab_pot = top["data"]["gab_pot"]
        if det_pot is None:
            # OCR do pot falhou — amplia ROI
            r = new_cfg["pot_roi"]
            new_cfg["pot_roi"] = [
                max(0, r[0] - 10),
                min(540, r[1] + 10),
                max(0, r[2] - 20),
                min(960, r[3] + 20),
            ]
            print(f"Ajuste: ampliando pot_roi → {new_cfg['pot_roi']}")
        elif det_pot and abs(det_pot - gab_pot) / gab_pot > 0.05:
            # Pot detectado mas valor errado — pode ser em BB vs $
            # Tenta ajustar buscando potencialmente em outra área
            new_cfg["pot_diff_thr"] = max(2.0, cfg.get("pot_diff_thr", 4.0) - 0.5)
            print(f"Ajuste: pot_diff_thr → {new_cfg['pot_diff_thr']}")

    elif field == "winner":
        # Amplia área de busca do vencedor
        r = new_cfg["winner_roi"]
        new_cfg["winner_roi"] = [
            max(0, r[0] - 20),
            min(540, r[1] + 20),
            max(0, r[2] - 30),
            min(960, r[3] + 30),
        ]
        print(f"Ajuste: ampliando winner_roi → {new_cfg['winner_roi']}")

    elif field == "hole_cards":
        r = new_cfg["hole_roi"]
        new_cfg["hole_roi"] = [
            max(0, r[0] - 15),
            min(540, r[1] + 15),
            max(0, r[2] - 30),
            min(960, r[3] + 30),
        ]
        print(f"Ajuste: ampliando hole_roi → {new_cfg['hole_roi']}")

    # Não muda nada se config igual
    if new_cfg == cfg:
        return None

    return new_cfg


def write_results(history: list[dict]) -> None:
    """Documenta o resultado final em RESULTS.md."""
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write("# Resultados do Agent Loop\n\n")
        f.write(f"Total de iterações: {len(history)}\n\n")
        f.write("| Iteração | Score | Pts |\n")
        f.write("|----------|-------|-----|\n")
        for i, h in enumerate(history, 1):
            f.write(f"| {i} | {h['pct']:.1f}% | {h['pts']}/300 |\n")

        if history:
            best = max(history, key=lambda x: x["pts"])
            f.write(f"\n**Melhor score:** {best['pct']:.1f}% ({best['pts']}/300) na iteração {history.index(best)+1}\n")

            final = history[-1]
            if final["pts"] < 300:
                f.write("\n## Platô identificado\n\n")
                f.write("O score não progrediu por 3 iterações consecutivas.\n\n")
                f.write("### Breakdown da última iteração:\n\n")
                if "report" in final:
                    report = final["report"]
                    for hs in report.get("hands", []):
                        f.write(f"#### {hs['table_id']}: {hs['total']}/100\n")
                        for k, v in hs["scores"].items():
                            f.write(f"- {k}: {v}\n")
                        f.write("\n")


def main():
    print("=" * 60)
    print("AGENT LOOP — Capturer_Hand")
    print("=" * 60)

    if not os.path.exists(VIDEO_PATH):
        print(f"ERRO: {VIDEO_PATH} não encontrado na raiz do projeto.")
        sys.exit(1)

    from src.gabarito_parser import parse_gabarito
    from src.scorer import score_hands, print_report

    gabarito = parse_gabarito(GABARITO_PATH)
    print(f"Gabarito: {len(gabarito)} mãos carregadas ({[h.table_id for h in gabarito]})")

    cfg = load_config()
    save_config(cfg)

    score_history: list[float] = []
    iter_history:  list[dict]  = []
    no_progress_count = 0
    iteration = 0

    while True:
        iteration += 1
        print(f"\n{'=' * 52}")
        print(f"=== ITERAÇÃO {iteration} ===")
        print(f"{'=' * 52}")

        apply_config(cfg)

        t0 = time.perf_counter()
        try:
            events, detected = run_pipeline(cfg)
        except Exception as e:
            print(f"ERRO no pipeline: {e}")
            traceback.print_exc()
            break

        elapsed = time.perf_counter() - t0
        print(f"Pipeline concluído em {elapsed:.1f}s")
        print(f"HandHistories detectadas: {len(detected)} ({[h.table_id for h in detected]})")

        report = score_hands(detected, gabarito)
        print_report(report)

        hh_text = write_session(detected)
        with open("output_hh.txt", "w", encoding="utf-8") as f:
            f.write(hh_text)
        print("\n--- Comparacao campo a campo (PS format) ---")
        compare_key_lines(detected, gabarito)
        print("--- Output PS salvo em output_hh.txt ---")

        pct = report["pct"]
        pts = report["total_pts"]
        print(f"\nScore final: {pct:.1f}% ({pts}/300)")

        score_history.append(pct)
        iter_history.append({
            "pct":    pct,
            "pts":    pts,
            "config": cfg.copy(),
            "report": report,
        })

        # Critério de parada: 100%
        if pct >= 100.0:
            print("\n🎉 SCORE 100%! Pipeline perfeito.")
            break

        # Analisa gaps e sugere ajuste
        gaps = analyze_gaps(report)
        if gaps:
            print(f"\nMaior gap: {gaps[0]['field']} ({gaps[0]['table_id']}) — {gaps[0]['gap']} pts perdidos")

        # Verifica progresso
        if len(score_history) >= 2:
            improved = score_history[-1] > score_history[-2]
            if not improved:
                no_progress_count += 1
                print(f"Sem progresso ({no_progress_count}/{MAX_NO_PROGRESS})")
            else:
                no_progress_count = 0
                print(f"Progresso: {score_history[-2]:.1f}% → {score_history[-1]:.1f}%")

        if no_progress_count >= MAX_NO_PROGRESS:
            print(f"\nParando: sem progresso em {MAX_NO_PROGRESS} iterações consecutivas.")
            break

        # Sugere e aplica ajuste
        new_cfg = suggest_adjustment(gaps, cfg, score_history)
        if new_cfg is None:
            print("Sem mais ajustes possíveis.")
            break

        cfg = new_cfg
        save_config(cfg)

    # Salva resultados
    write_results(iter_history)
    print(f"\nResultados salvos em {RESULTS_PATH}")

    best_pct = max(h["pct"] for h in iter_history) if iter_history else 0
    print(f"\nMelhor score atingido: {best_pct:.1f}%")

    return best_pct


if __name__ == "__main__":
    main()

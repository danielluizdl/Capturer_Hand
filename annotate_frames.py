"""
Ferramenta de anotação de frames extraídos por Claude.

Uso:
    python annotate_frames.py --dir key_frames --batch 30
    python annotate_frames.py --dir key_frames --table HL4017
    python annotate_frames.py --dir key_frames --status  # mostra progresso

Claude lê cada imagem, anota o que vê, e salva em annotations.json por mão.

Formato de annotations.json:
{
  "table_id": "HL4017",
  "hand_idx": 1,
  "annotated": true,
  "board": {
    "flop": ["Jd", "5s", "2h"],
    "turn": ["3d"],
    "river": ["7s"]
  },
  "hero_hole_cards": ["9s", "4s"],
  "showdown": {
    "taymonkha": ["Ac", "Td"],
    "easycall86": ["Ah", "8c"]
  },
  "winner": "taymonkha",
  "pot_bb": 21.5,
  "notes": ""
}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ANNOTATION_TEMPLATE = {
    "annotated": False,
    "board": {"flop": [], "turn": [], "river": []},
    "hero_hole_cards": [],
    "showdown": {},
    "winner": "",
    "pot_bb": None,
    "notes": "",
}


def _load_manifest(dir_path: Path) -> dict:
    manifest_path = dir_path / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest.json não encontrado em {dir_path}.\n"
            f"Execute primeiro: python extract_key_frames.py <video>"
        )
    return json.loads(manifest_path.read_text())


def _get_annotation_path(hand_dir: Path) -> Path:
    return hand_dir / "annotations.json"


def _load_annotation(hand_dir: Path) -> dict:
    ann_path = _get_annotation_path(hand_dir)
    if ann_path.exists():
        return json.loads(ann_path.read_text())
    meta = json.loads((hand_dir / "meta.json").read_text())
    ann = dict(ANNOTATION_TEMPLATE)
    ann["table_id"] = meta["table_id"]
    ann["hand_idx"] = meta["hand_idx"]
    return ann


def _save_annotation(hand_dir: Path, ann: dict):
    _get_annotation_path(hand_dir).write_text(
        json.dumps(ann, indent=2, ensure_ascii=False)
    )


def status(dir_path: Path):
    """Mostra progresso de anotação por mesa."""
    manifest = _load_manifest(dir_path)
    print(f"\nStatus de anotação — {dir_path}")
    print(f"Total de mãos: {manifest['total_hands']}")

    by_table: dict[str, dict] = {}
    for hand in manifest["hands"]:
        tid = hand["table_id"]
        if tid not in by_table:
            by_table[tid] = {"total": 0, "annotated": 0, "partial": 0}
        by_table[tid]["total"] += 1

        hand_dir = Path(hand["folder"])
        ann_path = _get_annotation_path(hand_dir)
        if ann_path.exists():
            ann = json.loads(ann_path.read_text())
            if ann.get("annotated"):
                by_table[tid]["annotated"] += 1
            else:
                by_table[tid]["partial"] += 1

    print()
    total_done = 0
    total_all  = 0
    for tid, stats in sorted(by_table.items()):
        pct = stats["annotated"] / stats["total"] * 100
        print(f"  {tid}: {stats['annotated']}/{stats['total']} anotadas ({pct:.0f}%)"
              + (f"  [{stats['partial']} parciais]" if stats["partial"] else ""))
        total_done += stats["annotated"]
        total_all  += stats["total"]

    print(f"\nTotal: {total_done}/{total_all} ({100*total_done/max(total_all,1):.0f}%)")
    print(f"\nPróximo batch:")
    print(f"  python annotate_frames.py --dir {dir_path} --batch 30")


def list_pending(dir_path: Path, table_filter: str | None = None, batch: int = 30) -> list[dict]:
    """Retorna até `batch` mãos ainda não anotadas."""
    manifest = _load_manifest(dir_path)
    pending = []
    for hand in manifest["hands"]:
        if table_filter and hand["table_id"] != table_filter:
            continue
        hand_dir = Path(hand["folder"])
        ann_path = _get_annotation_path(hand_dir)
        if ann_path.exists():
            ann = json.loads(ann_path.read_text())
            if ann.get("annotated"):
                continue
        pending.append(hand)
        if len(pending) >= batch:
            break
    return pending


def print_batch_for_annotation(dir_path: Path, table_filter: str | None, batch: int):
    """
    Imprime as imagens de um lote para que Claude possa anotar.
    Cada imagem é mostrada com Read tool na conversa.
    """
    pending = list_pending(dir_path, table_filter, batch)
    if not pending:
        print("Todas as mãos já foram anotadas!")
        return

    print(f"\nLote de {len(pending)} mãos para anotar")
    print("=" * 60)
    print("INSTRUÇÃO PARA CLAUDE:")
    print("Para cada mão abaixo, leia as imagens e preencha o JSON.")
    print("Salve com save_annotation(hand_dir, annotation_dict).")
    print("=" * 60)

    for i, hand in enumerate(pending, 1):
        hand_dir = Path(hand["folder"])
        print(f"\n[{i}/{len(pending)}] {hand['table_id']} — Mão {hand['hand_idx']:03d}")
        print(f"  Pasta: {hand_dir}")
        print(f"  Timestamps: {hand['start_ts']:.1f}s — {hand['end_ts']:.1f}s")
        print(f"  Board progression: {hand['board_progression']}")
        print(f"  Imagens disponíveis:")
        for img in sorted(hand_dir.glob("*.png")):
            print(f"    {img.name}")
        print()


def save_batch_annotations(dir_path: Path, annotations: list[dict]):
    """
    Recebe lista de dicts com anotações e salva cada uma.
    Chamado pelo agente de anotação após processar um lote.
    """
    saved = 0
    for ann in annotations:
        table_id = ann.get("table_id")
        hand_idx = ann.get("hand_idx")
        if not table_id or not hand_idx:
            print(f"  AVISO: anotação sem table_id ou hand_idx: {ann}")
            continue
        hand_dir = dir_path / table_id / f"hand_{hand_idx:03d}"
        if not hand_dir.exists():
            print(f"  AVISO: pasta não encontrada: {hand_dir}")
            continue
        ann["annotated"] = True
        _save_annotation(hand_dir, ann)
        saved += 1
    print(f"  {saved} anotações salvas")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Gerencia anotações de frames")
    ap.add_argument("--dir",    default="key_frames", help="Pasta com frames extraídos")
    ap.add_argument("--batch",  type=int, default=30, help="Tamanho do lote (padrão: 30)")
    ap.add_argument("--table",  default=None, help="Filtrar por mesa (ex: HL4017)")
    ap.add_argument("--status", action="store_true", help="Mostrar progresso")
    args = ap.parse_args()

    d = Path(args.dir)
    if args.status:
        status(d)
    else:
        print_batch_for_annotation(d, args.table, args.batch)

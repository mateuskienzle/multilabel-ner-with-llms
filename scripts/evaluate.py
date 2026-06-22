import argparse
import json
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate NER predictions against gold annotations")
    parser.add_argument("--predictions", required=True, help="JSON file produced by run_ner.py")
    parser.add_argument("--output", default=None, help="Optional: save metrics to JSON file")
    parser.add_argument(
        "--filter",
        choices=["all", "parsed_only"],
        default="all",
        help=(
            "all (default): include every segment, treating parse errors as empty predictions. "
            "parsed_only: exclude segments where JSON parsing failed (parse_status=parse_error), "
            "measuring model quality on segments it successfully responded to."
        ),
    )
    return parser.parse_args()


def normalize(text: str) -> str:
    return text.strip().upper()


def to_pairs(entities: list[dict]) -> set[tuple]:
    """Convert entity list to set of (normalized_text, type) pairs."""
    pairs = set()
    for e in entities:
        for t in e.get("types", []):
            pairs.add((normalize(e.get("text", "")), t))
    return pairs


# ---------------------------------------------------------------------------
# Metric computations
# ---------------------------------------------------------------------------

def safe_div(num: float, den: float) -> float:
    return num / den if den > 0 else 0.0


def prf(tp: int, fp: int, fn: int) -> dict:
    p = safe_div(tp, tp + fp)
    r = safe_div(tp, tp + fn)
    f1 = safe_div(2 * p * r, p + r)
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}


def example_based(segments: list[dict]) -> dict:
    """Average precision, recall, F1 computed per segment."""
    precisions, recalls, f1s = [], [], []

    for seg in segments:
        gold = to_pairs(seg["gold"])
        pred = to_pairs(seg["predicted"])

        if not gold and not pred:
            precisions.append(1.0)
            recalls.append(1.0)
            f1s.append(1.0)
            continue

        tp = len(gold & pred)
        p = safe_div(tp, len(pred)) if pred else 0.0
        r = safe_div(tp, len(gold)) if gold else 0.0
        f1 = safe_div(2 * p * r, p + r)

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    return {
        "precision": round(sum(precisions) / len(precisions), 4),
        "recall": round(sum(recalls) / len(recalls), 4),
        "f1": round(sum(f1s) / len(f1s), 4),
    }


def micro(segments: list[dict]) -> dict:
    """Micro precision, recall, F1 pooling all (span, type) pairs."""
    tp = fp = fn = 0
    for seg in segments:
        gold = to_pairs(seg["gold"])
        pred = to_pairs(seg["predicted"])
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)
    return prf(tp, fp, fn)


def per_type(segments: list[dict]) -> dict[str, dict]:
    """Per-entity-type precision, recall, F1."""
    tp_map: dict[str, int] = defaultdict(int)
    fp_map: dict[str, int] = defaultdict(int)
    fn_map: dict[str, int] = defaultdict(int)

    for seg in segments:
        gold = to_pairs(seg["gold"])
        pred = to_pairs(seg["predicted"])

        for text, t in gold & pred:
            tp_map[t] += 1
        for text, t in pred - gold:
            fp_map[t] += 1
        for text, t in gold - pred:
            fn_map[t] += 1

    all_types = sorted(set(tp_map) | set(fp_map) | set(fn_map))
    return {
        t: prf(tp_map[t], fp_map[t], fn_map[t])
        for t in all_types
    }


def macro(type_metrics: dict[str, dict]) -> dict:
    """Macro F1: unweighted average of per-type metrics."""
    if not type_metrics:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    keys = ["precision", "recall", "f1"]
    return {
        k: round(sum(m[k] for m in type_metrics.values()) / len(type_metrics), 4)
        for k in keys
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_results(metrics: dict) -> None:
    def row(label, m):
        print(f"  {label:<40} P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f}")

    print("\n=== Resultados ===\n")
    print("[ Agregadas ]")
    row("Example-based", metrics["example_based"])
    row("Micro (span-level)", metrics["micro"])
    row("Macro", metrics["macro"])

    print("\n[ Por tipo de entidade ]")
    for t, m in sorted(metrics["per_type"].items()):
        row(t, m)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    with open(args.predictions, encoding="utf-8") as f:
        segments = json.load(f)

    # --- Parse status summary (only if field is present in the file) ---
    has_status = all("parse_status" in s for s in segments)
    if has_status:
        from collections import Counter
        status_counts = Counter(s["parse_status"] for s in segments)
        print(f"\nParse status summary ({len(segments)} segmentos total):")
        for status, count in sorted(status_counts.items()):
            pct = 100 * count / len(segments)
            print(f"  {status:<15} {count:>6}  ({pct:.1f}%)")
    else:
        print("(arquivo sem campo parse_status — produzido por versão anterior do run_ner.py)")

    # --- Filter ---
    if args.filter == "parsed_only":
        if not has_status:
            print("\nAviso: --filter parsed_only requer parse_status no arquivo. Usando todos os segmentos.")
        else:
            before = len(segments)
            segments = [s for s in segments if s.get("parse_status") != "parse_error"]
            excluded = before - len(segments)
            print(f"\n[--filter parsed_only] Excluídos {excluded} segmentos com parse_error.")

    print(f"\nSegmentos avaliados: {len(segments)}")

    type_metrics = per_type(segments)
    metrics = {
        "filter": args.filter,
        "n_segments": len(segments),
        "example_based": example_based(segments),
        "micro": micro(segments),
        "macro": macro(type_metrics),
        "per_type": type_metrics,
    }

    print_results(metrics)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"\nMétricas salvas em: {args.output}")


if __name__ == "__main__":
    main()


import json
from pathlib import Path

ROOT = Path("artifacts/tinystories")

print(
    f"{'Attention':<12}"
    f"{'Params (M)':>12}"
    f"{'Val CE':>12}"
    f"{'Perplexity':>14}"
    f"{'Train tok/s':>16}"
)

print("-" * 66)

for attention in ("mha", "gqa", "mqa", "mla"):
    path = ROOT / f"{attention}_train_seed42" / "results.json"

    if not path.exists():
        print(f"{attention:<12} Missing results")
        continue

    with path.open(encoding="utf-8") as f:
        result = json.load(f)

    params = result["trainable_parameters"] / 1_000_000
    loss = result["best_eval_ce"]
    ppl = result["best_eval_ppl"]
    tok_per_sec = result["tokens_per_second"]

    ppl_text = f"{ppl:.2f}" if ppl is not None else "N/A"

    print(
        f"{attention:<12}"
        f"{params:>12.2f}"
        f"{loss:>12.4f}"
        f"{ppl_text:>14}"
        f"{tok_per_sec:>16.1f}"
    )

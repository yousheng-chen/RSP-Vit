import argparse, csv, json
from pathlib import Path

def fmt(v, f=".2f"):
    if v != v or v is None: return "--"
    if v >= 1e6: return f"{v/1e6:.2f}M"
    return f"{v:{f}}"

def load(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            for k in r:
                try: r[k] = float(r[k])
                except: pass
            rows.append(r)
    return rows

def md_table(rows):
    lines = ["# Deployment Complexity Analysis", "",
         "| Model | Params (M) | FLOPs (G) | Latency (ms) | GPU Mem (MB) | Ckpt (MB) | Train Time (s) | Test Acc (%) | Best Val (%) |",
         "|:--|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for r in rows:
        lines.append(f"| {r['model']} | {fmt(r.get('total_params',0)/1e6)} | {fmt(r.get('flops_g',0))} | {fmt(r.get('latency_mean_ms',0),'.2f')} | {fmt(r.get('gpu_peak_mb',0),'.1f')} | {fmt(r.get('checkpoint_mb',0),'.2f')} | {fmt(r.get('training_time_sec',0),'.0f')} | {fmt(r.get('test_acc',0),'.2f')} | {fmt(r.get('best_val_acc',0),'.2f')} |")
    return "\n".join(lines)

def latex_table(rows):
    lines = ["%", "\\begin{table}[tb!]",
         "  \\caption{Deployment Complexity Analysis}\\label{tab:deployment}",
         "  \\setcolumns{|l|c|c|c|c|c|c|c|c|}",
         "  \\hline",
         "  Model & Params (M) & FLOPs (G) & Latency (ms) & GPU Mem (MB) & Ckpt (MB) & Train Time (s) & Test Acc (\\%) & Best Val (\\%) \\\\",
         "  \\hline"]
    for r in rows:
        lines.append(f"  {r['model']} & {fmt(r.get('total_params',0)/1e6)} & {fmt(r.get('flops_g',0))} & {fmt(r.get('latency_mean_ms',0),'.2f')} & {fmt(r.get('gpu_peak_mb',0),'.1f')} & {fmt(r.get('checkpoint_mb',0),'.2f')} & {fmt(r.get('training_time_sec',0),'.0f')} & {fmt(r.get('test_acc',0),'.2f')} & {fmt(r.get('best_val_acc',0),'.2f')} \\\\")
    lines.extend(["  \\hline", "\\end{table}", "%"])
    return "\n".join(lines)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output-dir", default="tables")
    p.add_argument("--noise-level", default="-10db")
    a = p.parse_args()

    inp = Path(a.input)
    if not inp.exists():
        print(f"[ERROR] Not found: {inp}")
        return

    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = load(inp)
    nl = a.noise_level
    filtered = [r for r in rows if r.get("noise_level", "") == nl]
    if filtered:
        rows = filtered
        print(f"Noise level: {nl}")
    else:
        print(f"Cannot filter by {nl}, using all rows")

    (out / "benchmark_table.md").write_text(md_table(rows), encoding="utf-8")
    print(f"  Markdown: {out / 'benchmark_table.md'}")

    (out / "benchmark_table.tex").write_text(latex_table(rows), encoding="utf-8")
    print(f"  LaTeX:    {out / 'benchmark_table.tex'}")
    print(f"\nAll tables saved to {out}")

if __name__ == "__main__":
    main()

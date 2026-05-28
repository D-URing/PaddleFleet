import re, os
import matplotlib.pyplot as plt
import numpy as np

PAT = re.compile(r"loss: ([\d.]+),.*?global_step: (\d+),")

def parse(path):
    steps, losses = [], []
    seen = set()
    with open(path, errors="ignore") as f:
        for line in f:
            m = PAT.search(line)
            if not m:
                continue
            l = float(m.group(1))
            s = int(m.group(2))
            if s in seen:
                continue
            seen.add(s)
            steps.append(s); losses.append(l)
    return np.array(steps), np.array(losses)

def ema(y, alpha=0.05):
    out = np.empty_like(y, dtype=float)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = alpha * y[i] + (1-alpha) * out[i-1]
    return out

base = "output"
runs = {
    "GQA pretrain (12k-24k)": (f"{base}/qwen3_gqa_1p7B_pretrain/trainer-12/workerlog.0", 12000, 24000, 0),
    "GQA continue_v3":         (f"{base}/qwen3_gqa_1p7B_continue_v3/logs/workerlog.0", 0, 1e9, 24000),
    "VHA warmup029":           (f"{base}/qwen3_vha_1p7B_warmup029/trainer/workerlog.0", 0, 1e9, 24000),
    "VHA warmup030":           (f"{base}/qwen3_vha_1p7B_warmup030/trainer/workerlog.0", 0, 1e9, 24000),
    "VHA warmup031":           (f"{base}/qwen3_vha_1p7B_warmup031/logs/workerlog.0", 0, 1e9, 24000),
}

colors = {
    "GQA pretrain (12k-24k)": "#444444",
    "GQA continue_v3":         "#0072B2",
    "VHA warmup029":           "#D55E00",
    "VHA warmup030":           "#CC79A7",
    "VHA warmup031":           "#009E73",
}

fig, ax = plt.subplots(figsize=(15, 7))

for name, (p, lo, hi, off) in runs.items():
    if not os.path.exists(p):
        print(f"MISSING: {p}")
        continue
    s, l = parse(p)
    if len(s) == 0:
        print(f"EMPTY: {name}")
        continue
    mask = (s >= lo) & (s <= hi)
    s, l = s[mask], l[mask]
    if len(s) == 0:
        continue
    x = s + off
    ax.plot(x, l, color=colors[name], alpha=0.20, linewidth=0.6)
    if len(l) >= 5:
        ax.plot(x, ema(l, 0.05), color=colors[name], linewidth=1.8,
                label=f"{name} (n={len(l)}, min={l.min():.4f})")
    print(f"{name}: {len(l)} pts, range=[{l.min():.4f}, {l.max():.4f}], last={l[-1]:.4f}")

ax.axvline(24000, color="red", linestyle="--", linewidth=1, alpha=0.6)
ax.text(24000, ax.get_ylim()[1] * 0.98 if ax.get_ylim()[1] < 5 else 4.0,
        " ckpt-24000\n (continue split)",
        color="red", fontsize=9, va="top")

ax.set_xlabel("Global step (unified timeline; pretrain + continuation aligned at 24000)")
ax.set_ylabel("Train loss (CE; warmup includes KL distill component before step 2000)")
ax.set_title("GQA pretrain (12k-24k) → continuation: GQA continue_v3 vs VHA warmup 029/030/031")
ax.set_ylim(2.3, 4.3)
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
out = "output/loss_compare_gqa_vha_continuation.png"
plt.savefig(out, dpi=130)
print("saved", out)

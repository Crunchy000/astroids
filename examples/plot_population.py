"""Charts for a population run: fitness and network size over rounds, and lineage.

    python examples/plot_population.py population/population.json
    # -> population/population_elo.png, population/population_lineage.png
"""

import argparse
import json
import os
import statistics

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
BEST = "#2a78d6"
MEDIAN = "#52514e"
BAND = "#e4e3df"
FOUNDER_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3 (validated all-pairs)


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=TEXT_2, labelsize=9, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="y", color=GRID, lw=0.8)


def plot_population(json_path, out_dir=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    with open(json_path) as f:
        data = json.load(f)
    rounds = data["rounds"]
    if not rounds:
        return None
    out_dir = out_dir or os.path.dirname(json_path)
    xs = [r["round"] for r in rounds]
    founders = data["founders"]
    fcolor = {f: FOUNDER_COLORS[i] if i < len(FOUNDER_COLORS) else "#9a9893" for i, f in enumerate(founders)}

    # ---- fitness and network size ----
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8, 6.6), sharex=True, facecolor=SURFACE,
                                  gridspec_kw={"height_ratios": [2.2, 1]})
    for a in (ax, ax2):
        _style(a)
    elo = [[r["elo"][m["id"]] for m in r["members"]] for r in rounds]
    best = [max(e) for e in elo]
    med = [statistics.median(e) for e in elo]
    ax.fill_between(xs, [min(e) for e in elo], best, color=BAND, lw=0, zorder=1, label="population range")
    ax.plot(xs, med, color=MEDIAN, lw=1.5, marker="o", ms=4, zorder=3, label="median member")
    ax.plot(xs, best, color=BEST, lw=2, marker="o", ms=6, mec=SURFACE, mew=2, zorder=4, label="best member")
    ax.axhline(1000, color=TEXT_2, lw=1.5, ls=(0, (5, 4)), zorder=2)
    ax.annotate("built-in bots 1000", (0, 1000), xytext=(4, 5), textcoords="offset points",
                xycoords=("axes fraction", "data"), color=TEXT_2, fontsize=9)
    last = rounds[-1]
    top = max(last["members"], key=lambda m: last["elo"][m["id"]])
    ax.annotate(f"{top['id']} ({top['founder']} line) {best[-1]}", (xs[-1], best[-1]), xytext=(8, 0),
                textcoords="offset points", va="center", color=TEXT, fontsize=9, fontweight="bold",
                bbox={"boxstyle": "round,pad=0.2", "fc": SURFACE, "ec": "none"}, zorder=5)
    for ev in data.get("events", []):  # selection events: thin ticks at the bottom of the panel
        ax.axvline(ev["round"] + 0.5, color=GRID, lw=0.8, zorder=0)
    lo, hi = min(min(e) for e in elo), max(max(best), 1000)
    pad = max(30, (hi - lo) * 0.1)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_title("Population Elo after each round", loc="left", color=TEXT, fontsize=12, fontweight="bold", pad=8)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3, frameon=False, fontsize=9, labelcolor=TEXT,
              borderaxespad=0.3)

    widths = [[m["widths"][0] for m in r["members"]] for r in rounds]
    ax2.plot(xs, [statistics.mean(w) for w in widths], color=BEST, lw=2, marker="o", ms=4, label="mean")
    ax2.plot(xs, [max(w) for w in widths], color=MEDIAN, lw=1.2, ls=":", marker="o", ms=3, label="largest")
    ax2.set_title("Hidden units per layer", loc="left", color=TEXT, fontsize=11, fontweight="bold", pad=6)
    ax2.legend(loc="upper left", ncol=2, frameon=False, fontsize=9, labelcolor=TEXT)
    ax2.set_ylim(min(min(w) for w in widths) - 32, max(max(w) for w in widths) + 48)
    ax2.set_xlabel("Round", color=TEXT_2, fontsize=10)
    ax2.set_xticks(xs)
    ax2.set_xlim(xs[0] - 0.5, xs[-1] + 0.5 + 0.25 * max(len(xs), 4))
    fig.suptitle(f"Population of {data['size']} ({data['rules']} rules): bottom quarter replaced by mutated "
                 "copies of the top quarter at each grey line", x=0.02, ha="left", color=TEXT_2, fontsize=9)
    fig.tight_layout()
    elo_png = os.path.join(out_dir, "population_elo.png")
    fig.savefig(elo_png, dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # ---- lineage: each slot over time, coloured by founder ----
    n = data["size"]
    fig, ax = plt.subplots(figsize=(8, 0.42 * n + 1.9), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    prev_ids = None
    for r in rounds:
        for slot, m in enumerate(r["members"]):
            ax.add_patch(plt.Rectangle((r["round"] - 0.5, slot), 1, 1, fc=fcolor[m["founder"]], ec=SURFACE, lw=2))
            if prev_ids is not None and prev_ids[slot] != m["id"]:  # a newborn copy
                ax.plot(r["round"] - 0.5, slot + 0.5, marker="o", ms=5, color=SURFACE, mec=TEXT, mew=1, zorder=3)
        prev_ids = [m["id"] for m in r["members"]]
    for slot, m in enumerate(last["members"]):
        ax.text(xs[-1] + 0.65, slot + 0.5, f"{m['id']}  {last['elo'][m['id']]}  ({m['widths'][0]} units)",
                va="center", color=TEXT, fontsize=8)
    ax.set_xlim(xs[0] - 0.5, xs[-1] + 3.4)
    ax.set_ylim(n, 0)
    ax.set_yticks([s + 0.5 for s in range(n)], [f"slot {s + 1}" for s in range(n)])
    ax.set_xticks(xs)
    ax.tick_params(colors=TEXT_2, labelsize=8, length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xlabel("Round  (○ = mutated copy replaces a weak member)", color=TEXT_2, fontsize=9)
    ax.set_title("Lineage: which founder each member descends from", loc="left", color=TEXT, fontsize=11,
                 fontweight="bold", pad=8)
    ax.legend(handles=[Patch(fc=fcolor[f], label=f"{f} line") for f in founders], loc="upper center",
              bbox_to_anchor=(0.5, -0.32 if n <= 6 else -0.2), ncol=len(founders), frameon=False, fontsize=9,
              labelcolor=TEXT)
    fig.tight_layout()
    lineage_png = os.path.join(out_dir, "population_lineage.png")
    fig.savefig(lineage_png, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return elo_png, lineage_png


def main():
    p = argparse.ArgumentParser()
    p.add_argument("population_json")
    args = p.parse_args()
    print("wrote", *plot_population(args.population_json))


if __name__ == "__main__":
    main()

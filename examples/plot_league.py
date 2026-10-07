"""Charts for a league run: Elo over rounds, and who-beats-whom in the latest round.

    python examples/plot_league.py league/league.json   # -> league/elo.png, league/matrix.png
"""

import argparse
import json
import os

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3 (validated all-pairs)
REFERENCE = "#52514e"
# Diverging blue (wins) <-> red (losses) around a neutral gray at 50%.
WIN, MID, LOSS = "#2a78d6", "#f0efec", "#e34948"


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=TEXT_2, labelsize=9, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def plot_league(json_path, out_dir=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    with open(json_path) as f:
        data = json.load(f)
    rounds = data["rounds"]
    if not rounds:
        return None
    out_dir = out_dir or os.path.dirname(json_path)
    members = data["members"]
    rules = data.get("rules", "classic")

    # ---- Elo over rounds ----
    fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=SURFACE)
    _style(ax)
    xs = [0] + [r["round"] for r in rounds]
    ax.axhline(1000, color=REFERENCE, lw=1.5, ls=(0, (5, 4)), zorder=2)
    ax.annotate("starting champion 1000", (0, 1000), xytext=(4, 5), textcoords="offset points",
                xycoords=("axes fraction", "data"), color=TEXT_2, fontsize=9)
    lo, hi = 1000, 1000
    ends = []
    for m, color in zip(members, SERIES):
        ys = [1000] + [r["elo"][m] for r in rounds]  # every member starts as the champion
        lo, hi = min(lo, *ys), max(hi, *ys)
        ax.plot(xs, ys, color=color, lw=2, marker="o", ms=6, mec=SURFACE, mew=2, zorder=3, label=m)
        ends.append([ys[-1], ys[-1], m])
    pad = max(30, (hi - lo) * 0.12)
    # Direct end labels, nudged apart so equal ratings don't overprint.
    gap = (hi - lo + 2 * pad) * 0.07
    ends.sort(key=lambda e: e[0])
    for k in range(1, len(ends)):
        ends[k][1] = max(ends[k][1], ends[k - 1][1] + gap)
    for y, label_y, m in ends:
        ax.annotate(f"{m} {y}", (xs[-1], y), xytext=(xs[-1] + 0.05 * max(xs[-1], 1), label_y),
                    textcoords="data", va="center", color=TEXT, fontsize=9, fontweight="bold", zorder=4,
                    bbox={"boxstyle": "round,pad=0.2", "fc": SURFACE, "ec": "none"})
    hi = max(hi, ends[-1][1] - pad / 2)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlim(-0.3, xs[-1] + 0.3 + 0.22 * max(xs[-1], 1))
    ax.set_xticks(xs)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_xlabel("League round", color=TEXT_2, fontsize=10)
    ax.set_title("Elo rating per team model", loc="left", color=TEXT, fontsize=12, fontweight="bold", pad=8)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=len(members), frameon=False,
              fontsize=9, labelcolor=TEXT, borderaxespad=0.3)
    fig.suptitle(f"League ({rules} rules): tournament after each round, starting champion = 1000",
                 x=0.02, ha="left", color=TEXT_2, fontsize=10)
    fig.tight_layout()
    elo_png = os.path.join(out_dir, "elo.png")
    fig.savefig(elo_png, dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # ---- who beats whom (latest round) ----
    last = rounds[-1]
    players = sorted(last["elo"], key=lambda p: -last["elo"][p])
    n = len(players)
    grid = [[None] * n for _ in range(n)]
    for i, a in enumerate(players):
        for j, b in enumerate(players):
            if a != b and f"{a}|{b}" in last["matrix"]:
                grid[i][j] = last["matrix"][f"{a}|{b}"][0]
    cmap = LinearSegmentedColormap.from_list("winloss", [LOSS, MID, WIN])
    fig, ax = plt.subplots(figsize=(1.25 * n + 2.6, 0.8 * n + 1.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for i in range(n):
        for j in range(n):
            v = grid[i][j]
            # 2px surface gap between cells
            ax.add_patch(plt.Rectangle((j, i), 1, 1, fc=cmap(v) if v is not None else SURFACE,
                                       ec=SURFACE, lw=2))
            if v is not None:
                rgb = cmap(v)[:3]
                dark = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2] < 0.55
                ax.text(j + 0.5, i + 0.5, f"{v:.0%}", ha="center", va="center", fontsize=10,
                        color="#ffffff" if dark else TEXT, fontweight="bold")
            else:
                ax.text(j + 0.5, i + 0.5, "–", ha="center", va="center", color=TEXT_2)
    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    ax.set_xticks([k + 0.5 for k in range(n)], players, rotation=0)
    ax.set_yticks([k + 0.5 for k in range(n)], [f"{p} ({last['elo'][p]})" for p in players])
    ax.xaxis.tick_top()
    ax.tick_params(colors=TEXT, labelsize=9, length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title(f"Row's win rate against column, round {last['round']}", loc="left", color=TEXT,
                 fontsize=11, fontweight="bold", pad=28)
    fig.tight_layout()
    matrix_png = os.path.join(out_dir, "matrix.png")
    fig.savefig(matrix_png, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return elo_png, matrix_png


def main():
    p = argparse.ArgumentParser()
    p.add_argument("league_json")
    args = p.parse_args()
    print("wrote", *plot_league(args.league_json))


if __name__ == "__main__":
    main()

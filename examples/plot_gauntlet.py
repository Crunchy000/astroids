"""Chart gauntlet results: each team's win rate against every opponent.

    python examples/gauntlet.py A.zip B.zip --names "A,B" --json gauntlet.json
    python examples/plot_gauntlet.py gauntlet.json --out results/gauntlet.png
"""

import argparse
import json
import math

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cd1", "#9a9893"]  # categorical slots 1-4, then neutral grey


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a win rate ``p`` over ``n`` battles (stays inside 0-1)."""
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(c - h, 0.0), min(c + h, 1.0)


def plot_gauntlet(data: dict, out: str, title: str | None = None) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    teams = list(data["teams"])
    opponents = list(dict.fromkeys(o for t in teams for o in data["teams"][t]["vs"]))
    n = data["battles"]
    fig, ax = plt.subplots(figsize=(8, 0.62 * len(opponents) * max(len(teams), 2) / 2 + 1.9), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    bar = 0.8 / len(teams)
    for i, team in enumerate(teams):
        vs = data["teams"][team]["vs"]
        for j, opp in enumerate(opponents):
            if opp not in vs:
                continue
            w = vs[opp][0]
            y = j + (i - (len(teams) - 1) / 2) * bar
            lo, hi = wilson(w, n)
            ax.barh(y, w, height=bar * 0.86, color=COLORS[i % len(COLORS)], zorder=2,
                    label=f"{team} (score {data['teams'][team]['score']:.0%})" if j == 0 else None)
            ax.errorbar(w, y, xerr=[[w - lo], [hi - w]], fmt="none", ecolor=TEXT_2, elinewidth=0.9, capsize=2,
                        zorder=3)
            ax.text(1.025, y, f"{w:.0%}", va="center", fontsize=8, color=TEXT_2)
    ax.axvline(0.5, color=TEXT_2, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.set_yticks(range(len(opponents)), [f"vs {o}" for o in opponents])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.09)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax.tick_params(colors=TEXT_2, labelsize=9, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_xlabel(f"Win rate ({n} battles per opponent, half on each side; bars show 95% intervals)",
                  color=TEXT_2, fontsize=9)
    ax.set_title(title or "Gauntlet: win rate against each opponent", loc="left", color=TEXT, fontsize=12,
                 fontweight="bold", pad=8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=min(len(teams), 2), frameon=False,
              fontsize=9, labelcolor=TEXT)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("json")
    p.add_argument("--out", default="gauntlet.png")
    p.add_argument("--title")
    args = p.parse_args()
    with open(args.json) as f:
        print("wrote", plot_gauntlet(json.load(f), args.out, args.title))


if __name__ == "__main__":
    main()

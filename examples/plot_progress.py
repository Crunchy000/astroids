"""Plot a training run's evaluation history against the built-in bot.

    pip install matplotlib
    python examples/plot_progress.py models/ppo_pilot_eval.json            # -> models/ppo_pilot_eval.png
    python examples/plot_progress.py models/ppo_pilot_eval.json --aim-anneal 4000000

``train_ppo.py`` writes ``<out>_eval.json`` at every evaluation and, when
matplotlib is installed, refreshes this chart next to it automatically.
"""

import argparse
import json
import os

# Reference palette tokens (light surface).
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
PILOT = "#2a78d6"  # categorical slot 1
BOT = "#52514e"    # reference line: secondary ink, dashed


def plot(json_path, png_path=None, aim_anneal=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(json_path) as f:
        data = json.load(f)
    hist, bot = data["history"], data["bot"]
    if not hist:
        return None
    aim_anneal = aim_anneal or data.get("aim_anneal")
    png_path = png_path or os.path.splitext(json_path)[0] + ".png"
    x = [h["timesteps"] / 1e6 for h in hist]

    panels = [(m["key"], m["title"], m["fmt"]) for m in data.get("metrics", [])] or [
        ("return", "Average reward per battle", "{:+.2f}"), ("kd", "Kills per death", "{:.2f}")]
    bot_label = data.get("bot_label", "Built-in bot (same seat)")
    agent_label = data.get("agent_label", "PPO agent")
    fig, axes = plt.subplots(2, 1, figsize=(8, 6.4), sharex=True, facecolor=SURFACE)
    for ax, (key, title, fmt) in zip(axes, panels):
        y = [h[key] for h in hist]
        ax.set_facecolor(SURFACE)
        ax.axhline(bot[key], color=BOT, lw=1.5, ls=(0, (5, 4)), zorder=2, label=bot_label)
        ax.plot(x, y, color=PILOT, lw=2, marker="o", ms=7, mec=SURFACE, mew=2, zorder=3,
                label=agent_label)
        # Direct labels: latest agent value and the bot's line.
        ax.annotate(fmt.format(y[-1]), (x[-1], y[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", color=TEXT, fontsize=10, fontweight="bold",
                    bbox={"boxstyle": "round,pad=0.2", "fc": SURFACE, "ec": "none"}, zorder=4)
        ax.annotate(f"bot {fmt.format(bot[key])}", (0, bot[key]), xytext=(4, 5),
                    textcoords="offset points", xycoords=("axes fraction", "data"),
                    color=TEXT_2, fontsize=9)
        if aim_anneal:
            ax.axvspan(0, aim_anneal / 1e6, color=GRID, alpha=0.45, lw=0, zorder=0)
        ax.set_title(title, loc="left", color=TEXT, fontsize=12, fontweight="bold", pad=8)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.tick_params(colors=TEXT_2, labelsize=9, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        lo, hi = min(y + [bot[key]]), max(y + [bot[key]])
        pad = (hi - lo) * 0.12 or 0.5
        ax.set_ylim(lo - pad, hi + pad)
    if aim_anneal:
        axes[0].annotate("aim bonus fading out", (aim_anneal / 1e6 / 2, 1), xycoords=("data", "axes fraction"),
                         xytext=(0, -4), textcoords="offset points", ha="center", va="top",
                         color=TEXT_2, fontsize=9)
    axes[1].set_xlabel(data.get("xlabel", "Training steps (millions)"), color=TEXT_2, fontsize=10)
    axes[1].set_xlim(0, max(max(x), (aim_anneal or 0) / 1e6, 0.1) * 1.1)
    axes[0].legend(loc="lower right", frameon=False, fontsize=9, labelcolor=TEXT)
    fig.suptitle(data.get("title", "PPO agent vs built-in bot, blue seat 0 (20 battles per point)"),
                 x=0.02, ha="left", color=TEXT_2, fontsize=10)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return png_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("eval_json")
    p.add_argument("--out", help="PNG path (default: next to the JSON)")
    p.add_argument("--aim-anneal", type=float, help="shade the aim-shaping phase (steps)")
    args = p.parse_args()
    print("wrote", plot(args.eval_json, args.out, args.aim_anneal))


if __name__ == "__main__":
    main()

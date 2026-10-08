"""Record one battle between any two teams, with sound.

    python examples/record_match.py --blue league/models/striker_r12.npz --red bots --out videos/striker_vs_bots.mp4
    python examples/record_match.py --blue league/models/striker_r12.npz --red league/models/pack_r12.npz

Each side is ``bots`` (or ``bots:<style>``, e.g. ``bots:wedge``), a league snapshot (``.npz``) or an SB3 model (``.zip``,
exported on the fly). ``--rules`` is ``competitive`` (default) or ``classic``.
The battle runs until it ends (time limit or wipe-out) and the result banner
is held for a few seconds.
"""

import argparse
import os
import tempfile

import imageio.v2 as imageio

from astroids_gym.render import Renderer
from league import FRAME_SKIP, Battle, battle_over, make_rules
from record_video import add_soundtrack

FPS = 30


def resolve(spec: str, tmp: str) -> str:
    """'bots' and .npz pass through; an SB3 .zip is exported to .npz."""
    if spec.startswith("bots") or spec.endswith(".npz"):
        return spec
    from stable_baselines3 import PPO
    from league import export_policy
    return export_policy(PPO.load(spec, device="cpu"), os.path.join(tmp, os.path.basename(spec) + ".npz"))


def label(spec: str) -> str:
    return spec if spec.startswith("bots") else os.path.splitext(os.path.basename(spec))[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--blue", required=True)
    p.add_argument("--red", default="bots")
    p.add_argument("--rules", choices=["competitive", "classic", "dogfight"], default="competitive")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--hold", type=float, default=3.0, help="seconds to hold the result at the end")
    p.add_argument("--caption", help="text at the bottom (default: who flies each side)")
    p.add_argument("--out", default="videos/match.mp4")
    p.add_argument("--no-sound", action="store_true")
    args = p.parse_args()

    cfg, oc = make_rules(args.rules)
    with tempfile.TemporaryDirectory() as tmp:
        blue, red = resolve(args.blue, tmp), resolve(args.red, tmp)
        battle = Battle(args.seed, cfg)
        sim = battle.sim
        sim.sfx = None if args.no_sound else []
        renderer = Renderer(sim, "rgb_array", FPS)
        renderer.caption = args.caption or f"BLUE: {label(args.blue)}   vs   RED: {label(args.red)}"
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        silent = args.out if args.no_sound else args.out + ".silent.mp4"
        frames = 0
        with imageio.get_writer(silent, fps=FPS, codec="libx264", quality=7, macro_block_size=8) as writer:
            writer.append_data(renderer.draw())
            while not battle_over(battle)[0]:
                battle.step({**battle.team_actions(0, blue, oc), **battle.team_actions(1, red, oc)})
                writer.append_data(renderer.draw())
                frames += 1
            for _ in range(int(args.hold * FPS)):
                writer.append_data(renderer.draw())
                frames += 1
        if not args.no_sound:
            add_soundtrack(silent, args.out, sim.sfx, (frames + 1) / FPS, cfg.width)
        renderer.close()
    w = sim.winner()
    print(f"wrote {args.out}: {('BLUE', 'RED')[w] + ' wins' if w >= 0 else 'draw'} "
          f"({sim.team_score(0):.1f} - {sim.team_score(1):.1f}, {sim.time:.0f}s; FRAME_SKIP={FRAME_SKIP})")


if __name__ == "__main__":
    main()

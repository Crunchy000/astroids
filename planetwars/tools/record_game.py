"""Record a game of the neural entry against one of their sample agents as an MP4 (and a still).

Runs in the official Python engine (vendored in entry/) on a league map:
    python tools/record_game.py --seed 3 --out game.mp4 --still still.png
Needs Pillow and an ffmpeg binary.
"""
import argparse
import math
import pathlib
import random
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "entry"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from agent import NeuralAgent  # noqa: E402
from agents.greedy_heuristic_agent import GreedyHeuristicAgent  # noqa: E402
from agents.random_agents import CarefulRandomAgent  # noqa: E402
from core.forward_model import ForwardModel  # noqa: E402
from core.game_state import GameParams, Player  # noqa: E402
from core.game_state_factory import GameStateFactory  # noqa: E402
from pwjax.maps import cog2026_params  # noqa: E402

SCALE = 2
W, H = 640 * SCALE, 480 * SCALE
TOP = 96
BG = (12, 17, 30)
COLORS = {Player.Player1: (77, 163, 255), Player.Player2: (255, 122, 69), Player.Neutral: (128, 134, 146)}
TEXT = (226, 230, 238)
MUTED = (140, 148, 164)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def font(size, bold=False):
    try:
        return ImageFont.truetype(BOLD if bold else FONT, size)
    except OSError:
        return ImageFont.load_default()


F_PLANET, F_FLEET, F_HEAD, F_SMALL, F_BIG = font(22, True), font(15), font(26, True), font(19), font(64, True)


def starfield(seed=1):
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (W, H + TOP), BG)
    d = ImageDraw.Draw(img)
    for _ in range(260):
        x, y = rng.uniform(0, W), rng.uniform(TOP, H + TOP)
        c = int(rng.uniform(40, 90))
        d.point((x, y), fill=(c, c, c + 15))
    return img


def totals(gs, player):
    planets = sum(p.n_ships for p in gs.planets if p.owner == player)
    fleets = sum(p.transporter.n_ships for p in gs.planets if p.transporter and p.transporter.owner == player)
    return planets + fleets, sum(p.owner == player for p in gs.planets)


def draw(gs, base, names, max_ticks, banner=None):
    img = base.copy()
    d = ImageDraw.Draw(img, "RGBA")
    # fleets: a faint trail to the target, then the ship
    for p in gs.planets:
        t = p.transporter
        if t is None:
            continue
        col = COLORS[t.owner]
        dest = gs.planets[t.destination_index].position
        sx, sy = t.s.x * SCALE, t.s.y * SCALE + TOP
        d.line([(sx, sy), (dest.x * SCALE, dest.y * SCALE + TOP)], fill=col + (50,), width=2)
        ang = math.atan2(t.v.y, t.v.x)
        size = 9 + min(t.n_ships, 200) ** 0.5
        pts = [(sx + math.cos(ang) * size, sy + math.sin(ang) * size),
               (sx + math.cos(ang + 2.5) * size * 0.7, sy + math.sin(ang + 2.5) * size * 0.7),
               (sx + math.cos(ang - 2.5) * size * 0.7, sy + math.sin(ang - 2.5) * size * 0.7)]
        d.polygon(pts, fill=col + (255,))
        d.text((sx, sy - size - 4), f"{int(t.n_ships)}", font=F_FLEET, fill=col + (230,), anchor="md")
    for p in gs.planets:
        col = COLORS[p.owner]
        cx, cy, r = p.position.x * SCALE, p.position.y * SCALE + TOP, p.radius * SCALE
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col + (70,), outline=col + (255,), width=3)
        d.text((cx, cy), f"{int(p.n_ships)}", font=F_PLANET, fill=TEXT, anchor="mm")
    # header: names, tick, share of all ships
    d.rectangle([0, 0, W, TOP], fill=(18, 24, 40))
    s1, n1 = totals(gs, Player.Player1)
    s2, n2 = totals(gs, Player.Player2)
    d.text((24, 14), names[0], font=F_HEAD, fill=COLORS[Player.Player1])
    d.text((W - 24, 14), names[1], font=F_HEAD, fill=COLORS[Player.Player2], anchor="ra")
    d.text((W / 2, 16), f"tick {gs.game_tick} / {max_ticks}", font=F_SMALL, fill=MUTED, anchor="ma")
    d.text((24, 48), f"{int(s1)} ships · {n1} planets", font=F_SMALL, fill=TEXT)
    d.text((W - 24, 48), f"{int(s2)} ships · {n2} planets", font=F_SMALL, fill=TEXT, anchor="ra")
    share = s1 / (s1 + s2) if s1 + s2 > 0 else 0.5
    x0, x1, y = 24, W - 24, 80
    d.rounded_rectangle([x0, y, x1, y + 8], radius=4, fill=COLORS[Player.Player2])
    d.rounded_rectangle([x0, y, x0 + (x1 - x0) * share, y + 8], radius=4, fill=COLORS[Player.Player1])
    if banner:
        d.rectangle([0, TOP + H / 2 - 70, W, TOP + H / 2 + 70], fill=(0, 0, 0, 150))
        d.text((W / 2, TOP + H / 2), banner, font=F_BIG, fill=TEXT, anchor="mm")
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--opponent", choices=["greedy", "random"], default="greedy")
    ap.add_argument("--out", default="game.mp4")
    ap.add_argument("--still", default=None, help="also save a PNG of the frame at --still-tick")
    ap.add_argument("--still-tick", type=int, default=300)
    ap.add_argument("--every", type=int, default=2, help="game ticks per video frame")
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    p = cog2026_params(rng)
    params = GameParams(num_planets=p.num_planets, initial_neutral_ratio=p.initial_neutral_ratio,
                        min_growth_rate=p.min_growth_rate, max_growth_rate=p.max_growth_rate,
                        transporter_speed=p.transporter_speed)
    gs = GameStateFactory(params).create_game()
    me = NeuralAgent()
    other = GreedyHeuristicAgent() if args.opponent == "greedy" else CarefulRandomAgent()
    me.prepare_to_play_as(Player.Player1, params)
    other.prepare_to_play_as(Player.Player2, params)
    names = ("Our model", "Their greedy agent" if args.opponent == "greedy" else "Their careful random agent")
    model = ForwardModel(gs, params)
    base = starfield()
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{W}x{H + TOP}", "-r", str(args.fps), "-i", "-", "-c:v", "libx264",
                           "-pix_fmt", "yuv420p", "-crf", "23", "-movflags", "+faststart", args.out],
                          stdin=subprocess.PIPE)
    while not model.is_terminal():
        if gs.game_tick % args.every == 0:
            frame = draw(gs, base, names, params.max_ticks)
            ff.stdin.write(frame.tobytes())
            if args.still and gs.game_tick == args.still_tick:
                frame.save(args.still)
        model.step({Player.Player1: me.get_action(gs), Player.Player2: other.get_action(gs)})
    leader = model.get_leader()
    banner = {Player.Player1: f"{names[0]} wins", Player.Player2: f"{names[1]} wins"}.get(leader, "Draw")
    end = draw(gs, base, names, params.max_ticks, banner=f"{banner} at tick {gs.game_tick}")
    for _ in range(args.fps * 3):
        ff.stdin.write(end.tobytes())
    ff.stdin.close()
    ff.wait()
    print(f"{params.num_planets} planets, speed {params.transporter_speed:.2f}: {banner} at tick {gs.game_tick}")


if __name__ == "__main__":
    main()

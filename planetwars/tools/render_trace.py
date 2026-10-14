"""Render a recorded game (a JSON-lines trace of map and launches) as an MP4.

The trace comes from a game played in the Kotlin runner, for example our entry against TeamTitansAgentV3,
which only runs on the JVM. Here the launches are replayed in the JAX copy of the rules (checked against the
official engine tick for tick) and each tick is drawn the way tools/record_game.py draws a game:

    python tools/render_trace.py trace.jsonl --out game.mp4 --ours "Our planner" --theirs "TeamTitansAgentV3"

The default style (tools/aura.py) is Auralux-like, with a synthesized soundtrack; `--style classic` draws plain
shapes without sound. Needs numpy, scipy, Pillow, opencv-python-headless and an ffmpeg binary.
"""
import argparse
import json
import pathlib
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "entry"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

import record_game as rg  # noqa: E402
from core.game_state import Player  # noqa: E402
from pwjax import env  # noqa: E402

N_MAX = 32
KOTLIN_OWNER = {0: 1, 1: 2, 2: 0}  # Kotlin ordinal (Player1, Player2, Neutral) -> engine id (Neutral 0, P1 1, P2 2)
PLAYER = {0: Player.Neutral, 1: Player.Player1, 2: Player.Player2}


def load(path):
    rows = [json.loads(line) for line in open(path)]
    for r in rows:
        if r["type"] == "header":
            for p in r["planets"]:
                p["owner"] = KOTLIN_OWNER[p["owner"]]
    return rows


def initial_state(header, max_ticks):
    ps = header["planets"]
    n = len(ps)

    def pad(vals, dtype):
        a = np.zeros(N_MAX, dtype)
        a[:n] = vals
        return a

    bank = {
        "valid": np.arange(N_MAX) < n, "owner": pad([p["owner"] for p in ps], np.int32),
        "ships": pad([p["ships"] for p in ps], np.float64),
        "pos": np.stack([pad([p["x"] for p in ps], np.float64), pad([p["y"] for p in ps], np.float64)], -1),
        "growth": pad([p["growth"] for p in ps], np.float64), "radius": pad([p["radius"] for p in ps], np.float64),
        "speed": np.array([header["speed"]], np.float64), "max_ticks": np.array([max_ticks], np.int32),
    }
    return env.from_bank({k: jnp.asarray(v)[None] for k, v in bank.items()}, 0), n


def as_game_state(s, n):
    """The JAX state as the small object record_game.draw reads (planets with owner, ships, position, fleet)."""
    g = {k: np.asarray(v) for k, v in s._asdict().items()}
    planets = []
    for i in range(n):
        fleet = None
        if g["tr_on"][i]:
            fleet = SimpleNamespace(owner=PLAYER[int(g["tr_owner"][i])], destination_index=int(g["tr_dest"][i]),
                                    s=SimpleNamespace(x=g["tr_pos"][i][0], y=g["tr_pos"][i][1]),
                                    v=SimpleNamespace(x=g["tr_vel"][i][0], y=g["tr_vel"][i][1]), n_ships=float(g["tr_ships"][i]))
        planets.append(SimpleNamespace(owner=PLAYER[int(g["owner"][i])], n_ships=float(g["ships"][i]),
                                       position=SimpleNamespace(x=g["pos"][i][0], y=g["pos"][i][1]),
                                       radius=float(g["radius"][i]), transporter=fleet))
    return SimpleNamespace(planets=planets, game_tick=int(g["tick"]))


def render_auralux(args, rows, s, n, names, launches, nothing, step):
    import aura
    final = rows[-1]
    states = [as_game_state(s, n)]
    for t in range(final["t"]):
        acts = launches.get(t, {})
        s = step(s, acts.get(1, nothing), acts.get(2, nothing))
        states.append(as_game_state(s, n))
    p1, p2 = float(env.planet_ships(s, 1)), float(env.planet_ships(s, 2))
    assert abs(p1 - final["p1"]) < 1e-3 * max(1.0, final["p1"]) and abs(p2 - final["p2"]) < 1e-3 * max(1.0, final["p2"]), \
        f"replay ended at {p1:.1f} / {p2:.1f}, the game at {final['p1']} / {final['p2']}"
    winner = 1 if p1 > p2 else 2 if p2 > p1 else 0
    banner = f"{names[winner - 1]} wins" if winner else "Draw"
    events = aura.find_events(states)
    by_tick = {}
    for ev in events:
        by_tick.setdefault(ev.t, []).append(ev)
    frames = (final["t"] + args.every - 1) // args.every
    hold = args.fps * 3
    wav = str(pathlib.Path(args.out).with_suffix(".wav"))
    aura.synthesize(events, frames + hold, args.fps, args.every, winner, wav)
    base = aura.background()
    ff = aura.encode(args.out, wav, args.fps)
    tick_rate = args.fps * args.every
    for f in range(frames):
        t = f * args.every
        fade_in = min(1.0, (f + 1) / 12.0)
        ff.stdin.write(aura.draw_frame(states[t], by_tick, t, names, base, tick_rate, fade=fade_in).tobytes())
        if args.still and t == args.still_tick:
            from PIL import Image
            Image.fromarray(aura.draw_frame(states[t], by_tick, t, names, base, tick_rate)).save(args.still)
    last = final["t"]
    for k in range(hold):
        ff.stdin.write(aura.draw_frame(states[last], by_tick, last + k // 2, names, base, tick_rate, banner=banner,
                                       banner_alpha=min(1.0, (k + 1) / 15.0), fade=min(1.0, (hold - k) / 20.0)).tobytes())
    ff.stdin.close()
    ff.wait()
    pathlib.Path(wav).unlink()
    print(f"{n} planets, speed {rows[0]['speed']:.2f}: {banner} at tick {final['t']} ({len(events)} events)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--out", default="game.mp4")
    ap.add_argument("--ours", default="Our planner")
    ap.add_argument("--theirs", default="TeamTitansAgentV3")
    ap.add_argument("--style", choices=["auralux", "classic"], default="auralux",
                    help="auralux: glowing orbs, swarming fleets, bloom and a synthesized soundtrack; classic: plain shapes, silent")
    ap.add_argument("--every", type=int, default=1, help="game ticks per video frame")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--max-ticks", type=int, default=2000)
    ap.add_argument("--still", default=None)
    ap.add_argument("--still-tick", type=int, default=250)
    args = ap.parse_args()
    rows = load(args.trace)
    header, final = rows[0], rows[-1]
    s, n = initial_state(header, args.max_ticks)
    ours = header["remote_side"]
    names = (args.ours, args.theirs) if ours == 1 else (args.theirs, args.ours)
    launches = {}
    for r in rows:
        if r["type"] == "launch":
            launches.setdefault(r["t"], {})[r["p"]] = (jnp.int32(r["src"]), jnp.int32(r["dst"]), jnp.float64(r["n"]))
    nothing = (jnp.int32(-1), jnp.int32(-1), jnp.float64(0.0))
    step = jax.jit(env.step)
    if args.style == "auralux":
        return render_auralux(args, rows, s, n, names, launches, nothing, step)
    base = rg.starfield()
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{rg.W}x{rg.H + rg.TOP}", "-r", str(args.fps), "-i", "-", "-c:v", "libx264",
                           "-pix_fmt", "yuv420p", "-crf", "23", "-movflags", "+faststart", args.out], stdin=subprocess.PIPE)
    for t in range(final["t"]):
        if t % args.every == 0:
            frame = rg.draw(as_game_state(s, n), base, names, args.max_ticks)
            ff.stdin.write(frame.tobytes())
            if args.still and t == args.still_tick:
                frame.save(args.still)
        acts = launches.get(t, {})
        s = step(s, acts.get(1, nothing), acts.get(2, nothing))
    p1, p2 = float(env.planet_ships(s, 1)), float(env.planet_ships(s, 2))
    assert abs(p1 - final["p1"]) < 1e-3 * max(1.0, final["p1"]) and abs(p2 - final["p2"]) < 1e-3 * max(1.0, final["p2"]), \
        f"replay ended at {p1:.1f} / {p2:.1f}, the game at {final['p1']} / {final['p2']}"
    winner = names[0] if p1 > p2 else names[1] if p2 > p1 else None
    banner = f"{winner} wins at tick {final['t']}" if winner else f"Draw at tick {final['t']}"
    end = rg.draw(as_game_state(s, n), base, names, args.max_ticks, banner=banner)
    for _ in range(args.fps * 3):
        ff.stdin.write(end.tobytes())
    ff.stdin.close()
    ff.wait()
    print(f"{n} planets, speed {header['speed']:.2f}: {banner}")


if __name__ == "__main__":
    main()

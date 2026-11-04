"""An Auralux-style look and sound for a recorded Planet Wars game.

Planets are glowing orbs ringed in their owner's colour, with one small dot orbiting for each ship. A fleet is a
swarm of dots with a comet tail. Launches, arrivals and captures send out ripples, and the whole frame gets a soft
bloom. The sound is the browser game's (play/index.html), rendered by tools/aura_sound.py: pentatonic chimes for
launches, impacts and captures, a closing arpeggio for the winner, and the game's ambient music (a slow pad and a
wandering melody of plucks that gets busier with more fleets in flight). Used by tools/render_trace.py.
"""
import math
import subprocess
from collections import namedtuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import aura_sound
from core.game_state import Player

S = 2                                   # pixels per game unit
W, H, TOP = 1280, 960, 72
TOTAL_H = H + TOP
COL = {0: (150, 160, 182), 1: (70, 170, 255), 2: (255, 112, 72)}   # neutral, player 1, player 2 (RGB)
OID = {Player.Neutral: 0, Player.Player1: 1, Player.Player2: 2}
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

Event = namedtuple("Event", "t kind x y r player n src hostile")   # tick, 'launch' | 'arrive' | 'capture', position, planet radius, player, ships, planet, hits a planet that is not the fleet's


def font(size, bold=False):
    try:
        return ImageFont.truetype(BOLD if bold else FONT, size)
    except OSError:
        return ImageFont.load_default()


def colour(owner, k=1.0):
    return tuple(int(min(255, c * k)) for c in COL[owner])


def background():
    rng = np.random.default_rng(5)
    yy, xx = np.mgrid[0:TOTAL_H, 0:W].astype(np.float32)
    r2 = ((xx - W / 2) / (W / 2)) ** 2 + ((yy - TOTAL_H / 2) / (TOTAL_H / 2)) ** 2
    glow = np.clip(1.0 - r2 / 2.0, 0, 1)
    base = np.stack([5 + 9 * glow, 8 + 13 * glow, 17 + 25 * glow], -1) / 255.0
    for _ in range(360):
        x, y = rng.integers(0, W), rng.integers(TOP, TOTAL_H)
        base[y, x] += rng.uniform(0.10, 0.45) * np.array([0.7, 0.8, 1.0])
    return np.clip(base, 0, 1).astype(np.float32)


# ------------------------------------------------------------------ events

def find_events(states):
    """Launches, arrivals and captures, read off consecutive states (the list of `as_game_state` objects)."""
    events = []
    for t in range(len(states) - 1):
        a, b = states[t], states[t + 1]
        for i, (pa, pb) in enumerate(zip(a.planets, b.planets)):
            x, y, r = pa.position.x * S, pa.position.y * S + TOP, pa.radius * S
            if pa.transporter is None and pb.transporter is not None:
                events.append(Event(t, "launch", x, y, r, OID[pb.transporter.owner], pb.transporter.n_ships, i, False))
            if pa.transporter is not None and pb.transporter is None:
                d = a.planets[pa.transporter.destination_index]
                events.append(Event(t, "arrive", d.position.x * S, d.position.y * S + TOP, d.radius * S,
                                    OID[pa.transporter.owner], pa.transporter.n_ships, pa.transporter.destination_index,
                                    d.owner != pa.transporter.owner))
            if pa.owner != pb.owner:
                events.append(Event(t, "capture", x, y, r, OID[pb.owner], pb.n_ships, i, False))
    return events


# ------------------------------------------------------------------ frames

def _circle(img, x, y, r, color, thickness):
    cv2.circle(img, (int(round(x * 16)), int(round(y * 16))), int(round(max(r, 0.2) * 16)), color, thickness, cv2.LINE_AA, 4)


def _ease(a):
    return 1.0 - (1.0 - a) ** 2


EVENT_TICKS = {"launch": 14, "arrive": 10, "capture": 30}


def draw_frame(gs, events_by_tick, t, names, base, fps_ticks=30.0, banner=None, banner_alpha=1.0, fade=1.0):
    """One frame as an RGB uint8 array. `t` is the tick, `events_by_tick` maps tick -> events that began then."""
    sec = t / fps_ticks
    crisp = np.zeros((TOTAL_H, W, 3), np.uint8)
    labels = []
    active = [ev for tt in range(max(0, t - 30), t + 1) for ev in events_by_tick.get(tt, ())]

    # planets
    for i, p in enumerate(gs.planets):
        o = OID[p.owner]
        cx, cy, r = p.position.x * S, p.position.y * S + TOP, p.radius * S
        k = 0.55 if o == 0 else 1.0
        flash = max([1.0 - (t - ev.t) / 10.0 for ev in active if ev.kind == "capture" and ev.src == i and t - ev.t < 10] + [0.0])
        _circle(crisp, cx, cy, r, colour(o, 0.10 * k + 0.35 * flash), -1)
        _circle(crisp, cx, cy, r, colour(o, 0.95 * k), 2)
        _circle(crisp, cx, cy, r * 0.88, colour(o, 0.28 * k), 1)
        n = min(int(p.n_ships), 40)
        rings = [(0.38, 8, 1.3), (0.62, 14, -0.9), (0.84, 18, 0.6)] if r >= 18 else [(0.55, 8, 1.4)]
        left = n
        dot = tuple(int(c * 0.72 + 255 * 0.28) for c in COL[o]) if o else colour(o, 0.9)
        for ring, (frac, cap, omega) in enumerate(rings):
            m = left if ring == len(rings) - 1 else min(left, cap)
            left -= m
            for j in range(m):
                ang = i * 0.9 + ring * 1.3 + omega * sec + 2 * math.pi * j / max(m, 1)
                _circle(crisp, cx + math.cos(ang) * r * frac, cy + math.sin(ang) * r * frac, 2.5 if r >= 18 else 1.8, dot, -1)
        labels.append((cx, cy, str(int(p.n_ships)), int(min(max(r * 0.55, 13), 24)), (236, 241, 250)))

    # fleets
    for i, p in enumerate(gs.planets):
        f = p.transporter
        if f is None:
            continue
        o = OID[f.owner]
        fx, fy = f.s.x * S, f.s.y * S + TOP
        speed = math.hypot(f.v.x, f.v.y) or 1.0
        ux, uy = f.v.x / speed, f.v.y / speed
        dest = gs.planets[f.destination_index]
        _guide = colour(o, 0.10)
        cv2.line(crisp, (int(fx), int(fy)), (int(dest.position.x * S), int(dest.position.y * S + TOP)), _guide, 1, cv2.LINE_AA)
        length = 36 + 3.0 * math.sqrt(f.n_ships)
        for seg in range(6):          # comet tail
            a0, a1 = seg / 6.0, (seg + 1) / 6.0
            cv2.line(crisp, (int(fx - ux * length * a0), int(fy - uy * length * a0)),
                     (int(fx - ux * length * a1), int(fy - uy * length * a1)), colour(o, 0.55 * (1 - a0) ** 1.5), 2, cv2.LINE_AA)
        rng = np.random.default_rng(i * 7919 + 13)
        k = min(int(f.n_ships), 28)
        spread = 3 + 0.7 * math.sqrt(f.n_ships)
        for j in range(k):
            along = -abs(rng.normal(0, length * 0.28))
            lateral = rng.normal(0, spread) + 2.2 * math.sin(0.3 * t + j * 1.7)
            x, y = fx + ux * along - uy * lateral, fy + uy * along + ux * lateral
            _circle(crisp, x, y, 3.4 if j == 0 else 2.6, tuple(int(c * 0.72 + 255 * 0.28) for c in COL[o]), -1)
        labels.append((fx - uy * 16 - ux * 6, fy + ux * 16 - uy * 6, str(int(f.n_ships)), 14, colour(o, 0.95)))

    # ripples and sparks
    for ev in active:
        age = (t - ev.t) / EVENT_TICKS[ev.kind]
        if not 0 <= age < 1:
            continue
        e = _ease(age)
        if ev.kind == "launch":
            _circle(crisp, ev.x, ev.y, ev.r + 2 + 34 * e, colour(ev.player, 0.8 * (1 - age)), 2)
        elif ev.kind == "arrive":
            _circle(crisp, ev.x, ev.y, ev.r + 22 * e, colour(ev.player, 0.9 * (1 - age)), 2)
            rng = np.random.default_rng(ev.t * 131 + ev.src)
            for _ in range(5 + min(int(ev.n) // 8, 8)):
                ang = rng.uniform(0, 2 * math.pi)
                r0, r1 = ev.r + 2 + 10 * e, ev.r + 8 + 26 * e
                cv2.line(crisp, (int(ev.x + math.cos(ang) * r0), int(ev.y + math.sin(ang) * r0)),
                         (int(ev.x + math.cos(ang) * r1), int(ev.y + math.sin(ang) * r1)),
                         tuple(int(c * (1 - age)) for c in (255, 245, 235)), 2, cv2.LINE_AA)
        else:
            _circle(crisp, ev.x, ev.y, ev.r + 80 * e, colour(ev.player, 0.9 * (1 - age)), 3)
            _circle(crisp, ev.x, ev.y, ev.r + 48 * _ease(max(age - 0.15, 0) / 0.85), colour(ev.player, 0.5 * (1 - age)), 2)

    # ship share bar
    s1 = sum(p.n_ships for p in gs.planets if OID[p.owner] == 1) + sum(p.transporter.n_ships for p in gs.planets if p.transporter and OID[p.transporter.owner] == 1)
    s2 = sum(p.n_ships for p in gs.planets if OID[p.owner] == 2) + sum(p.transporter.n_ships for p in gs.planets if p.transporter and OID[p.transporter.owner] == 2)
    share = s1 / (s1 + s2) if s1 + s2 > 0 else 0.5
    x0, x1, yb = 40, W - 40, 58
    cut = int(x0 + (x1 - x0) * share)
    cv2.line(crisp, (x0, yb), (cut, yb), colour(1, 0.95), 4, cv2.LINE_AA)
    cv2.line(crisp, (cut, yb), (x1, yb), colour(2, 0.95), 4, cv2.LINE_AA)

    # bloom
    f32 = crisp.astype(np.float32) / 255.0
    small = cv2.resize(f32, (W // 4, TOTAL_H // 4), interpolation=cv2.INTER_AREA)
    b1 = cv2.resize(cv2.GaussianBlur(small, (0, 0), 2.0), (W, TOTAL_H), interpolation=cv2.INTER_LINEAR)
    b2 = cv2.resize(cv2.GaussianBlur(small, (0, 0), 6.0), (W, TOTAL_H), interpolation=cv2.INTER_LINEAR)
    out = np.clip(base + f32 + 0.8 * b1 + 1.0 * b2, 0, 1)
    if banner:
        out = out * (1.0 - 0.45 * banner_alpha)
    img = Image.fromarray((out * 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    for x, y, text, size, color in labels:
        d.text((x, y), text, font=font(size, True), fill=color, anchor="mm")
    big, small_font = font(26, True), font(17)
    d.text((40, 12), names[0], font=big, fill=colour(1, 1.0))
    d.text((40 + d.textlength(names[0], font=big) + 14, 20), f"{int(s1)} ships", font=small_font, fill=(190, 200, 218))
    d.text((W - 40, 12), names[1], font=big, fill=colour(2, 1.0), anchor="ra")
    d.text((W - 40 - d.textlength(names[1], font=big) - 14, 20), f"{int(s2)} ships", font=small_font, fill=(190, 200, 218), anchor="ra")
    d.text((W / 2, 20), f"tick {gs.game_tick}", font=font(18), fill=(150, 160, 182), anchor="ma")
    if banner:
        layer = Image.new("L", (W, TOTAL_H), 0)
        ImageDraw.Draw(layer).text((W / 2, TOTAL_H / 2), banner, font=font(58, True), fill=255, anchor="mm")
        halo = np.asarray(layer.filter(ImageFilter.GaussianBlur(14)), np.float32) / 255.0
        arr = np.clip(np.asarray(img, np.float32) + halo[..., None] * np.array(banner_colour(banner, names), np.float32) * 1.2 * banner_alpha, 0, 255)
        img = Image.fromarray(arr.astype(np.uint8))
        ImageDraw.Draw(img).text((W / 2, TOTAL_H / 2), banner, font=font(58, True), fill=(int(255 * banner_alpha),) * 3, anchor="mm")
    frame = np.asarray(img, np.float32) * fade
    return frame.astype(np.uint8)


def banner_colour(banner, names):
    return COL[1] if banner.startswith(names[0]) else COL[2] if banner.startswith(names[1]) else (200, 200, 200)


# ------------------------------------------------------------------ sound

def synthesize(events, n_frames, fps, tick_per_frame, winner, path, end_tick=None):
    """Write the soundtrack for `events` (see find_events) to a WAV file at `path`: the browser game's sound effects
    and music (tools/aura_sound.py), timed to the video. The winner's arpeggio plays at `end_tick` (the game's last
    tick; by default the tick after the last event)."""
    ticks, fleets = aura_sound.from_events(events)
    seconds_per_tick = 1.0 / (tick_per_frame * fps)
    end_tick = max(len(fleets) - 1, 0) if end_tick is None else end_tick
    track = aura_sound.soundtrack(ticks, fleets, seconds_per_tick, n_frames / fps, winner, end_tick=end_tick)
    aura_sound.write_wav(path, track)


def encode(path, audio, fps):
    """An ffmpeg process that takes raw RGB frames on stdin and muxes them with the WAV `audio`."""
    return subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{TOTAL_H}",
                             "-r", str(fps), "-i", "-", "-i", audio, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
                             "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", path], stdin=subprocess.PIPE)

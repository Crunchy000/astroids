"""An Auralux-style look and sound for a recorded Planet Wars game.

Planets are glowing orbs ringed in their owner's colour, with one small dot orbiting for each ship. A fleet is a
swarm of dots with a comet tail. Launches, arrivals and captures send out ripples, and the whole frame gets a soft
bloom. The sound is synthesized from nothing (no samples): a slow ambient pad, a glassy pluck for each launch, a
low thump when a fleet hits an enemy planet, a rising arpeggio and a bell when a planet changes hands, and a
chord when the game ends. Used by tools/render_trace.py.
"""
import math
import subprocess
import wave
from collections import namedtuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy.signal import fftconvolve

from core.game_state import Player

S = 2                                   # pixels per game unit
W, H, TOP = 1280, 960, 72
TOTAL_H = H + TOP
COL = {0: (150, 160, 182), 1: (70, 170, 255), 2: (255, 112, 72)}   # neutral, player 1, player 2 (RGB)
OID = {Player.Neutral: 0, Player.Player1: 1, Player.Player2: 2}
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
SR = 44100

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

PENT = [0, 3, 5, 7, 10]


def hz(base, step):
    octave, k = divmod(step, 5)
    return base * 2 ** ((PENT[k] + 12 * octave) / 12)


def _put(buf, start, sig, pan):
    i = int(start * SR)
    n = min(len(sig), buf.shape[0] - i)
    if n <= 0 or i < 0:
        return
    buf[i:i + n, 0] += sig[:n] * math.cos(pan * math.pi / 2)
    buf[i:i + n, 1] += sig[:n] * math.sin(pan * math.pi / 2)


def _t(seconds):
    return np.arange(int(seconds * SR)) / SR


def pluck(f, amp):
    t = _t(1.6)
    env = (1 - np.exp(-t / 0.003)) * np.exp(-t / 0.30)
    sig = np.sin(2 * np.pi * f * t) + 0.35 * np.sin(2 * np.pi * 2 * f * t) * np.exp(-t / 0.12) + 0.12 * np.sin(2 * np.pi * 3 * f * t) * np.exp(-t / 0.06)
    return (sig * env * amp).astype(np.float32)


def thump(amp):
    t = _t(0.5)
    phase = 2 * np.pi * np.cumsum(52 + 90 * np.exp(-t / 0.05)) / SR
    return (np.sin(phase) * (1 - np.exp(-t / 0.002)) * np.exp(-t / 0.11) * amp).astype(np.float32)


def bell(f, amp):
    t = _t(2.4)
    sig = sum(a * np.sin(2 * np.pi * f * m * t) * np.exp(-t / d) for m, a, d in ((1, 1.0, 1.3), (2.76, 0.5, 0.8), (5.4, 0.25, 0.45), (8.93, 0.12, 0.25)))
    return (sig * (1 - np.exp(-t / 0.002)) * amp).astype(np.float32)


def ping(f, amp):
    t = _t(0.7)
    return (np.sin(2 * np.pi * f * t) * (1 - np.exp(-t / 0.002)) * np.exp(-t / 0.12) * amp).astype(np.float32)


def pad(duration):
    """A slow bed of chords (A minor, F, C, G), six seconds each, notes detuned a little for width."""
    chords = [[110.0, 130.8, 164.8, 220.0], [87.3, 110.0, 130.8, 174.6], [130.8, 164.8, 196.0, 261.6], [98.0, 123.5, 146.8, 196.0]]
    buf = np.zeros((int(duration * SR) + SR, 2), np.float32)
    seg = 6.0
    for c in range(int(duration / seg) + 1):
        chord = chords[c % len(chords)]
        t = _t(seg + 2.0)
        env = np.minimum(np.minimum(t / 2.0, (seg + 2.0 - t) / 2.0), 1.0).clip(0, 1) ** 2
        for k, f in enumerate(chord):
            for side, detune in ((0, 0.997), (1, 1.003)):
                sig = sum(a * np.sin(2 * np.pi * f * detune * h * t + 0.5 * k) for h, a in ((1, 1.0), (2, 0.35), (3, 0.14), (4, 0.06)))
                sig *= (0.85 + 0.15 * np.sin(2 * np.pi * (0.11 + 0.03 * k) * t)) * env * 0.022
                i = int(c * seg * SR)
                n = min(len(sig), buf.shape[0] - i)
                if n > 0:
                    buf[i:i + n, side] += sig[:n]
    return buf


def synthesize(events, n_frames, fps, tick_per_frame, winner, path):
    """Write the soundtrack for `events` (see find_events) to a WAV file at `path`."""
    rng = np.random.default_rng(11)
    dur = n_frames / fps
    mix = pad(dur + 1.0)
    when = lambda t: t / tick_per_frame / fps  # noqa: E731
    last_launch = {1: -99, 2: -99}
    last_arrive = {}
    base = {1: 440.0, 2: 220.0}
    pan = {1: 0.32, 2: 0.68}
    for ev in sorted(events, key=lambda e: e.t):
        s, p = when(ev.t), ev.player
        big = min(1.0, 0.35 + math.log10(ev.n + 1) / 2.2)
        if ev.kind == "launch" and ev.t - last_launch[p] >= 6 and ev.n >= 2:
            last_launch[p] = ev.t
            _put(mix, s, pluck(hz(base[p], (ev.src * 2 + int(ev.n)) % 10), 0.20 * big), pan[p])
        elif ev.kind == "arrive":
            if ev.t - last_arrive.get(ev.src, -99) < 6:
                continue
            last_arrive[ev.src] = ev.t
            if ev.hostile:
                _put(mix, s, thump(0.30 * big), pan[p])
            elif ev.n >= 3:
                _put(mix, s, ping(hz(base[p] * 2, ev.src % 5 + 2), 0.07 * big), pan[p])
        elif ev.kind == "capture":
            steps = [0, 2, 4] if p == 1 else [-3, -1, 1]
            for k, st in enumerate(steps):
                _put(mix, s + 0.085 * k, pluck(hz(base[p] * 2, (ev.src % 3) + st + 3), 0.22), pan[p])
            _put(mix, s + 0.2, bell(hz(base[p], (ev.src % 5) + 5), 0.10), pan[p])
    if winner in (1, 2):
        end = dur - 3.0
        chord = [220.0, 277.2, 329.6, 440.0] if winner == 1 else [196.0, 246.9, 293.7, 392.0]
        t = _t(3.4)
        env = np.minimum(t / 0.5, 1.0) * np.exp(-t / 1.6)
        for k, f in enumerate(chord):
            _put(mix, end, (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * 2 * f * t)).astype(np.float32) * env * 0.09, 0.3 + 0.13 * k)
    mix = mix[: int(dur * SR)]
    ir_t = _t(1.4)
    wet = np.stack([fftconvolve(mix[:, c], rng.standard_normal(len(ir_t)) * np.exp(-ir_t / 0.42) * 0.004, mode="full")[: len(mix)] for c in (0, 1)], 1)
    mix = mix + 0.55 * wet
    fade = np.ones(len(mix), np.float32)
    k = int(1.0 * SR)
    fade[-k:] = np.linspace(1, 0, k)
    k0 = int(0.4 * SR)
    fade[:k0] = np.linspace(0, 1, k0)
    mix = np.tanh(mix / max(np.abs(mix).max(), 1e-6) * 1.6) / np.tanh(1.6) * 0.89 * fade[:, None]
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((mix * 32767).astype("<i2").tobytes())


def encode(path, audio, fps):
    """An ffmpeg process that takes raw RGB frames on stdin and muxes them with the WAV `audio`."""
    return subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{TOTAL_H}",
                             "-r", str(fps), "-i", "-", "-i", audio, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
                             "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", path], stdin=subprocess.PIPE)

"""Synthesised sound effects for recorded videos (NumPy only).

Mirrors the browser game's Web Audio sounds: a laser "pew" per shot (blue
higher than red), a bright crack for hits, a rumbling boom when a ship is
destroyed and a thud when an asteroid breaks, panned by screen position.

    sim.sfx = []                      # start logging sound events
    ... run the simulation ...
    track = render_soundtrack(sim.sfx, duration, width=sim.cfg.width)
    write_wav("battle.wav", track)
"""

from __future__ import annotations

import wave

import numpy as np

RATE = 44_100
# Minimum spacing per sound kind, so a full 5v5 firefight stays listenable.
MIN_GAP = {"shot": 0.035, "hit": 0.05, "boom": 0.06, "rock": 0.07}


def _env(n: int, attack: float, decay: float) -> np.ndarray:
    t = np.arange(n) / RATE
    a = np.clip(t / max(attack, 1e-4), 0, 1)
    return a * np.exp(-np.maximum(t - attack, 0) / (decay / 5))


def _sweep(f0: float, f1: float, n: int) -> np.ndarray:
    """Phase of an exponential frequency sweep f0 -> f1 over n samples."""
    f = f0 * (f1 / f0) ** (np.arange(n) / n)
    return 2 * np.pi * np.cumsum(f) / RATE


def _lowpass_sweep(x: np.ndarray, f0: float, f1: float) -> np.ndarray:
    """One-pole low-pass whose cutoff glides from f0 to f1."""
    n = len(x)
    fc = f0 * (f1 / f0) ** (np.arange(n) / n)
    alpha = 1 - np.exp(-2 * np.pi * fc / RATE)
    y = np.empty_like(x)
    acc = 0.0
    for i in range(n):  # templates are built once, so a plain loop is fine
        acc += alpha[i] * (x[i] - acc)
        y[i] = acc
    return y


def _templates(rng: np.random.Generator) -> dict:
    t = {}
    for team, f in ((0, 1350.0), (1, 950.0)):
        n = int(0.115 * RATE)
        t[("shot", team)] = 0.05 * np.sign(np.sin(_sweep(f, f * 0.3, n))) * _env(n, 0.004, 0.11)
    n = int(0.095 * RATE)
    crack = np.diff(rng.uniform(-1, 1, n + 1))  # high-passed noise
    t[("hit", None)] = 0.25 * crack * _env(n, 0.002, 0.09)
    n = int(0.95 * RATE)
    rumble = _lowpass_sweep(rng.uniform(-1, 1, n), 1800, 90) * 3.0
    sub = np.sin(_sweep(140, 35, n))
    t[("boom", None)] = (0.8 * rumble + 0.45 * sub) * _env(n, 0.005, 0.9)
    for size in (1, 2, 3):
        n = int((0.26 + 0.1 * size) * RATE)
        thud = _lowpass_sweep(rng.uniform(-1, 1, n), 500 / size + 250, 60) * 3.0
        t[("rock", size)] = (0.25 + 0.1 * size) * thud * _env(n, 0.004, 0.25 + 0.1 * size)
    return t


def render_soundtrack(events, duration: float, width: float = 1280.0, seed: int = 0) -> np.ndarray:
    """Mix logged ``(time, kind, x, detail)`` events into a stereo float32 track."""
    rng = np.random.default_rng(seed)
    tmpl = _templates(rng)
    out = np.zeros((int(duration * RATE) + RATE, 2), dtype=np.float32)
    last: dict[str, float] = {}
    for time, kind, x, detail in sorted(events, key=lambda e: e[0]):
        if time - last.get(kind, -1.0) < MIN_GAP[kind]:
            continue
        last[kind] = time
        key = (kind, detail if kind in ("shot", "rock") else None)
        sig = tmpl[key]
        if kind == "shot":  # slight pitch variety, like the browser version
            idx = np.arange(0, len(sig), rng.uniform(0.95, 1.05))
            sig = np.interp(idx, np.arange(len(sig)), sig)
        pan = float(np.clip(x / width * 1.6 - 0.8, -1, 1))
        gains = np.array([np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)])
        start = int(time * RATE)
        end = min(start + len(sig), len(out))
        out[start:end] += sig[:end - start, None] * gains
    out = out[:int(duration * RATE)]
    peak = np.abs(out).max()
    if peak > 0.8:  # limiter; headroom for AAC encoding, so busy moments don't clip
        out *= 0.8 / peak
    return out


def write_wav(path: str, track: np.ndarray) -> None:
    pcm = (np.clip(track, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm.tobytes())

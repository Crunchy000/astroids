"""The browser game's sound (play/index.html), rendered offline for match videos: the same notes, envelopes, filters
and echo, worked out sample by sample with numpy and scipy instead of Web Audio.

Sound effects, pentatonic (C4 to C6), as in the game:
  launch   a soft sine tick from the middle of the scale
  clash    a high sine chime when a fleet lands on a planet it does not own (higher when it is player 1's fleet)
  capture  a quick arpeggio when a planet changes hands: rising when player 1 (blue) takes it, falling for player 2
  pulse    a low triangle note now and then
  end      a long rising arpeggio when player 1 wins, a slower falling one when player 2 does
Music, an original ambient piece in the Auralux style: a slow pad of A minor, F, C and G (8 beats of 0.75 s each)
through a lowpass that opens and closes, and a melody of glassy plucks in A minor pentatonic through an echo, which
wanders a step or two at a time, lands on chord tones, and plays more notes the more fleets are in flight.

Videos usually run faster than the game (30 ticks a second against 10 at Normal speed), so the effects are paced as
the game is at Normal: at most one launch tick, one chime and one capture every 0.1 s, and the pulse every 2.5 s
(every 25 ticks in the game). The level is the game's mix made louder for a video, with a soft limiter on the peaks.
tests/test_sound.py checks the constants here against the game's page: change both together.

    ticks, fleets = from_events(aura.find_events(states))         # or events_between() tick by tick, as you play
    track = soundtrack(ticks, fleets, seconds_per_tick, duration, winner)   # float32 [samples, 2] at 44.1 kHz
    write_wav(path, track)
"""
import math
import wave
from collections import namedtuple

import numpy as np
from scipy.signal import lfilter

SR = 44100
SCALE = [261.63, 293.66, 329.63, 392.00, 440.00, 523.25, 587.33, 659.25, 783.99, 880.00, 1046.50]
BEAT, BEATS_PER_CHORD = 0.75, 8
CHORDS = [[110.0, 130.81, 164.81, 220.0], [87.31, 110.0, 130.81, 174.61], [130.81, 164.81, 196.0, 261.63], [98.0, 123.47, 146.83, 196.0]]
MELODY = [440.0, 523.25, 587.33, 659.25, 783.99, 880.0, 1046.5, 1174.66]
CHORD_TONES = [[0, 1, 3, 5, 6], [0, 1, 5, 6], [1, 3, 4, 6], [2, 4, 7]]
MASTER, MUSIC = 0.8, 0.9          # the game's master and music gains
GAP, PULSE_EVERY = 0.1, 2.5       # pacing of the effects in seconds (the game at Normal speed)
LOUDER, KNEE, CEILING = 3.0, 0.7, 0.97   # video level: the game's mix x3 (its peaks are about 0.2), limited above KNEE


# ------------------------------------------------------------------ building blocks (each mirrors a function in the page)

def _t(seconds):
    return np.arange(int(round(seconds * SR))) / SR


def _add(buf, start, sig, gains=None):
    """Mix `sig` (mono) into `buf` (mono, or stereo with per-channel `gains`) from `start` seconds."""
    i = int(round(start * SR))
    lo, hi = max(0, -i), min(len(sig), len(buf) - i)
    if hi <= lo:
        return
    if buf.ndim == 1:
        buf[i + lo:i + hi] += sig[lo:hi]
    else:
        buf[i + lo:i + hi] += sig[lo:hi, None] * np.asarray(gains)[None, :]


def _wave(kind, f, t):
    if kind == "triangle":
        return 2.0 / math.pi * np.arcsin(np.sin(2 * math.pi * f * t))
    return np.sin(2 * math.pi * f * t)


def _exp_ramp(v0, v1, t, length):
    """Web Audio's exponentialRampToValueAtTime from v0 to v1 over `length` seconds, then held at v1."""
    return v0 * (v1 / v0) ** np.clip(t / length, 0.0, 1.0)


def _pan(p):
    """A StereoPannerNode's gains for a mono source (equal power)."""
    x = (p + 1) / 2
    return [math.cos(x * math.pi / 2), math.sin(x * math.pi / 2)]


def _lowpass(f0, q_db):
    """Web Audio's lowpass biquad (its Q is in dB) as normalized (b, a)."""
    w0 = 2 * math.pi * f0 / SR
    alpha = math.sin(w0) / (2 * 10 ** (q_db / 20))
    c = math.cos(w0)
    b = np.array([(1 - c) / 2, 1 - c, (1 - c) / 2])
    a = np.array([1 + alpha, -2 * c, 1 - alpha])
    return b / a[0], a / a[0]


class Mix:
    """The game's audio graph as buffers: effects go straight to the master, music to `bus` (stereo plucks, the pad
    in mono before its filter, the plucks again into the echo)."""

    def __init__(self, seconds):
        self.n = int(round(seconds * SR))
        self.fx = np.zeros((self.n, 2))
        self.bus = np.zeros((self.n, 2))
        self.pad = np.zeros(self.n)
        self.echo = np.zeros((self.n, 2))

    def tone(self, at, f, kind="sine", duration=0.4, volume=0.1):
        """`tone`: instant attack, exponential decay to silence over `duration`, centred."""
        t = _t(duration)
        _add(self.fx, at, _wave(kind, f, t) * _exp_ramp(volume, 0.0001, t, duration), [1.0, 1.0])

    def arpeggio(self, at, steps, gap, volume, duration=0.25, kind="sine"):
        for i, k in enumerate(steps):
            self.tone(at + i * gap, SCALE[k], kind, duration, volume)

    def pluck(self, at, f, volume, pan):
        """`pluckNote`: a sine and a fading octave, 6 ms attack, gone by 1.6 s, panned, into the music and the echo."""
        t = _t(1.7)
        g = np.where(t < 0.006, _exp_ramp(0.0001, volume, t, 0.006), _exp_ramp(volume, 0.0001, t - 0.006, 1.594))
        sig = (np.sin(2 * math.pi * f * t) + _exp_ramp(0.3, 0.001, t, 0.3) * np.sin(2 * math.pi * 2 * f * t)) * g
        _add(self.bus, at, sig, _pan(pan))
        _add(self.echo, at, sig, _pan(pan))

    def pad_chord(self, at, notes, duration):
        """`padChord`: each note a triangle and a sine 7 cents apart, a 2 s swell, held, then 2.5 s to silence."""
        t = _t(duration + 2.6)
        for f in notes:
            for kind, cents, vol in (("triangle", -7, 0.016), ("sine", 7, 0.02)):
                env = np.interp(t, [0.0, 2.0, duration, duration + 2.5], [0.0, vol, vol, 0.0])
                _add(self.pad, at, _wave(kind, f * 2 ** (cents / 1200), t) * env)


def _swept_lowpass(x, block=128):
    """The pad's filter: lowpass, Q 0.4, cutoff 900 Hz +- 350 Hz on a 0.05 Hz sine (its LFO starts with the music)."""
    y, zi = np.zeros_like(x), np.zeros(2)
    for i in range(0, len(x), block):
        b, a = _lowpass(900 + 350 * math.sin(2 * math.pi * 0.05 * i / SR), 0.4)
        y[i:i + block], zi = lfilter(b, a, x[i:i + block], zi=zi)
    return y


def _echo(x):
    """The melody's echo: a 0.5625 s delay fed back at 0.38 through a 2400 Hz lowpass; returns the wet signal (x 0.5)."""
    d = int(round(BEAT * 0.75 * SR))
    b, a = _lowpass(2400, 1.0)               # a BiquadFilterNode's default Q
    d_in, wet, zi = np.zeros_like(x), np.zeros_like(x), np.zeros(2)
    for i in range(0, len(x), d):            # blocks no longer than the delay, so each block's input is already known
        k = min(d, len(x) - i)
        filtered, zi = lfilter(b, a, d_in[i - d:i - d + k] if i >= d else np.zeros(k), zi=zi)
        d_in[i:i + k] = x[i:i + k] + 0.38 * filtered
        wet[i:i + k] = 0.5 * filtered
    return wet


# ------------------------------------------------------------------ a game's soundtrack

Event = namedtuple("Event", "t kind player hostile")   # the fields of tools/aura.py's events that the sound reads


def snapshot(planets, owner_id):
    """Per planet (owner, fleet), fleet (owner, destination) or None, from the engine's planets; `owner_id` maps the
    engine's owners to 0 (neutral), 1 and 2."""
    return [(owner_id[p.owner], (owner_id[p.transporter.owner], p.transporter.destination_index) if p.transporter else None)
            for p in planets]


def events_between(t, before, after):
    """The events of tick t, from snapshots before and after it, as tools/aura.py's `find_events` reads them."""
    out = []
    for (o0, f0), (o1, f1) in zip(before, after):
        if f0 is None and f1 is not None:
            out.append(Event(t, "launch", f1[0], False))
        if f0 is not None and f1 is None:
            out.append(Event(t, "arrive", f0[0], before[f0[1]][0] != f0[0]))
        if o0 != o1:
            out.append(Event(t, "capture", o1, False))
    return out


def from_events(events):
    """What the sound needs from tools/aura.py's `find_events`: per tick, whether a fleet launched, whose fleet hit a
    planet it does not own (1 when both did, as in the game) and who took a planet (likewise); and the number of
    fleets in flight at each tick."""
    ticks, change = {}, {}
    for ev in events:
        e = ticks.setdefault(ev.t, {"launch": False, "clash": 0, "capture": 0})
        if ev.kind == "launch":
            e["launch"] = True
            change[ev.t] = change.get(ev.t, 0) + 1
        elif ev.kind == "arrive":
            change[ev.t] = change.get(ev.t, 0) - 1
            if ev.hostile and ev.player in (1, 2):
                e["clash"] = 1 if e["clash"] == 1 or ev.player == 1 else 2
        elif ev.kind == "capture" and ev.player in (1, 2):
            e["capture"] = 1 if e["capture"] == 1 or ev.player == 1 else 2
    last = max(ticks) + 1 if ticks else 0
    fleets, n = [], 0
    for t in range(last + 1):
        fleets.append(n)                     # in flight at tick t
        n += change.get(t, 0)
    return ticks, fleets


def soundtrack(ticks, fleets, seconds_per_tick, duration, winner, end_tick=None, seed=11):
    """The game's effects and music for a video, as float32 [samples, 2].

    `ticks[t]` says what happened between tick t and t + 1 (see `from_events`); it sounds when tick t + 1 is shown,
    at (t + 1) * seconds_per_tick. `fleets[t]` is the number of fleets in flight at tick t. `winner` (1, 2 or 0, a
    draw) picks the closing arpeggio, played at `end_tick` (the last tick in `fleets` by default).
    """
    rng = np.random.default_rng(seed)
    mix = Mix(duration)
    last = {"launch": -1.0, "clash": -1.0, "capture": -1.0}
    ok = lambda kind, at: at - last[kind] >= GAP - 1e-9  # noqa: E731
    for t in sorted(ticks):
        at, ev = (t + 1) * seconds_per_tick, ticks[t]
        if ev["launch"] and ok("launch", at):
            last["launch"] = at
            mix.tone(at, SCALE[3 + rng.integers(3)], "sine", 0.15, 0.03)
        if ev["clash"] and ok("clash", at):
            last["clash"] = at
            mix.tone(at, SCALE[(6 if ev["clash"] == 1 else 4) + rng.integers(5)], "sine", 0.3, 0.12)
        if ev["capture"] and ok("capture", at):
            last["capture"] = at
            if ev["capture"] == 1:
                mix.arpeggio(at, [4, 6, 8, 10], 0.07, 0.08)
            else:
                mix.arpeggio(at, [7, 5, 3, 1], 0.07, 0.07)
    end_at = (end_tick if end_tick is not None else max(0, len(fleets) - 1)) * seconds_per_tick
    at = PULSE_EVERY
    while at < end_at:
        mix.tone(at, SCALE[rng.integers(3)], "triangle", 0.6, 0.12)
        at += PULSE_EVERY
    if winner == 1:
        mix.arpeggio(end_at, [0, 2, 4, 5, 7, 9, 10], 0.1, 0.09, 0.5)
    elif winner == 2:
        mix.arpeggio(end_at, [7, 5, 3, 2, 0], 0.14, 0.09, 0.6, "triangle")

    # the music: `scheduleBeat` for every beat from 0.1 s, as when a game starts
    mel, k, at = 3, 0, 0.1
    nearest = lambda choices, m: min(choices, key=lambda c: (abs(c - m), choices.index(c)))  # noqa: E731
    in_flight = lambda s: fleets[min(len(fleets) - 1, int(s / seconds_per_tick))] if fleets else 0  # noqa: E731
    while at < duration:
        c, beat = (k // BEATS_PER_CHORD) % len(CHORDS), k % BEATS_PER_CHORD
        busy = min(1.0, in_flight(at) / 12)
        if beat == 0:
            mix.pad_chord(at, CHORDS[c], BEAT * BEATS_PER_CHORD)
            mel = nearest(CHORD_TONES[c], mel)
            mix.pluck(at, MELODY[mel], 0.04, rng.uniform(-0.4, 0.4))
        elif rng.random() < 0.28 + 0.35 * busy:
            step = [-2, -1, -1, 1, 1, 2][rng.integers(6)]
            if not 0 <= mel + step < len(MELODY):
                step = -step
            mel += step
            if beat % 4 == 0:
                mel = nearest(CHORD_TONES[c], mel)
            mix.pluck(at, MELODY[mel], 0.03, rng.uniform(-0.4, 0.4))
            if rng.random() < 0.25 * busy:
                m2 = max(0, min(len(MELODY) - 1, mel + (1 if rng.random() < 0.5 else -1)))
                mix.pluck(at + BEAT / 2, MELODY[m2], 0.022, rng.uniform(-0.4, 0.4))
        k += 1
        at += BEAT
    music = mix.bus + _swept_lowpass(mix.pad)[:, None] + np.stack([_echo(mix.echo[:, 0]), _echo(mix.echo[:, 1])], 1)
    music *= MUSIC * (1 - np.exp(-np.arange(mix.n) / SR / 0.4))[:, None]      # the music's 0.4 s fade-in
    out = (mix.fx + music) * MASTER * LOUDER

    # a soft limiter above KNEE, then fades: 0.4 s in, 1 s out
    mag = np.abs(out)
    over = mag > KNEE
    out[over] = np.sign(out[over]) * (KNEE + (CEILING - KNEE) * np.tanh((mag[over] - KNEE) / (CEILING - KNEE)))
    fade = np.ones(mix.n)
    k_in, k_out = min(mix.n, int(0.4 * SR)), min(mix.n, int(1.0 * SR))
    fade[:k_in] = np.linspace(0, 1, k_in)
    fade[mix.n - k_out:] = np.minimum(fade[mix.n - k_out:], np.linspace(1, 0, k_out))
    return (out * fade[:, None]).astype(np.float32)


def write_wav(path, track):
    """16-bit stereo WAV at 44.1 kHz."""
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(track, -1, 1) * 32767).astype("<i2").tobytes())

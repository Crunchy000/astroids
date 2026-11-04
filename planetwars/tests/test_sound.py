"""The videos' soundtrack (tools/aura_sound.py) stays the browser game's sound (play/index.html), and renders sanely."""
import json
import pathlib
import re
import sys

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "tools"))
pytest.importorskip("scipy")
import aura_sound as A  # noqa: E402

PAGE = (HERE / "play" / "index.html").read_text()


def js_const(name):
    m = re.search(rf"const {name} = (\[.*?\]);", PAGE)
    assert m, f"{name} is not in the page"
    return json.loads(m.group(1))


def test_same_notes_as_the_game():
    assert js_const("SCALE") == A.SCALE
    assert js_const("CHORDS") == A.CHORDS
    assert js_const("MELODY") == A.MELODY
    assert js_const("CHORD_TONES") == A.CHORD_TONES
    assert f"const BEAT = {A.BEAT}, BEATS_PER_CHORD = {A.BEATS_PER_CHORD};" in PAGE
    assert f"master.gain.value = {A.MASTER};" in PAGE
    assert f"soundOn && musicOn ? {A.MUSIC} : 0" in PAGE


def test_same_effects_as_the_game():
    # the calls aura_sound.soundtrack makes, as the page writes them
    for call in ["tone(pickOf(3, 3), 'sine', 0.15, 0.03)",
                 "tone(pickOf(ours ? 6 : 4, 5), 'sine', 0.3, 0.12)",
                 "arpeggio([4, 6, 8, 10], 0.07, 0.08)",
                 "arpeggio([7, 5, 3, 1], 0.07, 0.07)",
                 "tone(pickOf(0, 3), 'triangle', 0.6, 0.12)",
                 "arpeggio([0, 2, 4, 5, 7, 9, 10], 0.1, 0.09, 0.5)",
                 "arpeggio([7, 5, 3, 2, 0], 0.14, 0.09, 0.6, 'triangle')",
                 "[['triangle', -7, 0.016], ['sine', 7, 0.02]]",
                 "pluckNote(MELODY[mel], t, 0.04)", "pluckNote(MELODY[mel], t, 0.03)", "pluckNote(MELODY[m2], t + BEAT / 2, 0.022)",
                 "Math.random() < 0.28 + 0.35 * busy", "Math.random() < 0.25 * busy", "Math.min(1, fleets / 12)",
                 "delay.delayTime.value = BEAT * 0.75; fb.gain.value = 0.38; tone_.type = 'lowpass'; tone_.frequency.value = 2400; wet.gain.value = 0.5;",
                 "padFilter.frequency.value = 900; padFilter.Q.value = 0.4;",
                 "lfo.frequency.value = 0.05; depth.gain.value = 350;",
                 "g.gain.exponentialRampToValueAtTime(vol, t + 0.006);", "g.gain.exponentialRampToValueAtTime(0.0001, t + 1.6);"]:
        assert call in PAGE, f"the page no longer has {call!r}: change tools/aura_sound.py to match"


def test_echo_matches_a_sample_by_sample_delay_line():
    rng = np.random.default_rng(0)
    x = np.zeros(int(2.5 * A.SR))
    x[:2000] = rng.standard_normal(2000)
    d = int(round(A.BEAT * 0.75 * A.SR))
    b, a = A._lowpass(2400, 1.0)
    line, y, x1, x2, y1, y2 = np.zeros(len(x)), np.zeros(len(x)), 0.0, 0.0, 0.0, 0.0
    for i in range(len(x)):
        u = line[i - d] if i >= d else 0.0
        f = b[0] * u + b[1] * x1 + b[2] * x2 - a[1] * y1 - a[2] * y2
        x2, x1, y2, y1 = x1, u, y1, f
        line[i] = x[i] + 0.38 * f
        y[i] = 0.5 * f
    assert np.abs(A._echo(x) - y).max() < 1e-12


def events():
    E = A.Event
    return [E(0, "launch", 1, False), E(0, "launch", 2, False), E(1, "launch", 1, False),  # 1 is within 0.1 s of 0: one tick
            E(20, "arrive", 1, True), E(20, "capture", 1, False),
            E(45, "arrive", 2, True), E(45, "arrive", 1, False), E(45, "capture", 2, False)]


def test_events_to_ticks_and_fleets():
    ticks, fleets = A.from_events(events())
    assert ticks[0] == {"launch": True, "clash": 0, "capture": 0}
    assert ticks[20] == {"launch": False, "clash": 1, "capture": 1}
    assert ticks[45] == {"launch": False, "clash": 2, "capture": 2}
    assert fleets[0] == 0 and fleets[1] == 2 and fleets[2] == 3 and fleets[21] == 2 and fleets[46] == 0


def test_events_between_snapshots():
    before = [(1, (1, 2)), (2, None), (0, None)]
    after = [(1, None), (2, (2, 0)), (1, None)]
    got = A.events_between(7, before, after)
    assert got == [A.Event(7, "arrive", 1, True), A.Event(7, "launch", 2, False), A.Event(7, "capture", 1, False)]


def test_soundtrack():
    ticks, fleets = A.from_events(events())
    spt = 1 / 30
    track = A.soundtrack(ticks, fleets, spt, 6.0, winner=1, end_tick=60)
    assert track.shape == (6 * A.SR, 2) and track.dtype == np.float32
    assert np.isfinite(track).all() and np.abs(track).max() <= A.CEILING
    assert np.sqrt((track ** 2).mean()) > 0.01                                   # not silent
    assert np.abs(track[: int(0.4 * A.SR) // 4]).max() < np.abs(track).max() / 4  # fades in
    assert np.abs(track[-200:]).max() < 0.01                                    # and out
    again = A.soundtrack(ticks, fleets, spt, 6.0, winner=1, end_tick=60)
    assert np.array_equal(track, again)                                         # the same every time

    # the clash chime of tick 20 sounds at tick 21: a jump in level there
    i = int(21 * spt * A.SR)
    before, after = np.abs(track[i - 2000:i - 50]).max(), np.abs(track[i:i + 2000]).max()
    assert after > 1.5 * before


def test_write_wav(tmp_path):
    import wave
    path = tmp_path / "a.wav"
    A.write_wav(path, np.zeros((A.SR, 2), np.float32))
    with wave.open(str(path)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()) == (2, 2, A.SR, A.SR)

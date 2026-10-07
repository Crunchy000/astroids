"""Numba-compiled observation encoder: same output as ``obs.encode``, ~10x faster.

Used through ``obs.encode_battle``, which falls back to the NumPy version when
numba isn't installed (``pip install numba`` or the ``[fast]`` extra).
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit


@njit(cache=True)
def _wrap(d, size):  # same arithmetic as Simulation.wdx / wdy: (b - a + size/2) % size - size/2
    return (d + size / 2) % size - size / 2


@njit(cache=True)
def _topk(d2, valid, n, k, idx):
    """Write the indices of the k smallest valid d2 into idx (stable for ties); return the count."""
    cnt = 0
    for j in range(n):
        if not valid[j]:
            continue
        v = d2[j]
        pos = cnt
        while pos > 0 and d2[idx[pos - 1]] > v:
            pos -= 1
        if pos >= k:
            continue
        end = cnt if cnt < k else k - 1
        for q in range(end, pos, -1):
            idx[q] = idx[q - 1]
        idx[pos] = j
        if cnt < k:
            cnt += 1
    return cnt


@njit(cache=True)
def _clip(v):
    return -1.0 if v < -1.0 else 1.0 if v > 1.0 else v


@njit(cache=True)
def encode_kernel(ids, X, Y, VX, VY, A, HP, ALIVE, TEAM, COOL, INV,
                  BX, BY, BVX, BVY, BT, AX, AY, AVX, AVY, AR,
                  w, h, max_speed, ship_hp, fire_cooldown, bullet_speed,
                  n_en, n_al, n_bl, n_as, out):
    n, nb, na = X.shape[0], BX.shape[0], AX.shape[0]
    big = max(n, nb, na, 1)
    dxs, dys, d2 = np.empty(big), np.empty(big), np.empty(big)
    valid = np.empty(big, dtype=np.bool_)
    idx = np.empty(max(n_en, n_al, n_bl, n_as, 1), dtype=np.int64)
    pos_scale = 1.0 / (w / 2)
    ship_v_scale = 1.0 / (2 * max_speed)
    bullet_v_scale = 1.0 / (bullet_speed + 2 * max_speed)
    for r in range(ids.shape[0]):
        i = ids[r]
        ca, sa = math.cos(A[i]), math.sin(A[i])
        alive = ALIVE[i]
        row = np.zeros(out.shape[1])
        row[0] = 1.0 if alive else 0.0
        row[1] = (VX[i] * ca + VY[i] * sa) / max_speed
        row[2] = (-VX[i] * sa + VY[i] * ca) / max_speed
        row[3] = sa
        row[4] = ca
        row[5] = HP[i] / ship_hp if alive else 0.0
        row[6] = max(COOL[i], 0.0) / fire_cooldown if alive else 0.0
        row[7] = 1.0 if (alive and INV[i] > 0) else 0.0
        col = 8

        for j in range(n):
            dxs[j] = _wrap(X[j] - X[i], w)
            dys[j] = _wrap(Y[j] - Y[i], h)
            d2[j] = dxs[j] * dxs[j] + dys[j] * dys[j]
        for p in range(2):
            enemies = p == 0
            k = n_en if enemies else n_al
            dim = 10 if enemies else 8
            for j in range(n):
                valid[j] = ALIVE[j] and j != i and ((TEAM[j] != TEAM[i]) == enemies)
            cnt = _topk(d2, valid, n, k, idx)
            for q in range(cnt):
                j = idx[q]
                dx, dy = dxs[j], dys[j]
                rvxw, rvyw = VX[j] - VX[i], VY[j] - VY[i]
                b = col + q * dim
                row[b] = 1.0
                row[b + 1] = (dx * ca + dy * sa) * pos_scale
                row[b + 2] = (-dx * sa + dy * ca) * pos_scale
                row[b + 3] = (rvxw * ca + rvyw * sa) * ship_v_scale
                row[b + 4] = (-rvxw * sa + rvyw * ca) * ship_v_scale
                rel = A[j] - A[i]
                row[b + 5] = math.sin(rel)
                row[b + 6] = math.cos(rel)
                row[b + 7] = HP[j] / ship_hp
                if enemies:  # lead direction, exactly as bot.lead()
                    qa = rvxw * rvxw + rvyw * rvyw - bullet_speed ** 2
                    qb = 2 * (dx * rvxw + dy * rvyw)
                    qc = dx * dx + dy * dy
                    t = 0.0
                    disc = qb * qb - 4 * qa * qc
                    if disc >= 0 and qa != 0:
                        rt = math.sqrt(disc)
                        t1, t2 = (-qb - rt) / (2 * qa), (-qb + rt) / (2 * qa)
                        if t1 > 0 and t2 > 0:
                            t = min(t1, t2)
                        elif t1 > 0:
                            t = t1
                        elif t2 > 0:
                            t = t2
                    aim = math.atan2(dy + rvyw * t, dx + rvxw * t) - A[i]
                    row[b + 8] = math.cos(aim)
                    row[b + 9] = math.sin(aim)
            col += k * dim

        for j in range(nb):
            dxs[j] = _wrap(BX[j] - X[i], w)
            dys[j] = _wrap(BY[j] - Y[i], h)
            d2[j] = dxs[j] * dxs[j] + dys[j] * dys[j]
            valid[j] = BT[j] != TEAM[i]
        cnt = _topk(d2, valid, nb, n_bl, idx)
        for q in range(cnt):
            j = idx[q]
            dx, dy = dxs[j], dys[j]
            rvxw, rvyw = BVX[j] - VX[i], BVY[j] - VY[i]
            b = col + q * 5
            row[b] = 1.0
            row[b + 1] = (dx * ca + dy * sa) * pos_scale
            row[b + 2] = (-dx * sa + dy * ca) * pos_scale
            row[b + 3] = (rvxw * ca + rvyw * sa) * bullet_v_scale
            row[b + 4] = (-rvxw * sa + rvyw * ca) * bullet_v_scale
        col += n_bl * 5

        for j in range(na):
            dxs[j] = _wrap(AX[j] - X[i], w)
            dys[j] = _wrap(AY[j] - Y[i], h)
            d2[j] = dxs[j] * dxs[j] + dys[j] * dys[j]
            valid[j] = True
        cnt = _topk(d2, valid, na, n_as, idx)
        for q in range(cnt):
            j = idx[q]
            dx, dy = dxs[j], dys[j]
            rvxw, rvyw = AVX[j] - VX[i], AVY[j] - VY[i]
            b = col + q * 6
            row[b] = 1.0
            row[b + 1] = (dx * ca + dy * sa) * pos_scale
            row[b + 2] = (-dx * sa + dy * ca) * pos_scale
            row[b + 3] = (rvxw * ca + rvyw * sa) * ship_v_scale
            row[b + 4] = (-rvxw * sa + rvyw * ca) * ship_v_scale
            row[b + 5] = AR[j] / 48.0

        for c in range(out.shape[1]):
            out[r, c] = _clip(row[c])


def encode_fast(sim, ids, oc) -> np.ndarray:
    """Observations for ships ``ids`` of one battle, shape (len(ids), oc.size)."""
    cfg = sim.cfg
    ships, bl, ast = sim.ships, sim.bullets, sim.asteroids
    out = np.empty((len(ids), oc.size), dtype=np.float32)
    encode_kernel(
        np.asarray(ids, dtype=np.int64),
        np.array([s.x for s in ships]), np.array([s.y for s in ships]),
        np.array([s.vx for s in ships]), np.array([s.vy for s in ships]),
        np.array([s.a for s in ships]), np.array([float(s.hp) for s in ships]),
        np.array([s.alive for s in ships], dtype=np.bool_), np.array([s.team for s in ships], dtype=np.int64),
        np.array([s.cool for s in ships]), np.array([s.invuln for s in ships]),
        np.array([b.x for b in bl], dtype=np.float64), np.array([b.y for b in bl], dtype=np.float64),
        np.array([b.vx for b in bl], dtype=np.float64), np.array([b.vy for b in bl], dtype=np.float64),
        np.array([b.team for b in bl], dtype=np.int64),
        np.array([a.x for a in ast], dtype=np.float64), np.array([a.y for a in ast], dtype=np.float64),
        np.array([a.vx for a in ast], dtype=np.float64), np.array([a.vy for a in ast], dtype=np.float64),
        np.array([a.r for a in ast], dtype=np.float64),
        float(cfg.width), float(cfg.height), float(cfg.max_speed), float(cfg.ship_hp),
        float(cfg.fire_cooldown), float(cfg.bullet_speed),
        oc.n_enemies, oc.n_allies, oc.n_bullets, oc.n_asteroids, out)
    return out

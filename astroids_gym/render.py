"""pygame renderer mirroring the look of the browser canvas. Optional dependency."""

from __future__ import annotations

import math
import os

import numpy as np

from .sim import TAU, Simulation

TEAM_COLORS = [(76, 201, 240), (255, 123, 84)]
BG = (5, 6, 10)
ROCK = (138, 148, 166)
FLAME = (255, 209, 102)


class Renderer:
    def __init__(self, sim: Simulation, mode: str, fps: int):
        try:
            if mode == "rgb_array":
                os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
            import pygame
        except ImportError as e:  # pragma: no cover
            raise ImportError("Rendering needs pygame: pip install 'astroids-gym[render]'") from e
        self.pg = pygame
        self.sim = sim
        self.mode = mode
        self.fps = fps
        self.caption = ""  # optional text drawn top-right (e.g. who is flying)
        size = (int(sim.cfg.width), int(sim.cfg.height))
        pygame.init()
        if mode == "human":
            pygame.display.set_caption("Asteroids 5v5")
            self.screen = pygame.display.set_mode(size)
        else:
            self.screen = pygame.Surface(size)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 18, bold=True)
        self.small = pygame.font.SysFont("monospace", 15)
        self.big = pygame.font.SysFont("monospace", 34, bold=True)
        rng = np.random.default_rng(0)
        self.stars = [(rng.random() * size[0], rng.random() * size[1], rng.random()) for _ in range(160)]

    def _ghosts(self, x: float, y: float, r: float):
        w, h = self.sim.cfg.width, self.sim.cfg.height
        xs, ys = [x], [y]
        if x < r: xs.append(x + w)
        elif x > w - r: xs.append(x - w)
        if y < r: ys.append(y + h)
        elif y > h - r: ys.append(y - h)
        return [(px, py) for px in xs for py in ys]

    def draw(self):
        pg, sim, scr = self.pg, self.sim, self.screen
        if self.mode == "human":
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.close()
                    raise SystemExit
        scr.fill(BG)
        for x, y, b in self.stars:
            c = int(40 + b * 130)
            scr.set_at((int(x), int(y)), (c, c, c))

        cfg = sim.cfg
        if cfg.zone:
            self._draw_zone()

        for a in sim.asteroids:
            for gx, gy in self._ghosts(a.x, a.y, a.r * 1.2):
                n = len(a.shape)
                pts = [(gx + math.cos(a.a + k / n * TAU) * a.r * a.shape[k],
                        gy + math.sin(a.a + k / n * TAU) * a.r * a.shape[k]) for k in range(n)]
                pg.draw.polygon(scr, ROCK, pts, 1)

        for b in sim.bullets:
            pg.draw.circle(scr, TEAM_COLORS[b.team], (int(b.x), int(b.y)), 2)

        r = sim.cfg.ship_r
        for s in sim.ships:
            if not s.alive:
                continue
            color = TEAM_COLORS[s.team]
            if s.invuln > 0 and int(s.invuln * 10) % 2 == 0:
                color = tuple(c // 3 for c in color)
            ca, sa = math.cos(s.a), math.sin(s.a)
            local = [(r * 1.3, 0), (-r, r * 0.8), (-r * 0.55, 0), (-r, -r * 0.8)]
            for gx, gy in self._ghosts(s.x, s.y, r * 2):
                pts = [(gx + px * ca - py * sa, gy + px * sa + py * ca) for px, py in local]
                pg.draw.polygon(scr, color, pts, 2)
                if s.thrusting:
                    flame = [(-r * 0.7, r * 0.35), (-r * 1.6, 0), (-r * 0.7, -r * 0.35)]
                    pg.draw.lines(scr, FLAME, False,
                                  [(gx + px * ca - py * sa, gy + px * sa + py * ca) for px, py in flame], 2)
                for k in range(s.hp):
                    pg.draw.rect(scr, color, (gx - 7 + k * 5, gy + r + 6, 3, 3))
                if cfg.heat and s.heat > 0.02:  # weapon heat; red while the gun is locked
                    hc = (255, 77, 77) if s.overheated else FLAME
                    pg.draw.rect(scr, hc, (gx - 7, gy + r + 11, max(1, int(14 * min(s.heat, 1))), 2))

        self._draw_hud()
        if self.caption:
            text = self.small.render(self.caption, True, (150, 150, 160))
            scr.blit(text, ((scr.get_width() - text.get_width()) // 2, scr.get_height() - 26))
        if cfg.team_lives and sim.over:
            self._draw_banner()

        if self.mode == "human":
            pg.display.flip()
            self.clock.tick(self.fps)
            return None
        return np.transpose(pg.surfarray.array3d(scr), (1, 0, 2)).copy()

    def _text(self, txt, font, color, x, y, anchor="left"):
        surf = font.render(txt, True, color)
        if anchor == "right":
            x -= surf.get_width()
        elif anchor == "center":
            x -= surf.get_width() // 2
        self.screen.blit(surf, (x, y))

    def _draw_hud(self):
        sim, w = self.sim, self.screen.get_width()
        cfg = sim.cfg
        alive = [sum(1 for s in sim.ships if s.alive and s.team == t) for t in (0, 1)]
        if cfg.team_lives:
            side = [f"{int(sim.team_score(t))} · respawns {sim.lives[t]} · alive {alive[t]}" for t in (0, 1)]
            left = max(0.0, cfg.battle_seconds - sim.time)
            self._text(f"{int(left // 60)}:{int(left % 60):02d}", self.font, (220, 220, 220), w // 2, 8, "center")
        else:
            side = [f"{sim.score[t]} · alive {alive[t]}" for t in (0, 1)]
            self._text(f"t={sim.time:5.1f}s", self.font, (220, 220, 220), w // 2, 8, "center")
        self._text(f"BLUE {side[0]}", self.font, TEAM_COLORS[0], 12, 8)
        self._text(f"{side[1]} RED", self.font, TEAM_COLORS[1], w - 12, 8, "right")

    def _draw_zone(self):
        pg, sim, scr = self.pg, self.sim, self.screen
        cfg = sim.cfg
        inside = sim.zone_counts()
        holder = -1 if inside[0] == inside[1] else (0 if inside[0] > inside[1] else 1)
        rgb = TEAM_COLORS[holder] if holder >= 0 else (150, 155, 170)
        cx, cy, rad = int(cfg.width / 2), int(cfg.height / 2), int(cfg.zone_radius)
        tint = pg.Surface((2 * rad, 2 * rad), pg.SRCALPHA)
        pg.draw.circle(tint, (*rgb, 23 if holder >= 0 else 10), (rad, rad), rad)
        scr.blit(tint, (cx - rad, cy - rad))
        for k in range(0, 60, 2):  # dashed outline
            a0, a1 = k / 60 * TAU, (k + 1) / 60 * TAU
            pg.draw.arc(scr, tuple(int(c * (0.75 if holder >= 0 else 0.45)) for c in rgb),
                        (cx - rad, cy - rad, 2 * rad, 2 * rad), a0, a1, 2)

    def _draw_banner(self):
        pg, sim, scr = self.pg, self.sim, self.screen
        w, h = scr.get_width(), scr.get_height()
        shade = pg.Surface((w, 140), pg.SRCALPHA)
        shade.fill((5, 6, 10, 150))
        scr.blit(shade, (0, h // 2 - 70))
        win = sim.winner()
        how = "by wipe-out" if sim.eliminated(0) != sim.eliminated(1) else "on points"
        title = "DRAW" if win < 0 else f"{('BLUE', 'RED')[win]} WINS {how}"
        self._text(title, self.big, TEAM_COLORS[win] if win >= 0 else (220, 220, 220), w // 2, h // 2 - 36, "center")
        line = "   ·   ".join(f"{n} {sim.score[t]} kills + {sim.points[t]:.1f} zone"
                              for t, n in ((0, "BLUE"), (1, "RED")))
        self._text(line, self.small, (190, 190, 190), w // 2, h // 2 + 14, "center")

    def close(self):
        if self.mode == "human":
            self.pg.display.quit()
        self.pg.quit()

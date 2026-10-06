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

    def draw(self, highlight: set[int] | None = None):
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
                if highlight and s.id in highlight:
                    pg.draw.circle(scr, (255, 255, 255), (int(gx), int(gy)), int(r * 2), 1)
                for k in range(s.hp):
                    pg.draw.rect(scr, color, (gx - 7 + k * 5, gy + r + 6, 3, 3))

        hud = f"BLUE {sim.score[0]}   RED {sim.score[1]}   t={sim.time:5.1f}s"
        scr.blit(self.font.render(hud, True, (220, 220, 220)), (12, 8))
        if self.caption:
            text = self.font.render(self.caption, True, (220, 220, 220))
            scr.blit(text, (scr.get_width() - text.get_width() - 12, 8))

        if self.mode == "human":
            pg.display.flip()
            self.clock.tick(self.fps)
            return None
        return np.transpose(pg.surfarray.array3d(scr), (1, 0, 2)).copy()

    def close(self):
        if self.mode == "human":
            self.pg.display.quit()
        self.pg.quit()

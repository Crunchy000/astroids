/* Planet Wars in JavaScript: the official rules, league-style maps, and three opponents.

   The engine is core/forward_model.py rule for rule. The opponents:
     greedy  - their GreedyHeuristicAgent sample (entry/agents/greedy_heuristic_agent.py)
     planner - our planner with its hand-written score only (entry/planner.py `plan`)
     entry   - our league entry: the planner, its best launches re-ranked by the value network, looking one
               opponent reply ahead, plus two-planet neutral captures (entry/planner.py `plan_ranked`)
   Function names follow entry/planner.py so the two can be read side by side; tools/parity.js checks this file
   against the Python entry on recorded positions.

   Runs in a browser (window or worker: everything hangs off self.PW) and in Node (module.exports). */
(function (root) {
  'use strict';

  const WIDTH = 640, HEIGHT = 480;

  // ---------------------------------------------------------------- random numbers and maps

  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /** GameParamGenerator.randomParams() (pwjax/maps.py cog2026_params): what the live league plays. */
  function leagueParams(rand, numPlanets) {
    const raw = 10 + Math.floor(rand() * 21);
    const p = {
      width: WIDTH, height: HEIGHT, edgeSeparation: 25, radialSeparation: 1.5, growthToRadius: 200,
      numPlanets: raw % 2 === 0 ? raw : raw + 1, neutralRatio: 0.25 + rand() * 0.1, maxTicks: 2000,
      minShips: 2, maxShips: 20, minGrowth: 0.05, maxGrowth: 0.2, speed: 2 + rand() * 3,
    };
    if (numPlanets) p.numPlanets = numPlanets;
    return p;
  }

  /** GameStateFactory.createGame() (pwjax/maps.py create_game): rejection-sample the left half, then mirror it. */
  function createGame(params, rand, maxAttempts = 1000000) {
    const half = [];
    const nNeutral = Math.floor(Math.floor(params.numPlanets * params.neutralRatio) / 2);
    const w2 = params.width / 2, e = params.edgeSeparation;
    for (let tries = 0; half.length < Math.floor(params.numPlanets / 2); tries++) {
      if (tries >= maxAttempts) return null;
      const x = Math.floor(rand() * params.width / 2), y = Math.floor(rand() * params.height);
      const ships = rand() * (params.maxShips - params.minShips) + params.minShips;
      const growth = params.minGrowth + rand() * (params.maxGrowth - params.minGrowth);
      const r = growth * params.growthToRadius;
      if (x - e < r || x + e > w2 - r || y - e < r || y + e > params.height - r) continue;
      let clash = false;
      for (const p of half) {
        if (Math.hypot(p.x - x, p.y - y) < params.radialSeparation * (p.growth * params.growthToRadius + r)) { clash = true; break; }
      }
      if (clash) continue;
      half.push({ x, y, growth, ships, owner: half.length < nNeutral ? 0 : 1 });
    }
    const all = half.concat(half.map(p => ({
      x: params.width - p.x, y: params.height - p.y, growth: p.growth, ships: p.ships, owner: p.owner === 1 ? 2 : p.owner,
    })));
    return {
      tick: 0, speed: params.speed, maxTicks: params.maxTicks,
      planets: all.map((p, id) => ({
        id, owner: p.owner, ships: p.ships, x: p.x, y: p.y, growth: p.growth,
        radius: p.growth * params.growthToRadius, tr: null,
      })),
    };
  }

  /** A league-style game from a seed; `numPlanets` (even, 10 to 30) fixes the size. */
  function newGame(seed, numPlanets) {
    const rand = mulberry32(seed);
    for (;;) {
      const g = createGame(leagueParams(rand, numPlanets), rand);
      if (g) return g;
    }
  }

  // ---------------------------------------------------------------- the rules (core/forward_model.py)

  function applyAction(st, player, a) {
    if (!a) return false;
    const P = st.planets;
    if (!(a.src >= 0 && a.src < P.length) || !(a.dst >= 0 && a.dst < P.length) || !Number.isFinite(a.ships) || a.ships <= 0) return false;
    const s = P[a.src], t = P[a.dst];
    if (s.tr === null && s.owner === player && s.ships >= a.ships) {
      s.ships -= a.ships;
      let dx = t.x - s.x, dy = t.y - s.y;
      const m = Math.sqrt(dx * dx + dy * dy);
      if (m > 0) { const inv = 1.0 / m; dx *= inv; dy *= inv; }
      s.tr = { x: s.x, y: s.y, vx: dx * st.speed, vy: dy * st.speed, owner: player, src: a.src, dst: a.dst, ships: a.ships };
      return true;
    }
    return false;
  }

  /** One tick: actions[1], actions[2] are {src, dst, ships} or null. Returns which actions took effect. */
  function step(st, actions) {
    const ok = { 1: applyAction(st, 1, actions[1]), 2: applyAction(st, 2, actions[2]) };
    const P = st.planets, n = P.length;
    const pend1 = new Float64Array(n), pend2 = new Float64Array(n), pending = new Uint8Array(n);
    for (const p of P) {
      const tr = p.tr;
      if (!tr) continue;
      const d = P[tr.dst];
      const dx = tr.x - d.x, dy = tr.y - d.y;
      if (Math.sqrt(dx * dx + dy * dy) < d.radius) {
        pending[tr.dst] = 1;
        if (tr.owner === 1) pend1[tr.dst] += tr.ships; else pend2[tr.dst] += tr.ships;
        p.tr = null;
      } else {
        tr.x += tr.vx; tr.y += tr.vy;
      }
    }
    for (let i = 0; i < n; i++) {
      const p = P[i];
      if (p.owner === 0) {
        if (!pending[i]) continue;
        const net = pend1[i] - pend2[i];
        p.ships -= Math.abs(net);
        if (p.ships < 0) { p.owner = net > 0 ? 1 : 2; p.ships = -p.ships; }
      } else {
        p.ships += p.growth;
        if (!pending[i]) continue;
        const own = p.owner === 1 ? pend1[i] : pend2[i], opp = p.owner === 1 ? pend2[i] : pend1[i];
        p.ships += own - opp;
        if (p.ships < 0) { p.owner = 3 - p.owner; p.ships = -p.ships; }
      }
    }
    st.tick += 1;
    return ok;
  }

  function shipsOf(st, player) { let s = 0; for (const p of st.planets) if (p.owner === player) s += p.ships; return s; }
  function isTerminal(st) {
    if (st.tick > st.maxTicks) return true;
    let a = false, b = false;
    for (const p of st.planets) { if (p.owner === 1) a = true; else if (p.owner === 2) b = true; }
    return !a || !b;
  }
  /** 1 or 2 for the player with more ships on planets, 0 for a tie. */
  function leader(st) { const a = shipsOf(st, 1), b = shipsOf(st, 2); return a === b ? 0 : a > b ? 1 : 2; }
  function cloneState(st) { return JSON.parse(JSON.stringify(st)); }

  // ---------------------------------------------------------------- their greedy sample agent

  function greedyAction(st, me) {
    const P = st.planets;
    let source = null;
    for (const p of P) if (p.owner === me && p.tr === null && p.ships > 10 && (!source || p.ships > source.ships)) source = p;
    if (!source) return null;
    let target = null, best = Infinity;
    for (const t of P) {
      if (t.owner === me) continue;
      const dist = Math.hypot(source.x - t.x, source.y - t.y);
      const sc = (t.owner === 0 ? t.ships : t.ships * 1.5) + dist - 2 * t.growth;
      if (sc < best) { best = sc; target = t; }
    }
    if (!target) return null;
    const eta = Math.hypot(source.x - target.x, source.y - target.y) / st.speed;
    if (source.ships <= target.ships + target.growth * eta) return null;
    return { src: source.id, dst: target.id, ships: source.ships / 2 };
  }

  // ---------------------------------------------------------------- the planner (entry/planner.py)

  const HORIZON = 90, FRACTIONS = [0.25, 0.5, 1.0], SIZES = ['spare'], TOP = 128, BUDGET_MS = 90, CHUNK = 8;
  const LOOK_OURS = 8, LOOK_REPLIES = 4, LOOK_MS = 18, LOOK_SOURCES = 12, MIX = 0.5, PAIRS = 200, PAIR_MOST = 16, PAIR_WAIT = 30;
  const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());

  /** The game as flat arrays (agent.py `arrays`). */
  function arrays(st) {
    const P = st.planets, n = P.length;
    const s = {
      n, owner: new Int32Array(n), ships: new Float64Array(n), px: new Float64Array(n), py: new Float64Array(n),
      growth: new Float64Array(n), radius: new Float64Array(n), trOn: new Uint8Array(n), trOwner: new Int32Array(n),
      trDest: new Int32Array(n), trShips: new Float64Array(n), trX: new Float64Array(n), trY: new Float64Array(n),
      tick: st.tick, speed: st.speed, maxTicks: st.maxTicks,
    };
    P.forEach((p, i) => {
      s.owner[i] = p.owner; s.ships[i] = p.ships; s.px[i] = p.x; s.py[i] = p.y; s.growth[i] = p.growth; s.radius[i] = p.radius;
      if (p.tr) {
        s.trOn[i] = 1; s.trOwner[i] = p.tr.owner; s.trDest[i] = Math.min(Math.max(p.tr.dst, 0), n - 1);
        s.trShips[i] = p.tr.ships; s.trX[i] = p.tr.x; s.trY[i] = p.tr.y;
      }
    });
    return s;
  }

  /** Index of the tick (0 is the one being decided) in which a fleet `dist` from a planet's centre lands. */
  function arrivalStep(dist, radius, speed) { return Math.max(Math.floor((dist - radius) / speed) + 1, 0); }

  // one tick of the rules for one planet; the result is left in TR_O, TR_S
  let TR_O = 0, TR_S = 0;
  function tickRule(o, sh, g, p1, p2) {
    const pending = p1 > 0 || p2 > 0;
    if (o === 0) {
      if (!pending) { TR_O = 0; TR_S = sh; return; }
      const net = p1 - p2, ns = sh - Math.abs(net);
      if (ns < 0) { TR_O = net > 0 ? 1 : 2; TR_S = -ns; } else { TR_O = 0; TR_S = ns; }
      return;
    }
    if (!pending) { TR_O = o; TR_S = sh + g; return; }
    const ps = sh + g + (o === 1 ? p1 - p2 : p2 - p1);
    if (ps < 0) { TR_O = 3 - o; TR_S = -ps; } else { TR_O = o; TR_S = ps; }
  }

  /** Owners and ships of m planets just before each of H ticks ([H*m], tick-major) given what lands on them
      (q1, q2: [H*m]). Only ticks with a landing somewhere are stepped (`base_states`). */
  function baseStates(owner0, ships0, growth, q1, q2, H, m) {
    const po = new Int32Array(H * m), ps = new Float64Array(H * m);
    const owner = Int32Array.from(owner0), ships = Float64Array.from(ships0);
    let cur = 0;
    for (let k = 0; k <= H; k++) {
      if (k < H) {
        let busy = false;
        for (let c = 0; c < m; c++) if (q1[k * m + c] > 0 || q2[k * m + c] > 0) { busy = true; break; }
        if (!busy) continue;
      }
      const end = Math.min(k + 1, H);
      for (let t = cur; t < end; t++) {
        for (let c = 0; c < m; c++) {
          po[t * m + c] = owner[c];
          ps[t * m + c] = ships[c] + (t - cur) * (owner[c] === 0 ? 0.0 : growth[c]);
        }
      }
      if (k < H) {
        for (let c = 0; c < m; c++) {
          tickRule(owner[c], ps[k * m + c], growth[c], q1[k * m + c], q2[k * m + c]);
          owner[c] = TR_O; ships[c] = TR_S;
        }
        cur = k + 1;
      }
    }
    return { po, ps };
  }

  /** `run`: the rules for m planets through H ticks of landings (q1, q2: [H*m]); returns final owners and ships. */
  function run(owner0, ships0, growth, q1, q2, H, m) {
    const owner = Int32Array.from(owner0), ships = Float64Array.from(ships0);
    let cur = 0;
    for (let k = 0; k < H; k++) {
      let busy = false;
      for (let c = 0; c < m; c++) if (q1[k * m + c] > 0 || q2[k * m + c] > 0) { busy = true; break; }
      if (!busy) continue;
      for (let c = 0; c < m; c++) {
        const sh = owner[c] === 0 ? ships[c] : ships[c] + growth[c] * (k - cur);
        tickRule(owner[c], sh, growth[c], q1[k * m + c], q2[k * m + c]);
        owner[c] = TR_O; ships[c] = TR_S;
      }
      cur = k + 1;
    }
    for (let c = 0; c < m; c++) if (owner[c] !== 0) ships[c] = ships[c] + growth[c] * (H - cur);
    return { owner, ships };
  }

  /** `settle` for one planet: state just before tick `start`, then its landings (ticks ev, ships p1/p2 each). */
  function settle(owner, ships, g, start, ev, p1, p2, len, H) {
    let cur = start, o = owner, sh = ships;
    for (let e = 0; e < len; e++) {
      const t = ev[e];
      if (t >= H) continue;
      if (o !== 0) sh = sh + g * (t - cur);
      tickRule(o, sh, g, p1[e], p2[e]);
      o = TR_O; sh = TR_S; cur = t + 1;
    }
    TR_O = o; TR_S = o !== 0 ? sh + g * (H - cur) : sh;
  }

  /** Ships landing on each planet at each tick from the fleets in flight: a1, a2 [H*n]. */
  function arrivals(s, H) {
    const n = s.n, a1 = new Float64Array(H * n), a2 = new Float64Array(H * n);
    for (let f = 0; f < n; f++) {
      if (!s.trOn[f]) continue;
      const d = s.trDest[f], dx = s.trX[f] - s.px[d], dy = s.trY[f] - s.py[d];
      const k = arrivalStep(Math.sqrt(dx * dx + dy * dy), s.radius[d], s.speed);
      if (k < H) (s.trOwner[f] === 1 ? a1 : a2)[k * n + d] += s.trShips[f];
    }
    return { a1, a2 };
  }

  /** Each planet's landing ticks in ascending order (`landing_ticks`), as lists. */
  function landingTicks(a1, a2, H, n) {
    const out = [];
    for (let p = 0; p < n; p++) {
      const t = [];
      for (let k = 0; k < H; k++) if (a1[k * n + p] + a2[k * n + p] > 0) t.push(k);
      out.push(t);
    }
    return out;
  }

  /** numpy's sum of a row of n numbers (pairwise: eight running sums, then the rest), so that ties between launches
      break the same way as in the entry. Exact for n up to 128. */
  function npsum(a, n) {
    if (n < 8) { let r = 0.0; for (let i = 0; i < n; i++) r += a[i]; return r; }
    let r0 = a[0], r1 = a[1], r2 = a[2], r3 = a[3], r4 = a[4], r5 = a[5], r6 = a[6], r7 = a[7], i = 8;
    for (; i < n - (n % 8); i += 8) {
      r0 += a[i]; r1 += a[i + 1]; r2 += a[i + 2]; r3 += a[i + 3]; r4 += a[i + 4]; r5 += a[i + 5]; r6 += a[i + 6]; r7 += a[i + 7];
    }
    let r = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7));
    for (; i < n; i++) r += a[i];
    return 0.0 + r;
  }
  const SROW = Array.from({ length: 7 }, () => new Float64Array(64));

  /** Hand score for `me` of a position (owners and ships of every planet, a row of n at offset `off`): `score`. */
  function score1(owner, ships, off, s, me, near) {
    const n = s.n, g = s.growth, opp = 3 - me;
    if (SROW[0].length < n) for (let k = 0; k < SROW.length; k++) SROW[k] = new Float64Array(n);
    const [sm, so, gm, go, fm, fo, iso] = SROW;
    let nM = 0, nO = 0;
    for (let p = 0; p < n; p++) {
      const o = owner[off + p], sh = ships[off + p], m = o === me ? 1.0 : 0.0, q = o === opp ? 1.0 : 0.0, fut = sh + 20.0 * g[p];
      sm[p] = sh * m; so[p] = sh * q; gm[p] = g[p] * m; go[p] = g[p] * q; fm[p] = fut * m; fo[p] = fut * q;
      nM += m; nO += q;
    }
    for (let p = 0; p < n; p++) {
      let lone = owner[off + p] === me && nM > 1;
      for (let q = 0; lone && q < n; q++) if (near[p * n + q] && owner[off + q] === me) lone = false;
      iso[p] = lone ? g[p] : 0.0;
    }
    let sc = npsum(sm, n) - npsum(so, n);
    sc += 10.0 * (npsum(gm, n) - npsum(go, n));
    sc += 0.5 * (npsum(fm, n) - npsum(fo, n));
    sc -= 2.0 * npsum(iso, n);
    sc += 5.0 * (nM - nO);
    sc += (nO === 0 && nM > 0 ? 10000.0 : 0.0) - (nM === 0 && nO > 0 ? 10000.0 : 0.0);
    return sc;
  }

  function scoreRows(owner, ships, rows, s, me, near) {
    const out = new Float64Array(rows);
    for (let r = 0; r < rows; r++) out[r] = score1(owner, ships, r * s.n, s, me, near);
    return out;
  }

  /** Every launch the rules allow and where it leaves the map HORIZON ticks on, with its hand score (`candidates`). */
  function candidates(s, me, opt = {}) {
    const n = s.n, fractions = opt.fractions || FRACTIONS, extras = opt.extras || SIZES, reserved = opt.reserved || [];
    const H = Math.max(1, Math.min(opt.horizon || HORIZON, Math.floor(s.maxTicks - s.tick)));
    const { a1, a2 } = arrivals(s, H);
    const d = new Float64Array(n * n), near = new Uint8Array(n * n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const dx = s.px[i] - s.px[j], dy = s.py[i] - s.py[j];
      d[i * n + j] = Math.sqrt(dx * dx + dy * dy);
      near[i * n + j] = d[i * n + j] <= 160.0 && i !== j ? 1 : 0;
    }
    const { po, ps } = baseStates(s.owner, s.ships, s.growth, a1, a2, H, n);
    const baseOwner = new Int32Array(n), baseShips = new Float64Array(n);
    for (let p = 0; p < n; p++) {  // the last tick of the projection (`run` over a1[-1:], a2[-1:])
      tickRule(po[(H - 1) * n + p], ps[(H - 1) * n + p], s.growth[p], a1[(H - 1) * n + p], a2[(H - 1) * n + p]);
      baseOwner[p] = TR_O; baseShips[p] = TR_S;
    }
    const base = score1(baseOwner, baseShips, 0, s, me, near);

    const srcs = [];
    for (let p = 0; p < n; p++) if (s.owner[p] === me && !s.trOn[p] && s.ships[p] >= 1.0 && !reserved.includes(p)) srcs.push(p);
    if (!srcs.length) return null;
    const spare = new Float64Array(n);
    for (let p = 0; p < n; p++) {
      let kept = true, lo = Infinity;
      for (let t = 0; t < H; t++) { if (po[t * n + p] !== me) { kept = false; break; } lo = Math.min(lo, ps[t * n + p]); }
      spare[p] = kept ? Math.floor(lo) : 0.0;
    }
    const options = fractions.map(f => ['f', f]).concat(extras.map(parseSize));
    const ci = [], cj = [], cn = [];
    const tried = new Map(srcs.map(i => [i, []]));
    options.forEach(([kind, v], f) => {
      for (const i of srcs) {
        const x = new Float64Array(n - 1), dst = new Int32Array(n - 1);
        for (let j = 0, q = 0; j < n; j++) {
          if (j === i) continue;
          dst[q] = j;
          x[q++] = kind === 'spare' ? Math.floor(spare[i] * v) : Math.max(1.0, Math.floor(s.ships[i] * v));
        }
        const earlier = tried.get(i);
        for (let q = 0; q < n - 1; q++) {
          let ok = x[q] >= 1.0 && x[q] <= s.ships[i];
          if (ok && f >= fractions.length) for (const e of earlier) if (e[q] === x[q]) { ok = false; break; }
          if (ok) { ci.push(i); cj.push(dst[q]); cn.push(x[q]); }
        }
        earlier.push(x);
      }
    });
    if (!ci.length) return null;
    const C = ci.length;
    const ticks = landingTicks(a1, a2, H, n);
    const owner = new Int32Array(C * n), ships = new Float64Array(C * n);
    const ev = new Int32Array(H + 1), p1 = new Float64Array(H + 1), p2 = new Float64Array(H + 1);
    for (let c = 0; c < C; c++) {
      const i = ci[c], j = cj[c], x = cn[c], off = c * n;
      owner.set(baseOwner, off); ships.set(baseShips, off);
      // the target: as projected until this fleet lands, then through that landing and the later ones
      const kArr = arrivalStep(d[i * n + j], s.radius[j], s.speed);
      if (kArr < H) {
        const kc = kArr;
        let len = 0;
        ev[len] = kc; p1[len] = a1[kc * n + j]; p2[len] = a2[kc * n + j]; len++;
        if (me === 1) p1[0] += x; else p2[0] += x;
        for (const t of ticks[j]) if (t > kc) { ev[len] = t; p1[len] = a1[t * n + j]; p2[len] = a2[t * n + j]; len++; }
        settle(po[kc * n + j], ps[kc * n + j], s.growth[j], kc, ev, p1, p2, len, H);
        owner[off + j] = TR_O; ships[off + j] = TR_S;
      }
      // the source: launching leaves it with fewer ships to meet whatever arrives
      let len = 0;
      for (const t of ticks[i]) { ev[len] = t; p1[len] = a1[t * n + i]; p2[len] = a2[t * n + i]; len++; }
      settle(s.owner[i], s.ships[i] - x, s.growth[i], 0, ev, p1, p2, len, H);
      owner[off + i] = TR_O; ships[off + i] = TR_S;
    }
    return {
      C, ci: Int32Array.from(ci), cj: Int32Array.from(cj), cn: Float64Array.from(cn), owner, ships,
      score: scoreRows(owner, ships, C, s, me, near), baseOwner, baseShips, base, H, a1, a2, d, near, po, ps, ticks,
    };
  }

  function parseSize(spec) {
    const [kind, ...args] = spec.split(':');
    if (kind === 'f') return ['f', parseFloat(args[0])];
    if (kind === 'spare') return ['spare', args.length ? parseFloat(args[0]) : 1.0];
    throw new Error('unsupported launch size ' + spec);
  }

  /** Best launch by hand score alone as {src, dst, ships}, or null to wait (`plan`). */
  function plan(s, me) {
    const c = candidates(s, me);
    if (!c) return null;
    let best = 0;
    for (let k = 1; k < c.C; k++) if (c.score[k] > c.score[best]) best = k;
    return c.score[best] > c.base ? { src: c.ci[best], dst: c.cj[best], ships: c.cn[best] } : null;
  }

  /** The opponent's best `top` answers to each of our options by its hand score (`replies`, without f or d). */
  function replies(s, me, c, opts, top, sources) {
    const opp = 3 - me, H = c.H, n = s.n, O = opts.length, fractions = FRACTIONS, extras = SIZES.map(parseSize);
    const g = s.growth, d = c.d, near = c.near;
    const A1 = [], A2 = [], PO = [], PS = [], FO = [], FS = [], ships0 = [], ticks = [];
    for (let o = 0; o < O; o++) {
      const a1 = Float64Array.from(c.a1), a2 = Float64Array.from(c.a2), po = Int32Array.from(c.po), ps = Float64Array.from(c.ps);
      const fo = Int32Array.from(c.baseOwner), fs = Float64Array.from(c.baseShips), sh0 = Float64Array.from(s.ships);
      const k = opts[o];
      if (k >= 0) {  // only our source and target differ from the nobody-launches projection
        const i = c.ci[k], j = c.cj[k], x = c.cn[k];
        sh0[i] -= x;
        const kk = arrivalStep(d[i * n + j], s.radius[j], s.speed);
        if (kk < H) (me === 1 ? a1 : a2)[kk * n + j] += x;
        for (const p of [i, j]) {
          const q1 = new Float64Array(H), q2 = new Float64Array(H);
          for (let t = 0; t < H; t++) { q1[t] = a1[t * n + p]; q2[t] = a2[t * n + p]; }
          const b = baseStates([s.owner[p]], [sh0[p]], [g[p]], q1, q2, H, 1);
          for (let t = 0; t < H; t++) { po[t * n + p] = b.po[t]; ps[t * n + p] = b.ps[t]; }
          tickRule(b.po[H - 1], b.ps[H - 1], g[p], q1[H - 1], q2[H - 1]);
          fo[p] = TR_O; fs[p] = TR_S;
        }
      }
      A1.push(a1); A2.push(a2); PO.push(po); PS.push(ps); FO.push(fo); FS.push(fs); ships0.push(sh0);
      ticks.push(landingTicks(a1, a2, H, n));
    }
    const out = { owner: [], ships: [], found: [] };  // per option: up to `top` rows
    const free = [];
    for (let p = 0; p < n; p++) if (s.owner[p] === opp && !s.trOn[p] && s.ships[p] >= 1.0) free.push(p);
    free.sort((a, b) => s.ships[b] - s.ships[a] || a - b);
    const src = free.slice(0, sources);
    const nF = fractions.length + extras.length;
    const side = o => (o === opp ? 1.0 : o === me ? -1.0 : 0.0);
    const ev = new Int32Array(H + 1), p1 = new Float64Array(H + 1), p2 = new Float64Array(H + 1);
    for (let o = 0; o < O; o++) {
      const fo = FO[o], fs = FS[o], po = PO[o], ps = PS[o], a1 = A1[o], a2 = A2[o], tk = ticks[o];
      const answers = [];  // {sc, t, i, x, tO, tS, sO, sS}, in the order numpy's nonzero lists them
      if (src.length) {
        const per = new Float64Array(n), mine = new Uint8Array(n), friends0 = new Float64Array(n);
        let nMine0 = 0, nTheirs0 = 0;
        for (let p = 0; p < n; p++) {
          per[p] = side(fo[p]) * (1.5 * fs[p] + 20.0 * g[p] + 5.0);
          mine[p] = fo[p] === opp ? 1 : 0;
          nMine0 += mine[p]; nTheirs0 += fo[p] === me ? 1 : 0;
        }
        for (let p = 0; p < n; p++) { let f = 0; for (let q = 0; q < n; q++) if (mine[q] && near[p * n + q]) f++; friends0[p] = f; }
        const perSum = npsum(per, n);
        for (const i of src) {
          // launch sizes for this source
          const xs = [];
          for (const f of fractions) xs.push(Math.max(1.0, Math.floor(s.ships[i] * f)));
          let kept = true, lo = Infinity;
          for (let t = 0; t < H; t++) { if (po[t * n + i] !== opp) { kept = false; break; } lo = Math.min(lo, ps[t * n + i]); }
          const sp = kept ? Math.floor(lo) : 0.0;
          for (const [kind, v] of extras) xs.push(kind === 'spare' ? Math.floor(sp * v) : Math.max(1.0, Math.floor(s.ships[i] * v)));
          for (let f = 0; f < nF; f++) {
            const x = xs[f];
            let ok = x >= 1.0 && x <= s.ships[i];
            if (f >= fractions.length) for (let e = 0; e < f; e++) if (xs[e] === x) ok = false;
            if (!ok) continue;
            // the source: fewer ships from the start, through the landings on it
            let len = 0;
            for (const t of tk[i]) { ev[len] = t; p1[len] = a1[t * n + i]; p2[len] = a2[t * n + i]; len++; }
            settle(s.owner[i], ships0[o][i] - x, g[i], 0, ev, p1, p2, len, H);
            const sO = TR_O, sS = TR_S;
            for (let t = 0; t < n; t++) {
              if (t === i) continue;
              // the target: as projected until the fleet lands, then stepped through its landings and the later ones
              const kArr = arrivalStep(d[i * n + t], s.radius[t], s.speed);
              let tO, tS;
              if (kArr < H) {
                const kc = kArr;
                len = 0;
                ev[len] = kc; p1[len] = a1[kc * n + t]; p2[len] = a2[kc * n + t]; len++;
                if (opp === 1) p1[0] += x; else p2[0] += x;
                for (const u of tk[t]) if (u > kc) { ev[len] = u; p1[len] = a1[u * n + t]; p2[len] = a2[u * n + t]; len++; }
                settle(po[kc * n + t], ps[kc * n + t], g[t], kc, ev, p1, p2, len, H);
                tO = TR_O; tS = TR_S;
              } else { tO = fo[t]; tS = fs[t]; }
              // the opponent's hand score of this answer, from the projection without it and the two planets it changes
              let sc = perSum - per[t] - per[i] + side(tO) * (1.5 * tS + 20.0 * g[t] + 5.0) + side(sO) * (1.5 * sS + 20.0 * g[i] + 5.0);
              const dT = (tO === opp ? 1 : 0) - mine[t], dS = (sO === opp ? 1 : 0) - mine[i];
              const nMine = nMine0 + dT + dS;
              const nTheirs = nTheirs0 + (tO === me ? 1 : 0) - (fo[t] === me ? 1 : 0) + (sO === me ? 1 : 0) - (fo[i] === me ? 1 : 0);
              let lone = 0;
              if (nMine > 1) {
                for (let p = 0; p < n; p++) {
                  const m = p === t ? tO === opp : p === i ? sO === opp : mine[p];
                  if (!m) continue;
                  const fr = friends0[p] + near[t * n + p] * dT + near[i * n + p] * dS;
                  if (fr === 0) lone += g[p];
                }
              }
              sc = sc - 2.0 * lone + 5.0 * (nMine - nTheirs) + (nTheirs === 0 && nMine > 0 ? 10000.0 : 0.0) - (nMine === 0 && nTheirs > 0 ? 10000.0 : 0.0);
              answers.push({ sc, t, i, x, tO, tS, sO, sS, k: answers.length });
            }
          }
        }
      }
      answers.sort((a, b) => b.sc - a.sc || a.k - b.k);
      const rowsO = [], rowsS = [];
      for (const a of answers.slice(0, top)) {
        const ro = Int32Array.from(fo), rs = Float64Array.from(fs);
        ro[a.t] = a.tO; rs[a.t] = a.tS; ro[a.i] = a.sO; rs[a.i] = a.sS;
        rowsO.push(ro); rowsS.push(rs);
      }
      out.owner.push(rowsO); out.ships.push(rowsS);
    }
    return out;
  }

  /** Two of our planets taking one neutral together, timed to land in the same tick (`pair_launches`). */
  function pairLaunches(s, me, c, most = PAIR_MOST, maxWait = PAIR_WAIT) {
    const H = c.H, n = s.n, d = c.d, po = c.po, ps = c.ps;
    const spare = new Float64Array(n);
    for (let p = 0; p < n; p++) {
      if (!(s.owner[p] === me && !s.trOn[p] && s.ships[p] >= 1.0)) continue;
      let kept = true, lo = Infinity;
      for (let t = 0; t < H; t++) { if (po[t * n + p] !== me) { kept = false; break; } lo = Math.min(lo, ps[t * n + p]); }
      spare[p] = kept ? Math.floor(lo) : 0.0;
    }
    const taken = new Uint8Array(n);
    for (let k = 0; k < c.C; k++) if (c.owner[k * n + c.cj[k]] === me) taken[c.cj[k]] = 1;
    const targets = [], srcs = [];
    for (let p = 0; p < n; p++) if (s.owner[p] === 0 && c.baseOwner[p] === 0 && !taken[p]) targets.push(p);
    for (let p = 0; p < n; p++) if (spare[p] >= 1.0) srcs.push(p);
    if (!targets.length || srcs.length < 2) return null;
    const first = [], second = [], waits = [], k1s = [], k2s = [];
    for (const j of targets) {
      const k = srcs.map(i => arrivalStep(d[i * n + j], s.radius[j], s.speed));
      for (let a = 0; a < srcs.length; a++) for (let b = 0; b < srcs.length; b++) {
        if (a === b || k[a] < k[b]) continue;
        const wait = Math.max(1, k[a] - k[b]), t = k[b] + wait;
        if (wait > maxWait || t >= H || po[t * n + j] !== 0) continue;
        const need = ps[t * n + j] + 1.0;
        for (const full of [a, b]) {
          const rest = full === a ? b : a;
          const xFull = spare[srcs[full]], xRest = Math.max(1.0, Math.floor(need - xFull) + 1.0);
          if (xFull >= need || xRest > spare[srcs[rest]]) continue;
          const [xa, xb] = full === a ? [xFull, xRest] : [xRest, xFull];
          first.push([srcs[a], j, xa]); second.push([srcs[b], j, xb]); waits.push(wait); k1s.push(k[a]); k2s.push(t);
        }
      }
    }
    if (!first.length) return null;
    const P = first.length, m = 3 * P;
    const q1 = new Float64Array(H * m), q2 = new Float64Array(H * m);
    const cols = new Int32Array(m);
    for (let r = 0; r < P; r++) { cols[r] = first[r][1]; cols[P + r] = first[r][0]; cols[2 * P + r] = second[r][0]; }
    for (let t = 0; t < H; t++) for (let q = 0; q < m; q++) { q1[t * m + q] = c.a1[t * n + cols[q]]; q2[t * m + q] = c.a2[t * n + cols[q]]; }
    const ours = me === 1 ? q1 : q2;
    for (let r = 0; r < P; r++) { ours[k1s[r] * m + r] += first[r][2]; ours[k2s[r] * m + r] += second[r][2]; }
    const own0 = new Int32Array(m), sh0 = new Float64Array(m), gr = new Float64Array(m);
    for (let r = 0; r < P; r++) {
      sh0[r] = s.ships[first[r][1]]; sh0[P + r] = s.ships[first[r][0]] - first[r][2]; sh0[2 * P + r] = s.ships[second[r][0]] - second[r][2];
    }
    for (let q = 0; q < m; q++) { own0[q] = s.owner[cols[q]]; gr[q] = s.growth[cols[q]]; }
    const f = run(own0, sh0, gr, q1, q2, H, m);
    const owner = new Int32Array(P * n), ships = new Float64Array(P * n);
    for (let r = 0; r < P; r++) {
      owner.set(c.baseOwner, r * n); ships.set(c.baseShips, r * n);
      for (const q of [r, P + r, 2 * P + r]) { owner[r * n + cols[q]] = f.owner[q]; ships[r * n + cols[q]] = f.ships[q]; }
    }
    const hand = scoreRows(owner, ships, P, s, me, c.near);
    let keep = [];
    for (let r = 0; r < P; r++) if (owner[r * n + first[r][1]] === me) keep.push(r);
    keep.sort((a, b) => hand[b] - hand[a] || a - b);
    keep = keep.slice(0, most);
    if (!keep.length) return null;
    return keep.map(r => ({ first: first[r], second: second[r], wait: waits[r], owner: owner.subarray(r * n, r * n + n), ships: ships.subarray(r * n, r * n + n), hand: hand[r] }));
  }

  // ---------------------------------------------------------------- the value network (entry/pwnet.py value_fast)

  const N_PLANET_FEATS = 21, N_GLOBAL_FEATS = 9;

  /** The value path's weights rearranged as `pwnet.prepare` does: q, k, v as one matrix with the attention scale in q,
      and each layer norm's gain and bias folded into the matrix after it. `raw` maps names to {shape, data}. */
  function prepareNet(raw, heads) {
    const A = name => raw[name];
    const dm = A('in.w').shape[1], f = N_PLANET_FEATS, scale = 1.0 / Math.sqrt(dm / heads);
    const inw = A('in.w').data;
    const net = {
      heads, dm, ff: A('l0.w1').shape[1],
      wPlanet: Float64Array.from(inw.subarray(0, f * dm)), wGlob: Float64Array.from(inw.subarray(f * dm)),
      bIn: Float64Array.from(A('in.b').data), layers: [],
    };
    const fold = (gain, bias, w, rows, cols, b0) => {  // (x * gain + bias) @ w + b0  ->  x @ w' + b'
      const w2 = new Float64Array(rows * cols), b2 = new Float64Array(cols);
      for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) { w2[r * cols + c] = gain[r] * w[r * cols + c]; b2[c] += bias[r] * w[r * cols + c]; }
      if (b0) for (let c = 0; c < cols; c++) b2[c] += b0[c];
      return [w2, b2];
    };
    for (let i = 0; A(`l${i}.wq`); i++) {
      const wq = A(`l${i}.wq`).data, wk = A(`l${i}.wk`).data, wv = A(`l${i}.wv`).data;
      const wqkv = new Float64Array(dm * 3 * dm);
      for (let r = 0; r < dm; r++) for (let c = 0; c < dm; c++) {
        wqkv[r * 3 * dm + c] = wq[r * dm + c] * scale;
        wqkv[r * 3 * dm + dm + c] = wk[r * dm + c];
        wqkv[r * 3 * dm + 2 * dm + c] = wv[r * dm + c];
      }
      const [wqkvF, bqkv] = fold(A(`l${i}.ln1.g`).data, A(`l${i}.ln1.b`).data, wqkv, dm, 3 * dm, null);
      const [w1, b1] = fold(A(`l${i}.ln2.g`).data, A(`l${i}.ln2.b`).data, A(`l${i}.w1`).data, dm, net.ff, A(`l${i}.b1`).data);
      net.layers.push({
        wqkv: wqkvF, bqkv, wo: Float64Array.from(A(`l${i}.wo`).data), bo: Float64Array.from(A(`l${i}.bo`).data),
        w1, b1, w2: Float64Array.from(A(`l${i}.w2`).data), b2: Float64Array.from(A(`l${i}.b2`).data),
      });
    }
    const gw = A('g.w').data;
    const [wPool, bG] = fold(A('out.ln.g').data, A('out.ln.b').data, gw.subarray(0, dm * dm), dm, dm, A('g.b').data);
    net.wPool = wPool; net.bG = bG; net.wGGlob = Float64Array.from(gw.subarray(dm * dm));
    net.v1w = Float64Array.from(A('v1.w').data); net.v1b = Float64Array.from(A('v1.b').data);
    net.v2w = Float64Array.from(A('v2.w').data); net.v2b = A('v2.b').data[0];
    return net;
  }

  /** {name: {shape, data}} from the value.json manifest and the value.bin bytes (an ArrayBuffer). */
  function unpackWeights(meta, buffer) {
    const all = new Float32Array(buffer), raw = {};
    for (const [name, a] of Object.entries(meta.arrays)) {
      const size = a.shape.reduce((x, y) => x * y, 1);
      raw[name] = { shape: a.shape, data: all.subarray(a.offset, a.offset + size) };
    }
    return prepareNet(raw, meta.heads);
  }

  // out[M x N] = A[M x K] @ W[K x N] + bias (four rows at a time)
  function matmul(A, M, K, W, N, bias, out) {
    let i = 0;
    for (; i + 4 <= M; i += 4) {
      const o0 = i * N, o1 = o0 + N, o2 = o1 + N, o3 = o2 + N;
      for (let j = 0; j < N; j++) { const b = bias ? bias[j] : 0; out[o0 + j] = b; out[o1 + j] = b; out[o2 + j] = b; out[o3 + j] = b; }
      const r0 = i * K, r1 = r0 + K, r2 = r1 + K, r3 = r2 + K;
      for (let k = 0; k < K; k++) {
        const x0 = A[r0 + k], x1 = A[r1 + k], x2 = A[r2 + k], x3 = A[r3 + k], wo = k * N;
        for (let j = 0; j < N; j++) {
          const w = W[wo + j];
          out[o0 + j] += x0 * w; out[o1 + j] += x1 * w; out[o2 + j] += x2 * w; out[o3 + j] += x3 * w;
        }
      }
    }
    for (; i < M; i++) {
      const oo = i * N, ro = i * K;
      for (let j = 0; j < N; j++) out[oo + j] = bias ? bias[j] : 0;
      for (let k = 0; k < K; k++) { const x = A[ro + k], wo = k * N; for (let j = 0; j < N; j++) out[oo + j] += x * W[wo + j]; }
    }
  }

  // layer norm of each row, without gain and bias (those are folded into the next matrix)
  function norm(X, rows, D, out) {
    for (let r = 0; r < rows; r++) {
      const o = r * D;
      let mu = 0;
      for (let j = 0; j < D; j++) mu += X[o + j];
      mu /= D;
      let v = 0;
      for (let j = 0; j < D; j++) { const c = X[o + j] - mu; out[o + j] = c; v += c * c; }
      const inv = 1.0 / Math.sqrt(v * (1.0 / D) + 1e-5);
      for (let j = 0; j < D; j++) out[o + j] *= inv;
    }
  }

  const BUF = {};
  function buf(name, size) { if (!BUF[name] || BUF[name].length < size) BUF[name] = new Float64Array(size); return BUF[name]; }

  /** Run the network in WebAssembly (wasm/net.c, compiled to net.wasm) from now on: `module` is a WebAssembly.Module.
      Three to four times faster than the JavaScript below, in single precision like the entry. */
  function attachWasm(net, module) {
    const ex = new WebAssembly.Instance(module, {}).exports;
    const parts = [net.wPlanet, net.wGlob, net.bIn];
    for (const L of net.layers) parts.push(L.wqkv, L.bqkv, L.wo, L.bo, L.w1, L.b1, L.w2, L.b2);
    parts.push(net.wPool, net.bG, net.wGGlob, net.v1w, net.v1b, net.v2w, [net.v2b]);
    const total = parts.reduce((x, p) => x + p.length, 0);
    const wPtr = align16(ex.__heap_base.value);
    growTo(ex.memory, wPtr + total * 4);
    const f32 = new Float32Array(ex.memory.buffer);
    let off = wPtr / 4;
    for (const p of parts) { f32.set(p, off); off += p.length; }
    net.wasm = { ex, wPtr, free: align16(off * 4) };
    return net;
  }
  function align16(x) { return (x + 15) & ~15; }
  function growTo(mem, bytes) { const need = bytes - mem.buffer.byteLength; if (need > 0) mem.grow(Math.ceil(need / 65536)); }

  function valueWasm(net, planet, glob, B, n) {
    const { ex, wPtr, free } = net.wasm;
    const pPlanet = free, pGlob = align16(pPlanet + planet.length * 4), pOut = align16(pGlob + glob.length * 4), pScr = align16(pOut + B * 4);
    growTo(ex.memory, pScr + ex.scratch_size(net.dm, net.ff, B, n) * 4);
    const f32 = new Float32Array(ex.memory.buffer);
    f32.set(planet, pPlanet / 4);
    f32.set(glob, pGlob / 4);
    ex.value_batch(wPtr, net.dm, net.ff, net.heads, net.layers.length, pPlanet, pGlob, B, n, pOut, pScr);
    return Float64Array.from(new Float32Array(ex.memory.buffer, pOut, B));
  }

  /** Values of B positions of the same game: planet [B*n*21], glob [B*9] -> Float64Array(B). */
  function valueFast(net, planet, glob, B, n) {
    if (net.wasm) return valueWasm(net, planet, glob, B, n);
    const dm = net.dm, heads = net.heads, dh = dm / heads, ff = net.ff, R = B * n;
    const h = buf('h', R * dm), a = buf('a', R * dm), qkv = buf('qkv', R * 3 * dm), o = buf('o', R * dm), u = buf('u', R * ff);
    const t = buf('t', R * dm), att = buf('att', n), gv = buf('gv', dm);
    matmul(planet, R, N_PLANET_FEATS, net.wPlanet, dm, null, h);
    for (let b = 0; b < B; b++) {
      matmul(glob.subarray(b * N_GLOBAL_FEATS, (b + 1) * N_GLOBAL_FEATS), 1, N_GLOBAL_FEATS, net.wGlob, dm, net.bIn, gv);
      for (let p = 0; p < n; p++) { const off = (b * n + p) * dm; for (let j = 0; j < dm; j++) h[off + j] += gv[j]; }
    }
    for (const L of net.layers) {
      norm(h, R, dm, a);
      matmul(a, R, dm, L.wqkv, 3 * dm, L.bqkv, qkv);
      for (let b = 0; b < B; b++) {
        for (let hd = 0; hd < heads; hd++) {
          const qo = hd * dh, ko = dm + hd * dh, vo = 2 * dm + hd * dh;
          for (let qi = 0; qi < n; qi++) {
            const qrow = (b * n + qi) * 3 * dm;
            let mx = -Infinity;
            for (let kj = 0; kj < n; kj++) {
              const krow = (b * n + kj) * 3 * dm;
              let sdot = 0;
              for (let e = 0; e < dh; e++) sdot += qkv[qrow + qo + e] * qkv[krow + ko + e];
              att[kj] = sdot;
              if (sdot > mx) mx = sdot;
            }
            let sum = 0;
            for (let kj = 0; kj < n; kj++) { att[kj] = Math.exp(att[kj] - mx); sum += att[kj]; }
            const inv = 1.0 / sum, orow = (b * n + qi) * dm + hd * dh;
            for (let e = 0; e < dh; e++) o[orow + e] = 0;
            for (let kj = 0; kj < n; kj++) {
              const w = att[kj] * inv, vrow = (b * n + kj) * 3 * dm + vo;
              for (let e = 0; e < dh; e++) o[orow + e] += w * qkv[vrow + e];
            }
          }
        }
      }
      matmul(o, R, dm, L.wo, dm, L.bo, t);
      for (let r = 0; r < R * dm; r++) h[r] += t[r];
      norm(h, R, dm, a);
      matmul(a, R, dm, L.w1, ff, L.b1, u);
      for (let r = 0; r < R * ff; r++) if (u[r] < 0) u[r] = 0;
      matmul(u, R, ff, L.w2, dm, L.b2, t);
      for (let r = 0; r < R * dm; r++) h[r] += t[r];
    }
    norm(h, R, dm, a);
    const out = new Float64Array(B), pooled = new Float64Array(dm), g = new Float64Array(dm), g2 = new Float64Array(dm), v1 = new Float64Array(dm);
    for (let b = 0; b < B; b++) {
      pooled.fill(0);
      for (let p = 0; p < n; p++) { const off = (b * n + p) * dm; for (let j = 0; j < dm; j++) pooled[j] += a[off + j]; }
      for (let j = 0; j < dm; j++) pooled[j] /= n;
      matmul(pooled, 1, dm, net.wPool, dm, net.bG, g);
      matmul(glob.subarray(b * N_GLOBAL_FEATS, (b + 1) * N_GLOBAL_FEATS), 1, N_GLOBAL_FEATS, net.wGGlob, dm, null, g2);
      for (let j = 0; j < dm; j++) g[j] = Math.max(g[j] + g2[j], 0);
      matmul(g, 1, dm, net.v1w, dm, net.v1b, v1);
      let v = net.v2b;
      for (let j = 0; j < dm; j++) v += Math.max(v1[j], 0) * net.v2w[j];
      out[b] = v;
    }
    return out;
  }

  /** `still_features`: network inputs for positions with no fleets in flight (rows of owners and ships). */
  function stillFeatures(s, me, owners, shipss, tick) {
    const B = owners.length, n = s.n, F = N_PLANET_FEATS;
    const planet = new Float64Array(B * n * F), glob = new Float64Array(B * N_GLOBAL_FEATS);
    const maxTicks = s.maxTicks;
    for (let b = 0; b < B; b++) {
      const o = owners[b], sh = shipss[b];
      let myT = 0, opT = 0, myG = 0, opG = 0, nM = 0, nO = 0;
      for (let p = 0; p < n; p++) {
        const r = (b * n + p) * F, mine = o[p] === me, opp = o[p] === 3 - me, x = me === 2 ? WIDTH - s.px[p] : s.px[p], y = me === 2 ? HEIGHT - s.py[p] : s.py[p];
        planet[r] = 1; planet[r + 1] = mine ? 1 : 0; planet[r + 2] = opp ? 1 : 0; planet[r + 3] = o[p] === 0 ? 1 : 0;
        planet[r + 4] = sh[p] / 50.0; planet[r + 5] = Math.log1p(sh[p]) / 4.0; planet[r + 6] = s.growth[p] * 10.0;
        planet[r + 7] = x / WIDTH; planet[r + 8] = y / HEIGHT;
        planet[r + 18] = (mine ? sh[p] : -sh[p]) / 50.0; planet[r + 19] = mine && sh[p] >= 1.0 ? 1 : 0; planet[r + 20] = s.radius[p] / 40.0;
        if (mine) { myT += sh[p]; myG += s.growth[p]; nM++; } else if (opp) { opT += sh[p]; opG += s.growth[p]; nO++; }
      }
      const gl = b * N_GLOBAL_FEATS, nv = Math.max(n, 1);
      glob[gl] = tick / maxTicks; glob[gl + 1] = (maxTicks - tick) / 1000.0; glob[gl + 2] = s.speed / 5.0; glob[gl + 3] = nv / 30.0;
      glob[gl + 4] = myT / (myT + opT + 1.0); glob[gl + 5] = myG / (myG + opG + 1e-6);
      glob[gl + 6] = nM / nv; glob[gl + 7] = nO / nv; glob[gl + 8] = Math.log1p(myT + opT) / 8.0;
    }
    return { planet, glob };
  }

  function valueMany(net, s, me, owners, shipss, tick) {
    const { planet, glob } = stillFeatures(s, me, owners, shipss, tick);
    return valueFast(net, planet, glob, owners.length, s.n);
  }

  /** (1 - mix) * value + mix * tanh(hand / 200), and 10 more (less) where the opponent (we) has no planet left. */
  function totalsOf(net, s, me, owners, shipss, hand, tick, mix) {
    const v = valueMany(net, s, me, owners, shipss, tick), out = new Float64Array(owners.length);
    for (let b = 0; b < owners.length; b++) {
      let theirs = false, ours = false;
      for (const o of owners[b]) { if (o === 3 - me) theirs = true; else if (o === me) ours = true; }
      const won = !theirs, lost = !ours;
      out[b] = (1.0 - mix) * v[b] + mix * Math.tanh(hand[b] / 200.0) + (won && !lost ? 10.0 : 0.0) - (lost && !won ? 10.0 : 0.0);
    }
    return out;
  }

  function argmax(a) { let b = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[b]) b = i; return b; }

  /** `plan_ranked`: the best launches by hand score (and waiting) re-ranked by the value network, eight at a time
      until `top` are seen or the time budget would be exceeded; then the best few look one opponent reply ahead.
      Returns {src, dst, ships} or null to wait; `out.follow` is set for a two-planet capture and `out.info` says
      how much was looked at. */
  function planRanked(s, me, net, opt = {}) {
    const started = now();
    const top = opt.top || TOP, mix = opt.mix ?? MIX, budget = opt.budgetMs ?? BUDGET_MS, lookMs = opt.lookMs ?? LOOK_MS;
    const reserved = opt.reserved || [], out = opt.out, pairs = opt.pairs ?? PAIRS;
    let ours = opt.look === false ? 0 : LOOK_OURS;
    const nRep = LOOK_REPLIES, perOpt = nRep;
    const c = candidates(s, me, { reserved });
    if (!c) return null;
    const n = s.n, tick = s.tick + c.H;
    const order = Array.from(c.score.keys()).sort((a, b) => c.score[b] - c.score[a] || a - b).slice(0, top);
    // launches that leave the map the same (say, fleets that land after the horizon) are scored once, at the first
    const firstOf = new Map(), rep = new Int32Array(order.length), fresh = [];
    order.forEach((k, pos) => {
      const key = c.owner.subarray(k * n, k * n + n).join(',') + '|' + c.ships.subarray(k * n, k * n + n).join(',');
      if (!firstOf.has(key)) { firstOf.set(key, pos); fresh.push(pos); }
      rep[pos] = firstOf.get(key);
    });
    const tot = new Float64Array(order.length), done = new Uint8Array(order.length);
    let took = 0, wait = 0;
    const row = k => [c.owner.subarray(k * n, k * n + n), c.ships.subarray(k * n, k * n + n)];
    for (let lo = 0; lo < fresh.length; lo += CHUNK) {
      const reserve = ours ? lookMs + took * ours * perOpt / CHUNK : 0.0;
      if (lo && now() - started + took + reserve > budget) break;
      const t0 = now();
      const idx = fresh.slice(lo, lo + CHUNK).map(p => order[p]);
      const owners = idx.map(k => row(k)[0]), shipss = idx.map(k => row(k)[1]), hand = idx.map(k => c.score[k]);
      if (lo === 0) { owners.push(c.baseOwner); shipss.push(c.baseShips); hand.push(c.base); }
      const t = totalsOf(net, s, me, owners, shipss, hand, tick, mix);
      fresh.slice(lo, lo + CHUNK).forEach((p, q) => { tot[p] = t[q]; done[p] = 1; });
      if (lo === 0) wait = t[t.length - 1];
      took = Math.max(took, now() - t0);
    }
    const seen = [];
    for (let p = 0; p < order.length; p++) if (done[rep[p]]) seen.push(p);
    let cut = 0;
    while (cut < seen.length && seen[cut] < CHUNK) cut++;  // waiting goes after the first eight
    const seenK = seen.slice(0, cut).map(p => order[p]).concat([-1], seen.slice(cut).map(p => order[p]));
    const seenT = seen.slice(0, cut).map(p => tot[rep[p]]).concat([wait], seen.slice(cut).map(p => tot[rep[p]]));
    let best = argmax(seenT);
    const info = { launches: c.C, ranked: seen.length, looked: 0, pair: false };
    if (out) out.info = info;
    if (s.tick < pairs && out && !reserved.length) {  // two planets taking a neutral together, if that beats every single launch
      const pr = pairLaunches(s, me, c);
      if (pr) {
        const pTot = totalsOf(net, s, me, pr.map(x => x.owner), pr.map(x => x.ships), pr.map(x => x.hand), tick, mix);
        const b = argmax(pTot);
        if (pTot[b] > seenT[best]) {
          const [i2, j2, x2] = pr[b].second;
          out.follow = { wait: pr[b].wait, move: { src: i2, dst: j2, ships: x2 } };
          info.pair = true;
          const [i, j, x] = pr[b].first;
          return { src: i, dst: j, ships: x };
        }
      }
    }
    if (ours && took > 0) {
      const left = budget - (now() - started) - lookMs;
      ours = Math.min(ours, Math.floor(Math.trunc(left / (took / CHUNK)) / perOpt));
    }
    if (ours >= 2) {
      const pick = Array.from(seenT.keys()).sort((a, b) => seenT[b] - seenT[a] || a - b).slice(0, ours);
      const opts = pick.map(p => seenK[p]);
      const r = replies(s, me, c, opts, nRep, LOOK_SOURCES);
      const worst = pick.map(p => seenT[p]);
      const ro = [], rs = [], ro_o = [];
      r.owner.forEach((rows, o) => rows.forEach((x, q) => { ro.push(x); rs.push(r.ships[o][q]); ro_o.push(o); }));
      if (ro.length) {
        const hand = ro.map((x, q) => score1(x, rs[q], 0, s, me, c.near));
        const rt = totalsOf(net, s, me, ro, rs, hand, tick, mix);
        ro_o.forEach((o, q) => { if (rt[q] < worst[o]) worst[o] = rt[q]; });
      }
      best = pick[argmax(worst)];
      info.looked = ours;
    }
    const k = seenK[best];
    if (k < 0) return null;
    return { src: c.ci[k], dst: c.cj[k], ships: c.cn[k] };
  }

  // ---------------------------------------------------------------- opponents with state between ticks

  /** Our entry as it plays a game (planner.py PlannerAgent.get_action): `kind` is 'entry', 'planner' or 'greedy'. */
  function Agent(kind, net) {
    this.kind = kind; this.net = net; this.follow = null;
  }
  Agent.prototype.reset = function () { this.follow = null; };
  Agent.prototype.act = function (st, me, opt = {}) {
    if (this.kind === 'greedy') return { action: greedyAction(st, me), info: null };
    if (this.kind === 'planner' || !this.net) {
      const a = plan(arrays(st), me);
      if (a) a.ships = Math.min(a.ships, st.planets[a.src].ships);
      return { action: a, info: null };
    }
    const follow = this.follow;
    this.follow = null;
    if (follow && follow.tick >= st.tick) {  // the second half of a two-planet capture
      const { src: i, dst: j, ships: x } = follow.move, p = st.planets[i], t = st.planets[j];
      if (p.owner === me && p.tr === null && t.owner === 0) {
        if (follow.tick > st.tick) this.follow = follow;  // not yet: keep its planet out of other launches
        else if (p.ships >= x) return { action: { src: i, dst: j, ships: x }, info: { second: true } };
      }
    }
    const out = {};
    const a = planRanked(arrays(st), me, this.net, {
      out, budgetMs: opt.budgetMs, reserved: this.follow ? [this.follow.move.src] : [],
    });
    if (out.follow) this.follow = { tick: st.tick + out.follow.wait, move: out.follow.move };
    if (a) a.ships = Math.min(a.ships, st.planets[a.src].ships);
    return { action: a, info: out.info || null };
  };

  const PW = {
    WIDTH, HEIGHT, mulberry32, leagueParams, createGame, newGame, applyAction, step, isTerminal, leader, shipsOf, cloneState,
    greedyAction, arrays, arrivalStep, candidates, plan, replies, pairLaunches, planRanked, stillFeatures, valueFast,
    valueMany, totalsOf, prepareNet, unpackWeights, attachWasm, score1, Agent,
  };
  root.PW = PW;
  if (typeof module !== 'undefined' && module.exports) module.exports = PW;
})(typeof self !== 'undefined' ? self : globalThis);

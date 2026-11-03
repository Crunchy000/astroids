// Check pw.js against what the Python entry did (a file written by tools/dump_parity.py).
//   node planetwars/play/tools/parity.js PARITY.json [--js]   (--js: the network in JavaScript instead of WebAssembly)
'use strict';
const fs = require('fs');
const path = require('path');
const PW = require('../pw.js');

const HERE = path.join(__dirname, '..');
const data = JSON.parse(fs.readFileSync(process.argv[2]));
const bin = fs.readFileSync(path.join(HERE, 'value.bin'));
const net = PW.unpackWeights(JSON.parse(fs.readFileSync(path.join(HERE, 'value.json'))), bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.length));
if (!process.argv.includes('--js')) PW.attachWasm(net, new WebAssembly.Module(fs.readFileSync(path.join(HERE, 'net.wasm'))));

const same = (a, b) => (a === null && b === null) || (a && b && a.src === b.src && a.dst === b.dst && Math.abs(a.ships - b.ships) < 1e-9);
const fmt = a => (a ? `${a.src}->${a.dst} x${+a.ships.toFixed(3)}` : 'wait');
let failures = 0;

// 1. the rules: replay every recorded game and compare the states
let worst = 0, checked = 0;
for (const g of data.engine) {
  const st = PW.cloneState(g.start);
  g.actions.forEach(([a1, a2]) => {
    PW.step(st, { 1: a1, 2: a2 });
    const want = g.checkpoints[st.tick];
    if (!want) return;
    checked++;
    want.planets.forEach((p, i) => {
      const q = st.planets[i];
      if (p.owner !== q.owner || !p.tr !== !q.tr) { failures++; console.log(`engine: tick ${st.tick} planet ${i} differs`); }
      worst = Math.max(worst, Math.abs(p.ships - q.ships));
      if (p.tr && q.tr) worst = Math.max(worst, Math.abs(p.tr.x - q.tr.x), Math.abs(p.tr.y - q.tr.y));
    });
  });
  if (PW.isTerminal(st) !== true) { failures++; console.log('engine: the replay did not end where the recorded game did'); }
}
console.log(`engine: ${data.engine.length} games, ${checked} checkpoints, largest difference ${worst.toExponential(2)}`);

// 2. single positions: the hand-scored planner, the network's value, and the entry's choice with no time limit
const stats = { plan: [0, 0], ranked: [0, 0], value: 0, score: 0, follow: [0, 0] };
for (const pos of data.positions) {
  for (const me of [1, 2]) {
    const want = pos.sides[me], s = PW.arrays(pos.state);
    const c = PW.candidates(s, me);
    if ((c ? c.C : 0) !== want.n_candidates) { failures++; console.log(`tick ${s.tick} side ${me}: ${c && c.C} launches, Python ${want.n_candidates}`); }
    if (c) {
      stats.score = Math.max(stats.score, Math.abs(c.base - want.base_score));
      const v = PW.valueMany(net, s, me, [c.baseOwner], [c.baseShips], s.tick + c.H)[0];
      stats.value = Math.max(stats.value, Math.abs(v - want.base_value));
    }
    const p = PW.plan(s, me);
    stats.plan[0]++;
    if (same(p, want.plan)) stats.plan[1]++; else console.log(`tick ${s.tick} side ${me}: plan ${fmt(p)}, Python ${fmt(want.plan)}`);
    const out = {};
    const r = PW.planRanked(s, me, net, { budgetMs: 1e9, out });
    stats.ranked[0]++;
    if (same(r, want.ranked)) stats.ranked[1]++; else console.log(`tick ${s.tick} side ${me}: entry ${fmt(r)}, Python ${fmt(want.ranked)}`);
    if (want.follow || out.follow) {
      stats.follow[0]++;
      if (want.follow && out.follow && want.follow.wait === out.follow.wait && same(want.follow.move, out.follow.move)) stats.follow[1]++;
    }
  }
}
console.log(`positions: ${data.positions.length}, hand-scored planner same move ${stats.plan[1]}/${stats.plan[0]}, entry same move ${stats.ranked[1]}/${stats.ranked[0]}, ` +
  `two-planet follow-ups ${stats.follow[1]}/${stats.follow[0]}; largest difference: hand score ${stats.score.toExponential(2)}, value ${stats.value.toExponential(2)}`);

// 3. the entry as it plays, against their greedy agent: replay the recorded game, asking our agent at every tick
for (const g of data.agent) {
  const st = PW.cloneState(g.start), agent = new PW.Agent('entry', net);
  let agree = 0, first = -1;
  g.actions.forEach((acts, t) => {
    const mine = agent.act(PW.cloneState(st), g.entry, { budgetMs: 1e9 }).action;
    if (same(mine, acts[g.entry - 1])) agree++;
    else if (first < 0) { first = t; console.log(`agent game: first different move at tick ${t}: ${fmt(mine)}, Python ${fmt(acts[g.entry - 1])}`); }
    const theirs = PW.greedyAction(st, 3 - g.entry);
    if (!same(theirs, acts[2 - g.entry])) { failures++; console.log(`greedy differs at tick ${t}`); }
    PW.step(st, { 1: acts[0], 2: acts[1] });
  });
  console.log(`agent game (entry is player ${g.entry}): same move in ${agree} of ${g.actions.length} ticks`);
}
if (failures) { console.log(`${failures} failures`); process.exit(1); }

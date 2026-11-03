// The opponent runs here, off the page's thread, so the game keeps drawing while it thinks.
importScripts('pw.js');

let net = null, agent = null, wasm = false;

async function init() {
  const [meta, bin] = await Promise.all([fetch('value.json').then(r => r.json()), fetch('value.bin').then(r => r.arrayBuffer())]);
  net = PW.unpackWeights(meta, bin);
  try {
    const bytes = await fetch('net.wasm').then(r => r.arrayBuffer());
    PW.attachWasm(net, await WebAssembly.compile(bytes));
    wasm = true;
  } catch (e) {  // no WebAssembly SIMD here: the same network in plain JavaScript, about ten times slower
    wasm = false;
  }
  const warm = new PW.Agent('entry', net), g = PW.newGame(12345, 30);  // compile the hot paths before the first real move
  for (let i = 0; i < 3; i++) warm.act(g, 2, { budgetMs: 60 });
  postMessage({ type: 'ready', wasm });
}
const ready = init().catch(e => postMessage({ type: 'error', message: String(e) }));

onmessage = async e => {
  const m = e.data;
  await ready;
  if (m.type === 'new') {
    agent = new PW.Agent(m.kind, net);
  } else if (m.type === 'act' && agent) {
    const t0 = performance.now();
    let r;
    try {
      r = agent.act(m.state, m.me, { budgetMs: m.budgetMs });
    } catch (err) {  // never forfeit a tick over a bug
      r = { action: null, info: { error: String(err) } };
    }
    postMessage({ type: 'action', game: m.game, tick: m.state.tick, action: r.action, info: r.info, ms: performance.now() - t0 });
  }
};

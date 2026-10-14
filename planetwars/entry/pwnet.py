"""Planet Wars policy network: observation features and forward pass.

Written once against an array namespace `xp`, so the Docker entry runs it with numpy and training
runs the identical code with jax.numpy. Everything works on one unbatched game:

    s: the game as padded arrays (see `arrays_from_game_state`), me: 1 or 2 (the player we are).

The policy picks one (source, destination) pair or "do nothing", then one of `N_AMOUNTS` fleet sizes.
Positions are mirrored for Player 2, so the network always sees itself starting on the left.
"""
WIDTH, HEIGHT = 640.0, 480.0
FRACTIONS = (0.25, 0.5, 0.75, 1.0)
N_AMOUNTS = len(FRACTIONS) + 2  # plus "just enough to win" and "enough with a margin"
N_PLANET_FEATS = 21
N_GLOBAL_FEATS = 9
N_PAIR_FEATS = 5
NEG = -1e9


def features(xp, s, me):
    """Return (planet [N, N_PLANET_FEATS], global [N_GLOBAL_FEATS], pair [N, N, N_PAIR_FEATS], aux dict)."""
    ships = s.ships
    n = ships.shape[0]
    one = xp.ones_like(ships)
    valid = s.valid
    vf = valid.astype(ships.dtype)
    opp_id = 3 - me
    mine = valid & (s.owner == me)
    opp = valid & (s.owner == opp_id)
    neu = valid & (s.owner == 0)
    flip = me == 2
    x = xp.where(flip, WIDTH - s.pos[:, 0], s.pos[:, 0])
    y = xp.where(flip, HEIGHT - s.pos[:, 1], s.pos[:, 1])
    speed = s.speed * one[0]

    # fleets in flight (one per source planet)
    tr_mine = s.tr_on & (s.tr_owner == me)
    tr_opp = s.tr_on & (s.tr_owner == opp_id)
    dest_pos = s.pos[s.tr_dest]
    rem = xp.sqrt(xp.sum((s.tr_pos - dest_pos) ** 2, axis=-1))
    tr_eta = xp.maximum(rem - s.radius[s.tr_dest], 0.0) / speed
    to = s.tr_dest[:, None] == xp.arange(n)[None, :]                       # [src, dst]
    my_in = xp.sum(xp.where(to & tr_mine[:, None], s.tr_ships[:, None], 0.0), axis=0)
    opp_in = xp.sum(xp.where(to & tr_opp[:, None], s.tr_ships[:, None], 0.0), axis=0)
    big = 1e4
    my_eta = xp.min(xp.where(to & tr_mine[:, None], tr_eta[:, None], big), axis=0)
    opp_eta = xp.min(xp.where(to & tr_opp[:, None], tr_eta[:, None], big), axis=0)
    out_ships = xp.where(tr_mine, s.tr_ships, xp.where(tr_opp, -s.tr_ships, 0.0))
    sign = xp.where(mine, 1.0, -1.0)
    can_launch = mine & ~s.tr_on & (ships >= 1.0)

    planet = xp.stack([
        vf, mine.astype(vf.dtype), opp.astype(vf.dtype), neu.astype(vf.dtype),
        ships / 50.0, xp.log1p(ships) / 4.0, s.growth * 10.0, x / WIDTH, y / HEIGHT,
        s.tr_on.astype(vf.dtype), tr_mine.astype(vf.dtype), out_ships / 50.0,
        my_in / 50.0, opp_in / 50.0, xp.log1p(my_in) / 4.0, xp.log1p(opp_in) / 4.0,
        xp.where(my_eta < big, xp.exp(-my_eta / 50.0), 0.0), xp.where(opp_eta < big, xp.exp(-opp_eta / 50.0), 0.0),
        (sign * ships + my_in - opp_in) / 50.0, can_launch.astype(vf.dtype), s.radius / 40.0,
    ], axis=-1) * vf[:, None]

    my_total = xp.sum(xp.where(mine, ships, 0.0)) + xp.sum(xp.where(tr_mine, s.tr_ships, 0.0))
    opp_total = xp.sum(xp.where(opp, ships, 0.0)) + xp.sum(xp.where(tr_opp, s.tr_ships, 0.0))
    my_growth = xp.sum(xp.where(mine, s.growth, 0.0))
    opp_growth = xp.sum(xp.where(opp, s.growth, 0.0))
    n_valid = xp.maximum(xp.sum(vf), 1.0)
    tick = s.tick * one[0]
    max_ticks = s.max_ticks * one[0]
    glob = xp.stack([
        tick / max_ticks, (max_ticks - tick) / 1000.0, speed / 5.0, n_valid / 30.0,
        my_total / (my_total + opp_total + 1.0), my_growth / (my_growth + opp_growth + 1e-6),
        xp.sum(mine.astype(vf.dtype)) / n_valid, xp.sum(opp.astype(vf.dtype)) / n_valid,
        xp.log1p(my_total + opp_total) / 8.0,
    ])

    # pairs: launching from i to j
    d = xp.sqrt(xp.sum((s.pos[:, None, :] - s.pos[None, :, :]) ** 2, axis=-1))
    eta = d / speed
    strength = xp.where(opp[None, :], ships[None, :] + s.growth[None, :] * eta + opp_in[None, :] - my_in[None, :],
                        xp.where(neu[None, :], ships[None, :] + opp_in[None, :] - my_in[None, :],
                                 opp_in[None, :] - ships[None, :] - s.growth[None, :] * eta - my_in[None, :]))
    need = xp.maximum(strength, 0.0)
    src = ships[:, None] * xp.ones_like(eta)
    pair = xp.stack([eta / 100.0, xp.log1p(eta) / 5.0, strength / 50.0, (src - strength) / 50.0,
                     (src - need - 1.0) / 50.0], axis=-1)

    amounts = xp.stack([src * f for f in FRACTIONS] + [need + 1.0, need * 1.2 + 3.0], axis=-1)
    amounts = xp.minimum(xp.maximum(amounts, xp.minimum(src, 1.0)[..., None]), src[..., None])
    pair_ok = can_launch[:, None] & valid[None, :] & ~(xp.arange(n)[:, None] == xp.arange(n)[None, :])
    return planet, glob, pair, {"amounts": amounts, "pair_ok": pair_ok,
                                "potential": (my_total - opp_total) / (my_total + opp_total + 1.0)}


def _ln(xp, x, g, b):
    mu = xp.mean(x, axis=-1, keepdims=True)
    var = xp.mean((x - mu) ** 2, axis=-1, keepdims=True)
    return (x - mu) / xp.sqrt(var + 1e-5) * g + b


def _relu(xp, x):
    return xp.maximum(x, 0.0)


def n_layers(params) -> int:
    return sum(1 for k in params if k.endswith(".wq"))


def trunk(xp, params, heads, planet, glob, valid):
    """The transformer shared by every output: returns (token states [N, d_model], hidden global vector g)."""
    p = params
    n = planet.shape[0]
    tok = xp.concatenate([planet, xp.broadcast_to(glob[None, :], (n, glob.shape[0]))], axis=-1)
    h = tok @ p["in.w"] + p["in.b"]
    dm = h.shape[-1]
    dh = dm // heads
    kmask = valid[None, None, :]
    for i in range(n_layers(p)):
        a = _ln(xp, h, p[f"l{i}.ln1.g"], p[f"l{i}.ln1.b"])
        q = (a @ p[f"l{i}.wq"]).reshape(n, heads, dh)
        k = (a @ p[f"l{i}.wk"]).reshape(n, heads, dh)
        v = (a @ p[f"l{i}.wv"]).reshape(n, heads, dh)
        att = xp.einsum("qhd,khd->hqk", q, k) / (dh ** 0.5)
        att = xp.where(kmask, att, NEG)
        att = xp.exp(att - xp.max(att, axis=-1, keepdims=True))
        att = att / xp.sum(att, axis=-1, keepdims=True)
        o = xp.einsum("hqk,khd->qhd", att, v).reshape(n, dm)
        h = h + o @ p[f"l{i}.wo"] + p[f"l{i}.bo"]
        a = _ln(xp, h, p[f"l{i}.ln2.g"], p[f"l{i}.ln2.b"])
        h = h + _relu(xp, a @ p[f"l{i}.w1"] + p[f"l{i}.b1"]) @ p[f"l{i}.w2"] + p[f"l{i}.b2"]
    h = _ln(xp, h, p["out.ln.g"], p["out.ln.b"])
    vf = valid.astype(h.dtype)[:, None]
    pooled = xp.sum(h * vf, axis=0) / xp.maximum(xp.sum(vf), 1.0)
    g = _relu(xp, xp.concatenate([pooled, glob]) @ p["g.w"] + p["g.b"])
    return h, g


def value_only(xp, params, heads, planet, glob, valid):
    """Just the value output (skips the move heads): the cost of ranking a position."""
    _, g = trunk(xp, params, heads, planet, glob, valid)
    return (_relu(xp, g @ params["v1.w"] + params["v1.b"]) @ params["v2.w"] + params["v2.b"])[0]


def value_batch(xp, params, heads, planet, glob, valid):
    """`value_only` for B positions at once: planet [B, N, F], glob [B, G], valid [B, N] -> values [B]."""
    p = params
    b, n, _ = planet.shape
    tok = xp.concatenate([planet, xp.broadcast_to(glob[:, None, :], (b, n, glob.shape[-1]))], axis=-1)
    h = tok @ p["in.w"] + p["in.b"]
    dm = h.shape[-1]
    dh = dm // heads
    kmask = valid[:, None, None, :]
    for i in range(n_layers(p)):
        a = _ln(xp, h, p[f"l{i}.ln1.g"], p[f"l{i}.ln1.b"])
        q = (a @ p[f"l{i}.wq"]).reshape(b, n, heads, dh)
        k = (a @ p[f"l{i}.wk"]).reshape(b, n, heads, dh)
        v = (a @ p[f"l{i}.wv"]).reshape(b, n, heads, dh)
        att = xp.einsum("bqhd,bkhd->bhqk", q, k) / (dh ** 0.5)
        att = xp.where(kmask, att, NEG)
        att = xp.exp(att - xp.max(att, axis=-1, keepdims=True))
        att = att / xp.sum(att, axis=-1, keepdims=True)
        o = xp.einsum("bhqk,bkhd->bqhd", att, v).reshape(b, n, dm)
        h = h + o @ p[f"l{i}.wo"] + p[f"l{i}.bo"]
        a = _ln(xp, h, p[f"l{i}.ln2.g"], p[f"l{i}.ln2.b"])
        h = h + _relu(xp, a @ p[f"l{i}.w1"] + p[f"l{i}.b1"]) @ p[f"l{i}.w2"] + p[f"l{i}.b2"]
    h = _ln(xp, h, p["out.ln.g"], p["out.ln.b"])
    vf = valid.astype(h.dtype)[..., None]
    pooled = xp.sum(h * vf, axis=1) / xp.maximum(xp.sum(vf, axis=1), 1.0)
    g = _relu(xp, xp.concatenate([pooled, glob], axis=-1) @ p["g.w"] + p["g.b"])
    return (_relu(xp, g @ p["v1.w"] + p["v1.b"]) @ p["v2.w"] + p["v2.b"])[..., 0]


def forward(xp, params, heads, planet, glob, pair, valid, pair_ok):
    """Return (flat logits [N*N + 1] (last = do nothing), amount logits [N, N, N_AMOUNTS], value)."""
    p = params
    n = planet.shape[0]
    h, g = trunk(xp, params, heads, planet, glob, valid)
    value = (_relu(xp, g @ p["v1.w"] + p["v1.b"]) @ p["v2.w"] + p["v2.b"])[0]
    noop = (_relu(xp, g @ p["n1.w"] + p["n1.b"]) @ p["n2.w"] + p["n2.b"])
    hp = (h @ p["ps.w"])[:, None, :] + (h @ p["pd.w"])[None, :, :] + pair @ p["pp.w"] + p["pp.b"] + (g @ p["pg.w"])
    hp = _relu(xp, hp)
    hp = _relu(xp, hp @ p["p2.w"] + p["p2.b"])
    pair_logit = (hp @ p["p3.w"] + p["p3.b"])[..., 0]
    amount_logits = hp @ p["pa.w"] + p["pa.b"]
    flat = xp.concatenate([xp.where(pair_ok, pair_logit, NEG).reshape(n * n), noop])
    return flat, amount_logits, value


def param_shapes(d_model=128, layers=3, heads=4, d_pair=32, d_ff=256):
    """Every parameter's shape, by name (the npz the entry loads uses the same names)."""
    del heads
    fin = N_PLANET_FEATS + N_GLOBAL_FEATS
    s = {"in.w": (fin, d_model), "in.b": (d_model,)}
    for i in range(layers):
        s.update({f"l{i}.ln1.g": (d_model,), f"l{i}.ln1.b": (d_model,),
                  f"l{i}.wq": (d_model, d_model), f"l{i}.wk": (d_model, d_model), f"l{i}.wv": (d_model, d_model),
                  f"l{i}.wo": (d_model, d_model), f"l{i}.bo": (d_model,),
                  f"l{i}.ln2.g": (d_model,), f"l{i}.ln2.b": (d_model,),
                  f"l{i}.w1": (d_model, d_ff), f"l{i}.b1": (d_ff,), f"l{i}.w2": (d_ff, d_model), f"l{i}.b2": (d_model,)})
    s.update({"out.ln.g": (d_model,), "out.ln.b": (d_model,),
              "g.w": (d_model + N_GLOBAL_FEATS, d_model), "g.b": (d_model,),
              "v1.w": (d_model, d_model), "v1.b": (d_model,), "v2.w": (d_model, 1), "v2.b": (1,),
              "n1.w": (d_model, d_model), "n1.b": (d_model,), "n2.w": (d_model, 1), "n2.b": (1,),
              "ps.w": (d_model, d_pair), "pd.w": (d_model, d_pair), "pp.w": (N_PAIR_FEATS, d_pair), "pp.b": (d_pair,),
              "pg.w": (d_model, d_pair), "p2.w": (d_pair, d_pair), "p2.b": (d_pair,),
              "p3.w": (d_pair, 1), "p3.b": (1,), "pa.w": (d_pair, N_AMOUNTS), "pa.b": (N_AMOUNTS,)})
    return s


def decode(flat_index, amount_index, amounts):
    """(src, dst, ships) for a chosen flat pair index and amount option; src = -1 is do nothing."""
    n = amounts.shape[0]
    if flat_index >= n * n:
        return -1, -1, 0.0
    i, j = divmod(int(flat_index), n)
    return i, j, float(amounts[i, j, int(amount_index)])

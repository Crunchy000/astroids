"""Make a trained policy deeper without changing how it plays.

    python -m pwjax.grow runs/pw1/latest.npz deeper.npz --layers 6

New transformer layers go after the existing ones. Each is a residual block whose output
projections (`wo`, `bo`, `w2`, `b2`) start at zero, so it adds nothing to the activations and the
network gives the same outputs as before; training then puts the extra capacity to use.
"""
import argparse

import numpy as np

from .policy import pwnet


def grow(arrays: dict, layers: int, seed: int = 0) -> dict:
    """Copy of `arrays` (a params dict) with `layers` transformer layers in total."""
    have = pwnet.n_layers(arrays)
    assert layers >= have, f"already {have} layers"
    out = dict(arrays)
    d_model, d_ff = arrays["l0.wq"].shape[0], arrays["l0.w1"].shape[1]
    rng = np.random.default_rng(seed)
    for i in range(have, layers):
        shapes = pwnet.param_shapes(d_model=d_model, layers=i + 1, d_ff=d_ff)
        for name, shape in shapes.items():
            if not name.startswith(f"l{i}."):
                continue
            w = name.split(".", 1)[1]
            if w.endswith(".g"):
                out[name] = np.ones(shape, np.float32)
            elif w in ("wo", "bo", "w2", "b2") or len(shape) == 1:
                out[name] = np.zeros(shape, np.float32)
            else:
                out[name] = (rng.normal(size=shape) / np.sqrt(shape[0])).astype(np.float32)
    return out


def main(argv=None):
    a = argparse.ArgumentParser()
    a.add_argument("src")
    a.add_argument("dst")
    a.add_argument("--layers", type=int, required=True, help="total number of layers in the new model")
    a.add_argument("--seed", type=int, default=0)
    args = a.parse_args(argv)
    z = np.load(args.src)
    arrays = {k: z[k] for k in z.files}
    grown = grow({k: v for k, v in arrays.items() if not k.startswith("cfg.")}, args.layers, args.seed)
    grown.update({k: v for k, v in arrays.items() if k.startswith("cfg.")})
    np.savez(args.dst, **grown)
    print(f"{pwnet.n_layers(arrays)} -> {args.layers} layers, {sum(v.size for v in grown.values())} values")


if __name__ == "__main__":
    main()

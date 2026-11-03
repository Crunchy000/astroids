"""Write the entry's value network (entry/value.npz) as value.bin + value.json for the browser version.

Only the weights on the value path are kept (the move heads are never used by the planner). value.bin is the
float32 arrays back to back, little-endian; value.json names each one with its shape and offset (in floats).

    python planetwars/play/tools/export_value.py
"""
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ENTRY = HERE.parents[1] / "entry"
OUT = HERE.parent


def main(src=ENTRY / "value.npz"):
    z = np.load(src)
    keep = [k for k in sorted(z.files) if not k.startswith("cfg.") and k.split(".")[0] in
            ("in", "out", "g", "v1", "v2") or k[:1] == "l" and k[1].isdigit()]
    heads = int(z["cfg.heads"]) if "cfg.heads" in z.files else 4
    meta, chunks, off = {"heads": heads, "arrays": {}}, [], 0
    for k in keep:
        a = np.ascontiguousarray(z[k], dtype="<f4")
        meta["arrays"][k] = {"shape": list(a.shape), "offset": off}
        chunks.append(a.tobytes())
        off += a.size
    (OUT / "value.bin").write_bytes(b"".join(chunks))
    (OUT / "value.json").write_text(json.dumps(meta) + "\n")
    print(f"{len(keep)} arrays, {off} floats, {off * 4 / 1024:.0f} KB")


if __name__ == "__main__":
    main()

import subprocess

def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-3000:])

sh("mkdir -p /content/pw /content/runs && tar xzf /content/pw.tgz -C /content/pw && ls /content/pw")
sh("nvidia-smi -L")
sh("python -c 'import optax' 2>/dev/null || pip install -q optax")
sh("python -c 'import jax, optax; print(jax.__version__, optax.__version__, jax.devices())'")

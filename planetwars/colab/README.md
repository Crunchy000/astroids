# Training on a Colab GPU

Driven from a terminal with the `colab` CLI (`pip install google-colab-cli`). The repo is private, so
the code goes up as a tarball rather than a `git clone`.

```bash
cd planetwars
tar czf /tmp/pw.tgz --exclude='__pycache__' --exclude='entry/model.npz' pwjax entry colab
colab new -s pw --gpu L4
colab upload -s pw /tmp/pw.tgz /content/pw.tgz
colab exec -s pw < colab/setup.py                  # unpack, check the GPU, install optax if missing
RUN=bc1 MOD=pwjax.bc ARGS="--envs 2048 --iters 400" python colab/make_launch.py | colab exec -s pw
RUN=pw1 ARGS="--init /content/runs/bc1/latest.npz --vf-warmup 20" python colab/make_launch.py | colab exec -s pw
colab exec -s pw < colab/status.py                 # tail of every run's log
colab download -s pw /content/runs/pw1/latest.npz runs/pw1/latest.npz
colab stop -s pw                                    # stops billing
```

Training writes `log.jsonl`, `latest.npz` (weights in the entry's format) and `ckpt.pkl` (resume
with `--resume`) under `/content/runs/RUN/`.

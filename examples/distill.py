"""Distil a trained team model into an attention-over-entities model.

    python examples/distill.py --teacher results/population_best.zip --out results/entity_student.zip

Training an attention policy from scratch would throw away everything the
teacher learned, so the student first learns to copy it: the teacher plays
battles against a varied set of opponents (bots, the bot styles, itself, older
league models), every living teacher ship's observation is recorded with the
teacher's action probabilities, and the student's actor is trained to match them
(KL divergence per action head). Then ``--dagger`` rounds let the *student* fly
and the teacher label what it should have done in the student's own situations,
which fixes mistakes that compound when the student drifts off the teacher's
path. The student's critic is the teacher's (converted to a centralised critic),
so PPO can continue from it straight away, e.g. as a ``train_population.py``
founder.
"""

import argparse
import multiprocessing as mp
import os
import tempfile
import time

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

from league import Battle, NumpyPolicy, battle_over, export_policy, make_rules, play_match  # noqa: E402

OPPONENTS = ["bots", "bots:rusher", "bots:sniper", "bots:zone", "self", "results/league_striker.zip"]


def record(job):
    """One battle: ``pilot`` flies one side (sampling its actions), ``opp`` the other. Every
    2nd step, the living pilot ships' observations and the teacher's logits for them."""
    pilot, teacher, opp, seed = job
    cfg, oc = make_rules("competitive")
    rng = np.random.default_rng(seed)
    b, side = Battle(seed, cfg), seed % 2
    pol, teach = NumpyPolicy(pilot), NumpyPolicy(teacher)
    obs_out, logit_out = [], []
    while not battle_over(b)[0]:
        o = b.obs(side, oc)
        acts = {**b.team_actions(side, pilot, oc, actions=pol(o, rng))}
        if opp == "self":
            oo = b.obs(1 - side, oc)
            acts |= b.team_actions(1 - side, pilot, oc, actions=pol(oo, rng))
            o = np.concatenate([o, oo])
        else:
            acts |= b.team_actions(1 - side, opp, oc, rng=rng)
        if b.steps % 2 == 0:
            alive = o[:, 0] > 0.5
            obs_out.append(o[alive])
            logit_out.append(teach.logits(o[alive]))
        b.step(acts)
    return np.concatenate(obs_out).astype(np.float32), np.concatenate(logit_out).astype(np.float32)


def collect(pilot, teacher, opponents, battles, seed, workers):
    jobs = [(pilot, teacher, opponents[i % len(opponents)], seed + i) for i in range(battles)]
    ctx = mp.get_context("forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn")
    with ctx.Pool(workers) as pool:
        out = pool.map(record, jobs, chunksize=1)
    return np.concatenate([o for o, _ in out]), np.concatenate([lg for _, lg in out])


def fit(policy, obs, teacher_logits, epochs, lr, batch=1024, seed=0):
    """Train the actor to match the teacher's per-head action distributions; returns held-out stats."""
    import torch
    import torch.nn.functional as F

    from astroids_gym import ACTION_SPACE
    heads = [int(n) for n in ACTION_SPACE.nvec]
    g = torch.Generator().manual_seed(seed)
    n = len(obs)
    perm = torch.randperm(n, generator=g)
    val, tr = perm[: n // 20], perm[n // 20:]
    x, y = torch.as_tensor(obs), torch.as_tensor(teacher_logits)
    params = [p for name, p in policy.named_parameters()
              if name.startswith(("pi_features_extractor.", "mlp_extractor.policy_net.", "action_net."))]
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * (len(tr) // batch + 1))

    def loss_and_agree(idx):
        lat = policy.mlp_extractor.forward_actor(policy.pi_features_extractor(x[idx]))
        out = policy.action_net(lat)
        loss, agree = 0.0, []
        for s, t in zip(out.split(heads, 1), y[idx].split(heads, 1)):
            loss = loss + F.kl_div(F.log_softmax(s, 1), F.log_softmax(t, 1), log_target=True, reduction="batchmean")
            agree.append((s.argmax(1) == t.argmax(1)).float().mean().item())
        return loss, agree

    for ep in range(epochs):
        policy.train()
        order = tr[torch.randperm(len(tr), generator=g)]
        for i in range(0, len(order), batch):
            loss, _ = loss_and_agree(order[i:i + batch])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
        policy.eval()
        with torch.no_grad():
            vl, agree = loss_and_agree(val)
        print(f"  epoch {ep + 1}/{epochs}: held-out KL {vl.item():.4f}, argmax agreement "
              f"turn {agree[0]:.0%} thrust {agree[1]:.0%} fire {agree[2]:.0%}", flush=True)
    return vl.item(), agree


def head_to_head(a, b, battles, workers, seed=700_000):
    jobs = [((a, b, seed + k // 2, "competitive") if k % 2 == 0 else (b, a, seed + k // 2, "competitive"))
            for k in range(battles)]
    ctx = mp.get_context("forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn")
    with ctx.Pool(workers) as pool:
        res = pool.map(play_match, jobs, chunksize=1)
    return float(np.mean([((o if x == a else -o) + 1) / 2 for x, _, _, _, o in res]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="results/population_best.zip")
    p.add_argument("--out", default="results/entity_student.zip")
    p.add_argument("--battles", type=int, default=48, help="teacher battles to record")
    p.add_argument("--dagger", type=int, default=1, help="rounds of student battles labelled by the teacher")
    p.add_argument("--dagger-battles", type=int, default=24)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--d", type=int, default=64, help="token width")
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--threads", type=int, default=2, help="torch threads for fitting")
    p.add_argument("--eval-battles", type=int, default=40)
    args = p.parse_args()

    import torch
    from stable_baselines3 import PPO

    from astroids_gym.policies import EntityPolicy
    from train_league import LeagueVecEnv, grow_model, net_widths

    torch.set_num_threads(args.threads)
    tmp = tempfile.mkdtemp()
    teacher_zip = PPO.load(args.teacher, device="cpu")
    teacher = export_policy(teacher_zip, os.path.join(tmp, "teacher.npz"))
    opponents = [o if o.startswith("bots") or o == "self" else export_policy(PPO.load(o, device="cpu"),
                 os.path.join(tmp, os.path.basename(o) + ".npz")) for o in OPPONENTS if o.startswith(("bots", "self"))
                 or os.path.exists(o)]

    # The student: entity actor; critic = the teacher's, converted to a centralised critic.
    venv = LeagueVecEnv(1, 1, seed=0, rules="competitive", central=True)
    widths = net_widths(teacher_zip)
    hyper = {"n_steps": 64, "batch_size": 80, "learning_rate": 1e-4, "ent_coef": 0.003}
    critic = grow_model(teacher_zip, venv, hyper, 0)
    student = PPO(EntityPolicy, venv, n_epochs=critic.n_epochs, gamma=critic.gamma, gae_lambda=critic.gae_lambda,
                  clip_range=0.2, device="cpu", verbose=0, seed=0, **hyper,
                  policy_kwargs={"net_arch": widths, "n_actor_inputs": venv.n_actor_inputs,
                                 "entity_kwargs": {"d": args.d, "heads": args.heads}})
    csd, ssd = critic.policy.state_dict(), student.policy.state_dict()
    ssd |= {k: v.clone() for k, v in csd.items() if k.startswith(("mlp_extractor.value_net.", "value_net."))}
    student.policy.load_state_dict(ssd)
    n_params = sum(p.numel() for n, p in student.policy.named_parameters()
                   if n.startswith(("pi_features_extractor.", "mlp_extractor.policy_net.", "action_net.")))
    print(f"student actor: {n_params:,} parameters (teacher actor: "
          f"{sum(w.size + b.size for w, b in NumpyPolicy(teacher).layers) + NumpyPolicy(teacher).out[0].size:,})")

    t = time.time()
    obs, logits = collect(teacher, teacher, opponents, args.battles, 1, args.workers)
    print(f"recorded {len(obs):,} teacher decisions in {time.time() - t:.0f}s", flush=True)
    fit(student.policy, obs, logits, args.epochs, args.lr)
    for r in range(args.dagger):
        path = export_policy(student, os.path.join(tmp, f"student_{r}.npz"))
        o2, l2 = collect(path, teacher, opponents, args.dagger_battles, 10_000 * (r + 1), args.workers)
        obs, logits = np.concatenate([obs, o2]), np.concatenate([logits, l2])
        print(f"dagger round {r + 1}: +{len(o2):,} student situations labelled by the teacher", flush=True)
        fit(student.policy, obs, logits, max(2, args.epochs // 2), args.lr / 3, seed=r + 1)

    student.save(args.out)
    path = export_policy(student, os.path.splitext(args.out)[0] + ".npz")
    for opp in (teacher, "bots"):
        w = head_to_head(path, opp, args.eval_battles, args.workers)
        print(f"student vs {'teacher' if opp == teacher else opp}: wins {w:.0%} of {args.eval_battles}", flush=True)
    venv.close()


if __name__ == "__main__":
    main()

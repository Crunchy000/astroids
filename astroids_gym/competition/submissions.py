"""Evaluate an entry: against the baseline teams, or straight from a submission issue.

A submission is a GitHub issue whose body has a YAML block, as for Planet Wars RTS::

    ```yaml
    id: my-team
    repo_url: https://github.com/<owner>/<repo>/commit/<sha>
    commit: <sha>   # optional when the URL names it
    ```

``evaluate_submission`` fetches that commit, builds its ``Dockerfile``, runs the
container with no internet access, plays real-time matches against every
baseline and writes a Markdown report (``.github/workflows/evaluate-entry.yml``
posts it on the issue).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import traceback
from typing import Callable

from .league import head_to_head, match_key, play_job, summarise

MODELS = os.environ.get("ASTORIDS_MODELS",
                        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                     "competition", "models"))


def baselines() -> dict[str, str]:
    """The teams every entry is scored against: bot styles and earlier champions."""
    m = lambda name: os.path.join(MODELS, name + ".npz")  # noqa: E731
    return {
        "bots": "bots",
        "bots:fingerfour": "bots:fingerfour",
        "bots:pincer": "bots:pincer",
        "league striker": m("league_striker"),
        "population m16": m("population_m16"),
        "m10": m("m10"),
    }


def evaluate(spec: str, name: str = "entry", opponents: dict[str, str] | None = None, games: int = 10,
             seed: int = 900_000, realtime: bool = True, progress: Callable[[str], None] | None = print) -> list[dict]:
    """``games`` matches against each opponent (half on each side, paired seeds) -> results."""
    opponents = opponents or baselines()
    specs = {name: spec, **opponents}
    jobs = []
    for k, opp in enumerate(opponents):
        for n in range(games):
            s = seed + 1000 * k + n // 2
            jobs.append((name, opp, s) if n % 2 == 0 else (opp, name, s))
    results = []
    for i, (b, r, s) in enumerate(jobs, 1):
        res = play_job((b, r, s, specs, realtime, False))
        results.append(res)
        if progress:
            progress(f"[{i}/{len(jobs)}] {b} {res['score']['blue']:.1f} - {res['score']['red']:.1f} {r}: "
                     f"{res['winner']}")
    assert all(res["key"] == match_key(b, r, s) for res, (b, r, s) in zip(results, jobs))
    return results


def report_markdown(results: list[dict], name: str, header: str = "") -> str:
    """Per-opponent table for ``name``: won-drawn-lost, win rate, margin, decision time, late steps."""
    h2h = head_to_head(results)
    rows = []
    for opp in dict.fromkeys(n for res in results for n in res["names"].values() if n != name):
        mine = [res for res in results if opp in res["names"].values()]
        s = summarise(mine)[name]
        forfeit = f", {s['forfeits']} forfeited" if s["forfeits"] else ""
        rows.append(f"| {opp} | {s['won']}-{s['drawn']}-{s['lost']}{forfeit} | {100 * h2h[name, opp]:.0f}% | "
                    f"{s['margin']:+.1f} | {s['decision_ms']:.1f} | {100 * s['late']:.1f}% |")
    total = summarise(results)[name]
    lines = [header, ""] if header else []
    lines += ["| Opponent | W-D-L | Win rate | Margin | Decision ms | Late |", "|---|---|---:|---:|---:|---:|", *rows, "",
              f"**Overall: {100 * total['win_rate']:.0f}% win rate over {total['matches']} matches** "
              f"(average margin {total['margin']:+.1f}, mean decision time {total['decision_ms']:.1f} ms, "
              f"{100 * total['late']:.1f}% of steps late)."]
    return "\n".join(lines) + "\n"


# ---- submissions --------------------------------------------------------------------------
_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,47}$")
_REPO = re.compile(r"^https://github\.com/([\w.-]+)/([\w.-]+?)(?:\.git)?(?:/(?:commit|tree)/([0-9a-fA-F]{7,40}))?/?$")
_SHA = re.compile(r"^[0-9a-fA-F]{7,40}$")


def parse_issue(body: str) -> dict:
    """The submission in an issue body -> {"id", "repo", "commit"} (raises ``ValueError`` with the reason)."""
    m = re.search(r"```ya?ml[^\n]*\n(.*?)```", body or "", re.S)
    if not m:
        raise ValueError("no ```yaml block found in the issue")
    fields = {}
    for line in m.group(1).splitlines():
        line = re.sub(r"\s+#.*$", "", line).strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[key.strip()] = value.strip().strip("'\"")
    entry_id = fields.get("id", "").lower()
    if not _ID.match(entry_id):
        raise ValueError("`id` must be 1-48 characters: lower-case letters, digits, '-' or '_'")
    rm = _REPO.match(fields.get("repo_url", ""))
    if not rm:
        raise ValueError("`repo_url` must be a https://github.com/<owner>/<repo>/commit/<sha> link")
    commit = fields.get("commit") or rm.group(3) or ""
    if not _SHA.match(commit):
        raise ValueError("name the commit to evaluate, in `repo_url` (/commit/<sha>) or in `commit`")
    return {"id": entry_id, "repo": f"https://github.com/{rm.group(1)}/{rm.group(2)}", "commit": commit.lower()}


def fetch(repo: str, commit: str, dest: str) -> str:
    """Check out ``commit`` of ``repo`` into ``dest``; returns the full commit hash."""
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    git = lambda *a, **kw: subprocess.run(["git", *a], check=True, env=env, capture_output=True, text=True,  # noqa: E731
                                          timeout=600, **kw)
    os.makedirs(dest, exist_ok=True)
    git("init", "-q", dest)
    try:
        git("-C", dest, "fetch", "-q", "--depth", "1", repo, commit)
        git("-C", dest, "checkout", "-q", "--detach", "FETCH_HEAD")
    except subprocess.CalledProcessError:  # a short hash can't be fetched directly
        shutil.rmtree(dest)
        git("clone", "-q", "--filter=blob:none", repo, dest)
        git("-C", dest, "checkout", "-q", "--detach", commit)
    return git("-C", dest, "rev-parse", "HEAD").stdout.strip()


def evaluate_submission(body: str, games: int = 10, workdir: str | None = None,
                        progress: Callable[[str], None] | None = print) -> tuple[str, list[dict]]:
    """Parse, fetch, build, run and evaluate the entry in an issue body -> (Markdown report, results)."""
    from . import docker
    try:
        entry = parse_issue(body)
    except ValueError as e:
        return f"### ❌ Could not read the submission\n\n{e}.\n", []
    tmp = workdir or tempfile.mkdtemp(prefix="astorids-entry-")
    src = os.path.join(tmp, entry["id"])
    title = f"### Evaluation of `{entry['id']}`"
    try:
        sha = fetch(entry["repo"], entry["commit"], src)
        if not os.path.exists(os.path.join(src, "Dockerfile")):
            return f"{title}\n\n❌ No `Dockerfile` at the top of {entry['repo']} at `{sha[:12]}`.\n", []
        if progress:
            progress(f"building {entry['repo']} at {sha[:12]}")
        tag = docker.build(src, f"astorids-entry-{entry['id']}:{sha[:12]}")
        with docker.Container(tag, internal=True) as c:
            if progress:
                progress(f"entry up at {c.url}")
            results = evaluate(c.url, entry["id"], games=games, progress=progress)
        header = (f"{title}\n\n✅ Built {entry['repo']} at `{sha[:12]}`; it served its agent on port 8080 with no "
                  f"internet access. {len(results)} real-time matches (33 ms per decision, random match settings), "
                  f"{games} against each baseline, half on each side.")
        return report_markdown(results, entry["id"], header), results
    except Exception as e:  # noqa: BLE001 - reported on the issue
        detail = "".join(traceback.format_exception_only(type(e), e))[-6000:]
        return f"{title}\n\n❌ The evaluation failed:\n\n```\n{detail}\n```\n", []
    finally:
        if workdir is None:
            shutil.rmtree(tmp, ignore_errors=True)

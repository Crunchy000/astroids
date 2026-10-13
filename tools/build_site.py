"""Build the static site: the browser game, the video gallery and the docs.

    pip install markdown
    python tools/build_site.py --out public

The game (index.html), the gallery (videos.html) and results/ are copied unchanged, apart from one
"Docs" link in the gallery's menu. The READMEs become pages under docs/, with their relative links
rewritten so they work on the site (links to code point at the repository instead).
"""
import argparse
import html
import posixpath
import re
import shutil
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "https://github.com/Crunchy000/astroids/blob/ccr-ccbf1ca0-erm5dw/"

# (markdown file, page name, card title, card text for the docs index)
DOCS = [
    ("README.md", "asteroids", "The game", "Rules, the Python port (Gymnasium and PettingZoo), training and results."),
    ("competition/README.md", "competition", "Competition", "How to enter a team in the Planet Wars-style competition."),
    ("competition/leaderboard.md", "leaderboard", "League leaderboard", "The first league: 12 entries, 660 real-time matches."),
    ("planetwars/README.md", "planetwars", "Planet Wars entry", "Our JAX-trained entry for the real Planet Wars RTS league."),
    ("planetwars/colab/README.md", "planetwars-colab", "Planet Wars on a Colab GPU", "Training the Planet Wars entry on a GPU runtime."),
]
MEDIA_EXT = {".mp4", ".png", ".jpg", ".json"}

CSS = """
:root { --bg:#05060a; --panel:#0d0f16; --line:#1d212c; --text:#e6e7ea; --text-2:#9aa0ac;
        --blue:#4cc9f0; --red:#ff7b54; --mono:ui-monospace,Menlo,Consolas,monospace; }
* { box-sizing:border-box; }
html, body { margin:0; background:var(--bg); color:var(--text); }
body { font:16px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif; }
a { color:var(--blue); }
header, main, footer { max-width:880px; margin:0 auto; padding:0 16px; }
header { padding-top:28px; padding-bottom:12px; display:flex; flex-wrap:wrap; gap:8px 24px;
         align-items:baseline; justify-content:space-between; border-bottom:1px solid var(--line); }
h1.site { font:700 20px/1.2 var(--mono); margin:0; }
h1.site a { color:var(--text); text-decoration:none; } h1.site .b { color:var(--blue); }
nav { font:14px var(--mono); display:flex; flex-wrap:wrap; gap:6px 18px; }
main { padding-top:12px; padding-bottom:40px; overflow-wrap:break-word; }
h1, h2, h3 { line-height:1.3; } h1 { font-size:28px; } h2 { font-size:21px; margin-top:2em; }
h3 { font-size:17px; margin-top:1.6em; }
p, li { max-width:75ch; }
code { font:0.9em var(--mono); background:var(--panel); padding:1px 5px; border-radius:4px; overflow-wrap:anywhere; }
pre { background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:12px 14px;
      overflow-x:auto; } pre code { background:none; padding:0; font-size:13.5px; overflow-wrap:normal; }
.table-wrap { overflow-x:auto; margin:16px 0; }
table { border-collapse:collapse; font:13.5px var(--mono); }
th, td { padding:6px 12px; border-bottom:1px solid var(--line); white-space:nowrap; text-align:left; }
th { color:var(--text-2); }
img, video { max-width:100%; height:auto; border-radius:6px; }
video { width:100%; background:#000; }
blockquote { margin:16px 0; padding:0 16px; border-left:3px solid var(--line); color:var(--text-2); }
.cards { display:grid; grid-template-columns:repeat(auto-fill,minmax(250px,1fr)); gap:14px; margin:20px 0 8px; }
.card { display:block; background:var(--panel); border:1px solid var(--line); border-radius:8px;
        padding:14px 16px; text-decoration:none; color:var(--text-2); }
.card:hover { border-color:var(--blue); } .card b { display:block; color:var(--text); margin-bottom:4px; }
figure { margin:20px 0; } figcaption { color:var(--text-2); font-size:14px; margin-top:6px; }
footer { color:var(--text-2); font-size:13px; padding-bottom:32px; }
"""

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="{description}">
<style>{css}</style>
</head>
<body>
<header>
  <h1 class="site"><a href="./"><span class="b">Asteroids 5v5</span> docs</a></h1>
  <nav><a href="../">Play the game</a><a href="../videos.html">Videos</a><a href="./">Docs</a><a href="{source}">Source</a></nav>
</header>
<main>
{body}
</main>
<footer>Pages are built from the READMEs in the repository.</footer>
</body>
</html>
"""


def rewrite_links(body, md_path, pages, media, source):
    """Make a README's relative links and images resolve on the site."""
    here = posixpath.dirname(md_path)

    def fix(m):
        attr, value = m.group(1), m.group(2)
        if re.match(r"([a-z][a-z0-9+.-]*:|#|//)", value):
            return m.group(0)
        path, _, frag = value.partition("#")
        target = posixpath.normpath(posixpath.join(here, path))
        if target in pages:
            new = pages[target] + ".html" + ("#" + frag if frag else "")
        elif target in ("index.html", "videos.html"):
            new = "../" + target
        elif target in media:
            new = "../" + media[target]
        elif (ROOT / target).is_dir():
            new = source.replace("blob/", "tree/", 1) + target
        elif (ROOT / target).exists():
            new = source + target
        else:
            print(f"warning: {md_path}: no such file {value}")
            return m.group(0)
        return f'{attr}="{new}"'

    return re.sub(r'\b(href|src)="([^"]*)"', fix, body)


def collect_media():
    """Repo path -> site path for every file the site serves next to the pages."""
    media = {}
    for p in sorted((ROOT / "results").iterdir()):
        if p.suffix in MEDIA_EXT:
            media[f"results/{p.name}"] = f"results/{p.name}"
    for p in sorted((ROOT / "planetwars/media").iterdir()):
        media[f"planetwars/media/{p.name}"] = f"media/{p.name}"
    return media


def render(md_file, pages, media, source):
    text = (ROOT / md_file).read_text()
    title = re.match(r"#\s+(.+)", text).group(1).strip()
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "toc", "sane_lists"])
    body = rewrite_links(body, md_file, pages, media, source)
    body = body.replace("<table>", '<div class="table-wrap"><table>').replace("</table>", "</table></div>")
    summary = re.search(r"<p>(.*?)</p>", body, re.S)
    description = html.escape(re.sub(r"<[^>]+>", "", summary.group(1)) if summary else title)[:200]
    return title, description, body


def index_page():
    cards = "\n".join(
        f'<a class="card" href="{name}.html"><b>{html.escape(card)}</b>{html.escape(blurb)}</a>'
        for _, name, card, blurb in DOCS)
    return f"""<h1>Docs and videos</h1>
<p>Two fleets of five triangle ships fight it out in an Asteroids-style arena. This site has the browser
game, the write-ups of how the agents are trained, and videos of the matches.</p>
<div class="cards">{cards}
<a class="card" href="../videos.html"><b>Match videos</b>Trained teams against the bots and each other, 1280×720.</a>
</div>
<h2>Planet Wars: our model against their greedy agent</h2>
<figure>
<video controls preload="metadata" poster="../media/game_vs_greedy.png" src="../media/game_vs_greedy.mp4"></video>
<figcaption>Blue is our model, orange is the sample Greedy Heuristic agent, on a 30-planet league map. Numbers are
ships; arrows are fleets in flight. <a href="planetwars.html">How it was trained</a>.</figcaption>
</figure>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="public")
    ap.add_argument("--source-url", default=SOURCE,
                    help="where links to code in the READMEs point (ends with the branch and a slash)")
    args = ap.parse_args()
    out = (ROOT / args.out).resolve()
    source = args.source_url if args.source_url.endswith("/") else args.source_url + "/"
    if out == ROOT or out in ROOT.parents:
        raise SystemExit(f"refusing to build into {out}")
    if out.exists():
        shutil.rmtree(out)
    (out / "docs").mkdir(parents=True)

    media = collect_media()
    for repo_path, site_path in media.items():
        (out / site_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / repo_path, out / site_path)
    shutil.copy(ROOT / "index.html", out / "index.html")
    gallery = (ROOT / "videos.html").read_text()
    assert "</nav>" in gallery
    (out / "videos.html").write_text(gallery.replace("</nav>", '<a href="docs/">Docs</a></nav>', 1))
    (out / ".nojekyll").touch()

    pages = {f: name for f, name, _, _ in DOCS}
    for md_file, name, _, _ in DOCS:
        title, description, body = render(md_file, pages, media, source)
        page = PAGE.format(title=html.escape(title), description=description, css=CSS, body=body,
                           source=html.escape(source.replace("blob/", "tree/", 1)))
        (out / "docs" / f"{name}.html").write_text(page)
    root_url = source.replace("blob/", "tree/", 1)
    (out / "docs" / "index.html").write_text(PAGE.format(
        title="Asteroids 5v5: docs and videos", description="Docs, training write-ups and match videos for the "
        "Asteroids 5v5 agents and our Planet Wars entry.", css=CSS, body=index_page(), source=html.escape(root_url)))

    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"built {out} ({size / 1e6:.0f} MB, {len(media)} media files, {len(DOCS) + 1} doc pages)")


if __name__ == "__main__":
    main()

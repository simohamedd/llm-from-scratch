"""LLM Radar — keeps this repo up to date with the LLM ecosystem.

Runs weekly in GitHub Actions (.github/workflows/llm-radar.yml):
  1. Fetches the latest release (or newest tag/commit) of every repo in watchlist.json
  2. Fetches the newest LLM papers from arXiv
  3. Rewrites docs/LLM_RADAR.md and radar/state.json
  4. Lists what changed since last week -> the workflow opens a Pull Request with it

Standard library only, so it runs anywhere:  python radar/llm_radar.py
"""
from __future__ import annotations

import datetime as dt
import json
import os
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WATCHLIST = ROOT / "radar" / "watchlist.json"
STATE = ROOT / "radar" / "state.json"
REPORT = ROOT / "docs" / "LLM_RADAR.md"
CHANGES = ROOT / "radar" / "CHANGES.md"


def _get(url: str, accept: str = "application/vnd.github+json"):
    headers = {"Accept": accept, "User-Agent": "llm-from-scratch-radar"}
    if "api.github.com" in url and os.getenv("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as r:
        return r.read()


def latest_version(repo: str) -> dict:
    base = f"https://api.github.com/repos/{repo}"
    try:
        rel = json.loads(_get(f"{base}/releases/latest"))
        return {"version": rel["tag_name"], "date": rel["published_at"][:10], "url": rel["html_url"]}
    except Exception:
        pass
    try:
        tags = json.loads(_get(f"{base}/tags?per_page=1"))
        if tags:
            return {"version": tags[0]["name"], "date": "", "url": f"https://github.com/{repo}/tags"}
        commit = json.loads(_get(f"{base}/commits?per_page=1"))[0]
        return {"version": commit["sha"][:7], "date": commit["commit"]["committer"]["date"][:10],
                "url": commit["html_url"]}
    except Exception as e:  # network/rate-limit: keep last known value
        return {"error": str(e)[:80]}


def latest_papers(query: str, n: int) -> list[dict]:
    url = ("http://export.arxiv.org/api/query?sortBy=submittedDate&sortOrder=descending&max_results="
           f"{n}&search_query={urllib.parse.quote(query)}")
    try:
        root = ET.fromstring(_get(url, accept="application/atom+xml"))
    except Exception:
        return []
    ns = {"a": "http://www.w3.org/2005/Atom"}
    return [{"title": " ".join(e.findtext("a:title", "", ns).split()),
             "url": e.findtext("a:id", "", ns), "date": e.findtext("a:published", "", ns)[:10]}
            for e in root.findall("a:entry", ns)]


def main() -> None:
    cfg = json.loads(WATCHLIST.read_text())
    old = json.loads(STATE.read_text()) if STATE.exists() else {"tools": {}}
    tools, changes = {}, []
    for t in cfg["tools"]:
        info = latest_version(t["repo"])
        prev = old["tools"].get(t["repo"], {})
        if "error" in info:
            info = prev or info
        elif prev.get("version") != info["version"]:
            changes.append(f"- **{t['repo']}** ({t['category']}): `{prev.get('version', 'new')}` → "
                           f"`{info['version']}` — [release notes]({info['url']}) · relates to chapter `{t['chapter']}`")
        tools[t["repo"]] = {**t, **info}

    papers = latest_papers(cfg["arxiv_query"], cfg["arxiv_max_results"])
    today = dt.date.today().isoformat()
    STATE.write_text(json.dumps({"updated": today, "tools": tools}, indent=2) + "\n")

    lines = [f"# 📡 LLM Radar\n\n_Auto-generated weekly by `radar/llm_radar.py` — last update **{today}**._\n",
             "Each tool is linked to the chapter where you build the same idea from scratch.\n",
             "| Tool | Category | Latest | Date | Chapter |", "|---|---|---|---|---|"]
    for repo, i in sorted(tools.items(), key=lambda kv: kv[1]["category"]):
        v = f"[{i.get('version', '?')}]({i['url']})" if i.get("url") else i.get("version", "?")
        lines.append(f"| [{repo}](https://github.com/{repo}) | {i['category']} | {v} | {i.get('date', '')} | `{i['chapter']}` |")
    if papers:
        lines += ["\n## 📄 Newest LLM papers on arXiv\n"]
        lines += [f"- {p['date']} — [{p['title']}]({p['url']})" for p in papers]
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n")
    CHANGES.write_text("## What changed this week\n\n" + ("\n".join(changes) if changes else "_No new releases._") + "\n")
    print(f"{len(tools)} tools, {len(changes)} updated, {len(papers)} papers")


if __name__ == "__main__":
    main()

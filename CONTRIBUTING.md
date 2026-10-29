# Contributing & branching model

```
main  ◄──── release PR ──── develop ◄──── feature/*   (new techniques)
                              ▲  ▲
                              │  └──── radar/weekly-update   (bot, every Monday)
                              └─────── fix/*
chapter/01 → 02 → … → 07      (frozen course snapshots, updated only when a chapter changes)
```

1. Branch from `develop`: `git checkout develop && git checkout -b feature/<name>`
2. Keep each technique **from scratch** and small (≈100-200 lines), with a test in `tests/` and a doc in `docs/`.
3. Open a PR into `develop`. CI must be green.
4. `develop` → `main` when a set of features is stable.

## When the LLM Radar opens a PR
1. Read the release notes it links.
2. Ask: *does this introduce a new idea worth teaching?* (e.g. a new attention variant, a new alignment method, a new quantization format).
3. If yes → open an issue with the "Track a new LLM tool / technique" template and a `feature/*` branch.
4. Merge the Radar PR to refresh `docs/LLM_RADAR.md`.

## One-time GitHub setting
For the Radar bot to open PRs: **Settings → Actions → General → Workflow permissions →
"Read and write permissions"** and tick **"Allow GitHub Actions to create and approve pull requests"**.

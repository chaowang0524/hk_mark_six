# Project Status

Living dev notes — current state, known blockers, and next steps. Updated as work
progresses (unlike `CLAUDE.md`, which holds stable architecture/instructions, and
`PLAN.md`, which is the original design doc). Keep the "as of" date current.

## Deployment status (as of 2026-10-10)

A non-technical friend (Windows, no coding knowledge) needs to get predictions without
running any CLI. Current approach: a Streamlit web app (`app.py`) they open as a browser
link — no install, just click a button.

**Live at `http://47.108.222.33/`** — hosted on a dedicated Alibaba Cloud VPS, SSH alias
`marksix` (`ssh marksix`, user `admin`, passwordless sudo). Ubuntu 24.04, 2 vCPU, 1.6GB RAM,
region Asia/Shanghai (mainland China).

Server setup:
- Repo cloned to `~/lottery`, venv at `~/lottery/.venv`, deps installed via plain `pip`
  (this box has real pip, unlike the local uv-managed `.venv`).
- `app.py` runs as systemd service `marksix-streamlit` (port 8501, loopback only,
  `Restart=on-failure`, enabled at boot). Unit file: `/etc/systemd/system/marksix-streamlit.service`.
- nginx reverse-proxies port 80 → 127.0.0.1:8501 (config: `/etc/nginx/sites-available/marksix`,
  `proxy_read_timeout 600s` to survive the slow retrain path).
- ufw firewall active: only 22 (SSH) and 80 (HTTP) open.
- Models (`extra_xgb.joblib`, `main_xgb.joblib`) were trained **on the server** against a
  snapshot of the local machine's data — `raw_draws.csv` was scp'd over (small, fine), but
  `features.csv` (~25MB) was too slow over the link, so it was rebuilt on the server instead
  via `python main.py preprocess` from the transferred `raw_draws.csv`. Full `train --model
  xgb --full` took ~20 min on this box's 2 cores (vs. ~5 min locally) — don't assume local
  timings apply there.

**Known blocker — scraping is broken, decision deferred:**
- The scraper's data source (`pilio.idv.tw`, a Taiwan domain) gets its TLS connection
  actively reset from the `marksix` server — the Great Firewall blocking it, not a config
  issue. Separately, scraping isn't currently working from the local Mac either (unrelated
  reason, not yet diagnosed).
- Decision (explicit from user): don't fix this now — ship with the current static data
  snapshot, revisit later. Likely direction floated by the user: a manual raw-results input
  feature instead of relying on the scraper, but this is not designed or built yet.
- Consequence: `app.py`'s "🔄 Refresh data & retrain" button calls `scrape()` and will error
  out if clicked, since scraping doesn't work anywhere right now. Not yet resolved whether to
  disable/hide that button in the meantime — asked the user, no answer yet.

**Next steps when picking this back up:**
1. Decide on the manual-input (or alternative) fix for data refresh.
2. Decide what `app.py` should do about the currently-broken retrain button.
3. Nothing else is blocking — hosting, service management, and firewall are done and stable.

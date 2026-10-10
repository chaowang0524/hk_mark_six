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

**Scraper is still broken — worked around, not fixed:**
- The scraper's data source (`pilio.idv.tw`, a Taiwan domain) gets its TLS connection
  actively reset from the `marksix` server — the Great Firewall blocking it, not a config
  issue. Separately, scraping isn't currently working from the local Mac either (unrelated
  reason, not diagnosed). `src/scraper.py` itself was not touched/fixed.
- Decision (explicit from user): don't fix the scraper now — build manual data entry instead.

**Manual data entry — shipped and live (2026-10-10):**
- New "📋 Data" page in `app.py`, password-gated (`st.text_input` + `st.secrets`).
  Password is `9527` (placeholder — change it before giving the link to anyone you don't
  fully trust), read from `.streamlit/secrets.toml`, gitignored, present both locally and
  on the server at `~/lottery/.streamlit/secrets.toml` (`chmod 600`).
- Shows the last 10 draws, a form to add a new one (validated: 6 distinct numbers 1–49,
  extra ≠ any main, date after the last draw; draw ID like `26/109` auto-assigned), and a
  two-step confirm to delete the most recent entry (undo for mis-entry).
- Logic lives in `src/data_entry.py` (`read_tail`/`add_draw`/`delete_last`), reusing
  `scraper.py`'s `Draw`/ID-assignment code so hand-entered rows are indistinguishable in
  format from scraped ones. Covered by `tests/test_data_entry.py` (10 tests).
- The "🔁 Retrain from current data" button (renamed from "🔄 Refresh data & retrain") no
  longer calls `scrape()` — it just rebuilds features and retrains from whatever is
  currently in `raw_draws.csv`. Resolves the previously-open "disable the button?" question.
- Deployed: pushed as commits `7e29e2a`/`6f35c97`, pulled on the server, service restarted,
  health check and manual click-through both confirmed working.

**Next steps when picking this back up:**
1. Nothing blocking — manual entry is now the live data-acquisition path going forward.
2. Optional/future: revisit the scraper (GFW block on the server, undiagnosed issue locally)
   if automatic updates become worth the effort again.
3. Minor housekeeping, not urgent: commit `6f35c97` accidentally bundled in an unrelated
   change to `十次试验.txt` (a personal notes file) alongside the Data-page feature — content
   is intact, just filed under the wrong commit message. No action taken; flagged to the user.

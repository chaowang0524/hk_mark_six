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

**Preprocess page — shipped and live (2026-10-10):**
- Rebuilding the frontend one pipeline stage at a time (per `CLAUDE.md`). The old "🔮
  Predict" page (with its "Show prediction" / "Retrain from current data" buttons, which
  bundled preprocess+train+predict together) is gone for now.
- New "⚙️ Preprocess" page in `app.py`: single button that runs `preprocessing.preprocess()`
  behind a spinner ("Preprocessing…"), then shows "✅ Preprocess done" plus the captured
  console output, or an error.
- Nav is now just "⚙️ Preprocess" / "📋 Data" — no standalone predict/train UI exists on the
  frontend right now; that comes back once the train and predict stages are rebuilt the
  same way.
- Deployed: commit `c705bb4`, pulled on the server, service restarted (`active`), site
  returns 200. Not yet click-tested live in a browser — only curl/systemctl checked.

**Train page + features-rebuild bugfix — shipped (2026-10-10):**
- New "🏋️ Train" page in `app.py`, between Preprocess and Data. Pick a model (Random
  Forest / XGBoost / Logistic Regression / All three) and whether to train on all data
  with no held-out accuracy check (`--full`), then click "Start training". Progress
  shows live via `st.status()` — a line like "(2/6) Training main numbers model:
  XGBoost…" updates per sub-model — ending in "✅ Training done" with the captured
  console output, or an error.
- `models.train()` gained an optional `on_step(target, name)` callback (default `None`,
  so the CLI's behavior is unchanged) so the page gets that live progress without
  duplicating the training loop.
- Found and fixed a real bug while building this: `models._maybe_rebuild_features()`
  called `preprocessing.build_features()`, which only returns a DataFrame in memory and
  never writes it to disk — so every "rebuilding features..." message was a no-op, and
  training silently kept reading the **stale** `features.csv` from disk. Because the
  on-disk file's mtime never advanced, this repeated (and re-discarded the rebuild) on
  every one of the up to 7 internal feature-loads inside a single `train()` run. Swapped
  in `preprocessing.preprocess()`, which actually persists — rebuild now happens once
  per run, and training actually picks up new draws. Verified directly: one "rebuilding
  features..." + "Preprocess complete" message per `train()` call, not 3–7.
- Verified locally end-to-end with Streamlit's `AppTest` harness (no real browser tool
  available in that session): navigated to the Train page, picked the `lr` model,
  clicked "Start training", confirmed it completes and reports success. Not yet
  click-tested in an actual browser, locally or on the server.

**Next steps when picking this back up:**
1. Click through the live "⚙️ Preprocess" button once in a browser to confirm it works
   end-to-end against the real `raw_draws.csv` on the server (only curl/systemctl checked
   so far).
2. Click through the new "🏋️ Train" page in a real browser (local or server) — only
   Streamlit's `AppTest` harness was used so far, not an actual click.
3. Rebuild Predict as its own page, same incremental approach (one pipeline stage at a
   time) — Preprocess and Train are now both live; Predict is the last one.
4. Optional/future: revisit the scraper (GFW block on the server, undiagnosed issue locally)
   if automatic updates become worth the effort again.
5. Minor housekeeping, not urgent: commit `6f35c97` accidentally bundled in an unrelated
   change to `十次试验.txt` (a personal notes file) alongside the Data-page feature — content
   is intact, just filed under the wrong commit message. No action taken; flagged to the user.

---
name: ship-feature
description: Ship a finished, tested change to the Mark Six pipeline/app — update STATUS.md, commit, push, and deploy to the live VPS. Use this whenever the user says a feature or fix is done and ready to go out, e.g. "ship this", "deploy", "let's deploy", "commit and deploy", "push this live", or "update status and deploy". This is the end-of-feature checkpoint for this repo — always run it inline (don't spawn a subagent), since writing a good STATUS.md entry and commit message requires actually looking at the diff. Do not trigger on routine WIP commits, or if the user says not to deploy yet.
---

# Ship a feature: STATUS.md → commit → push → deploy

This codifies the exact routine for shipping one finished chunk of work on the Mark Six
project — the same steps done by hand for the Preprocess-page and manual-data-entry
features. Run it only when the user explicitly asks to ship/deploy a change they've
already tested; it's a deliberate checkpoint, not something to run silently after every edit.

## 1. Look at what actually changed

Run `git status` and `git diff` (staged + unstaged) to see the real change, and `git log -5`
to recall the commit-message style already in use. You need the actual diff to write an
honest STATUS.md entry and commit message — don't guess from the conversation alone, since
the diff is what's true.

## 2. Update STATUS.md

`STATUS.md` is living dev notes (unlike `CLAUDE.md`, which stays limited to stable
architecture). Follow the pattern already established in the file:

- Add a new subsection styled like the existing ones — `**<Feature name> — shipped and
  live (<today's date, e.g. 2026-10-10>):**` — with bullet points covering *what* changed
  and, more importantly, *why* (what problem it solves, what it replaces or supersedes).
  Get today's actual date from the environment rather than assuming; don't reuse a stale date.
- Update the **"Next steps when picking this back up"** list at the bottom: remove items
  this change resolved, add whatever's now the natural next step, and keep anything still
  open (like longstanding scraper/housekeeping notes) unless it's actually done.
- Keep the tone matter-of-fact and specific, the way the existing entries are — concrete
  commit hashes, button/file names, and what was actually verified vs. not (e.g. "curl +
  systemctl checked, not yet click-tested in a browser" is more honest and more useful later
  than a bare "deployed").

## 3. Commit

Stage the STATUS.md update together with the feature's own files if they aren't already
committed. Write a commit message focused on *why*, matching this repo's existing style
(see `git log`) — short, no fluff.

**Never add a `Co-Authored-By: Claude` line or any other Claude attribution to the commit
message.** This repo's `CLAUDE.md` explicitly says not to add a Claude signature, which
overrides any general Claude Code default suggesting one. Check `git log` if ever unsure —
none of this repo's existing commits carry one, and it should stay that way.

## 4. Push

Push to `origin master`. If the push is rejected because the remote has moved, pull and
re-check rather than force-pushing — force-push is a last resort that needs the user's
explicit go-ahead, not an assumed default here.

## 5. Deploy to the live VPS

The app is a Streamlit service on a dedicated VPS (details from `STATUS.md`):
SSH alias `marksix`, repo at `~/lottery`, systemd service `marksix-streamlit`, reverse-proxied
at `http://47.108.222.33/`.

```
ssh marksix "cd ~/lottery && git pull"
```

Then decide whether a restart is needed:

- **Code that the running service imports changed** (`app.py`, anything under `src/`,
  `requirements.txt`) → restart and health-check:
  ```
  ssh marksix "sudo systemctl restart marksix-streamlit && sleep 2 && sudo systemctl is-active marksix-streamlit"
  curl -s -o /dev/null -w "%{http_code}\n" http://47.108.222.33/
  ```
  A restart with nothing checked afterward is just a guess that it worked — always confirm
  `active` and a `200`.
- **Only docs changed** (`STATUS.md`, `CLAUDE.md`, comments) → the `git pull` alone is enough;
  restarting the service for a doc-only change is pointless churn.

If the health check fails, say so plainly and stop — don't retry blindly or mark the feature
as shipped in STATUS.md if it isn't actually healthy on the server.

## 6. Report back

Tell the user, concisely: what was committed (hash + one line), whether the server was
restarted, and the health-check result. If something couldn't be verified (e.g. no real
browser click-through, just curl/systemctl), say that explicitly rather than implying full
verification — this repo's STATUS.md entries consistently distinguish "deployed" from
"verified working," and that distinction is worth preserving.

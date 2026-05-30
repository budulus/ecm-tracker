---
name: commit-push
description: >-
  Commit the working tree and push the ECM Tracker repo to its self-hosted Forgejo remote.
  Use whenever the user says "commit and push", "commit and push all", "push the changes",
  "push to origin", "push it", or otherwise asks to land work on the remote. This repo's remote
  is a Forgejo instance reachable only over Tailscale and authenticated through Git Credential
  Manager, so pushes fail in ways a generic "git push" doesn't anticipate — this skill knows the
  required co-author footer, runs the headless tests as a pre-commit gate, and diagnoses the
  Tailscale-vs-credential failure mode that has stalled pushes here before.
---

# Commit & push (ECM Tracker → Forgejo)

This repo is a solo project that lands work **directly on `main`**, and its only remote is a
self-hosted Forgejo server. The non-obvious part isn't the commit — it's that the push can fail
for reasons specific to this setup. This skill captures the whole flow so "commit and push" is
reliable instead of improvised.

## Repo facts you can rely on

- **Remote:** `origin` → `http://placksiserver.tail87cfa8.ts.net:3000/rhopf/ecm_tracker.git`
  The host is a **Tailscale** address (`*.ts.net`). If Tailscale isn't connected on this machine,
  the host is simply unreachable — which often surfaces as a confusing connection/timeout error
  rather than an obvious "VPN is down" message.
- **Auth:** Git Credential Manager (`credential.helper manager`, `provider generic`), TLS via
  `schannel`. GCM prompts for credentials **interactively**, and this tool runs git
  non-interactively — so when stored credentials are missing or expired, an automated `git push`
  cannot satisfy the prompt and will fail.
- **Branch:** history is entirely direct-to-`main`. Commit to `main` unless the user says
  otherwise. (The global "branch first on the default branch" default is overridden by this repo's
  established solo workflow.)
- **Tests (pre-commit gate):** run from the project root in PowerShell —
  `$env:QT_QPA_PLATFORM = "offscreen"; uv run python -m tests.test_pipeline`
  `$env:QT_QPA_PLATFORM = "offscreen"; uv run python -m tests.test_plugins`

## Procedure

1. **Understand the change.** Run `git status` and `git diff` (and `git diff --staged`). Summarize
   what actually changed so the commit message describes the work, not just "update files".

2. **Gate on tests when source changed.** If the diff touches Python under `app/`, `plugins/`, or
   `tests/`, run both offscreen test suites above and confirm they pass before committing — this is
   the only remote, so don't push a broken tree. If the change is docs-only (e.g. `*.md`,
   `app/gui/help.html`), or the user has already validated this session, you may skip the gate; say
   that you skipped it and why.

3. **Commit to `main`.** Stage the relevant files and write a concise, specific message. Match the
   existing history's style (a short imperative summary line, e.g. `Add ROI shape tools (...)`).
   End every commit message with the required footer:

   ```
   Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
   ```

   Make one commit unless the changes are logically separable into clean atomic commits — don't
   manufacture splits that don't reflect the work.

4. **Push.** `git push origin main`.

5. **If the push fails, diagnose before retrying** (this is the part that bit S67):
   - **Connection / could-not-resolve / timeout** → the Tailscale host is unreachable. Tell the
     user to confirm Tailscale is connected (the remote is `*.ts.net`). Don't keep retrying a push
     to an unreachable host.
   - **401 / authentication failed / credentials rejected** → GCM credentials are missing or
     expired, and this tool can't answer GCM's interactive prompt. Ask the user to run the push
     themselves so GCM can prompt:

     ```
     ! git push origin main
     ```

     (The `!` prefix runs it in-session so the output lands here.) If credentials are stale rather
     than absent, they can clear the entry for `placksiserver.tail87cfa8.ts.net` in Windows
     Credential Manager and let GCM re-prompt on the next push.

6. **Report honestly.** State exactly what landed: the commit hash + summary, and whether the push
   reached `origin/main`. If the push did **not** succeed, say so plainly and leave the commit in
   place (it's safe locally) — never report a push as done when it wasn't. A quick
   `git status -sb` confirms whether `main` is ahead of `origin/main`.

## Notes

- To land just the push without a new commit (e.g. a prior commit is still unpushed), skip
  steps 1–3 and go straight to step 4.
- If the user wants the push verified after they run it interactively, re-check with
  `git rev-parse @{u}..HEAD` (empty output = nothing unpushed).

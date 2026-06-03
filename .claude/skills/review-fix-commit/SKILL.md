---
name: review-fix-commit
description: >-
  Review the current changes, fix every finding one by one, validate with the headless offscreen
  tests, then commit. Use after finishing a feature or bugfix in the ECM Tracker repo, or whenever
  the user says "review and fix", "review and clean up", "fix all the findings", "fix all the
  issues one by one", "address the review comments", or asks to tidy up the diff before committing.
  This wraps the existing /code-review skill with the disciplined fix-and-verify loop this repo
  expects (numbered triage, surgical per-finding fixes, offscreen-test gate, one descriptive
  commit) so a review doesn't end as a pile of unaddressed comments.
---

# Review → fix → commit (ECM Tracker)

The recurring shape here is: review the work, then actually *resolve* every finding, prove the app
still passes its headless tests, and land it in one clean commit. The point of this skill is the
follow-through — a review that produces findings nobody acts on is wasted, and an "all fixed"
claim that wasn't tested is worse.

## Procedure

1. **Review.** Run the existing `/code-review` skill on the current diff (default effort is fine;
   use higher effort if the user asks for thoroughness). Let it surface correctness bugs and
   cleanup/simplification opportunities — don't hand-roll a parallel review.

2. **Triage into a numbered list.** Collect the findings into a single severity-ordered list
   (most serious first) and show it to the user before changing code. This mirrors how work is
   tracked here ("fix all issues one by one, starting at nr 1") and lets the user veto anything
   they disagree with. If a finding is wrong or not worth fixing, say so with a reason rather than
   complying reflexively — apply judgment to review feedback, don't rubber-stamp it.

3. **Fix one by one, starting at #1.** Make each fix surgical: change only what the finding
   requires, match surrounding style, and don't "improve" adjacent code. Remove only the
   imports/variables your own change orphaned. Keep the findings list updated as you go so it's
   clear what's done and what remains.

4. **Validate — this is the gate, not a formality.** From the project root, run both headless
   suites and confirm they pass before claiming the fixes are done:

   ```
   $env:QT_QPA_PLATFORM = "offscreen"; uv run python -m tests.test_pipeline
   $env:QT_QPA_PLATFORM = "offscreen"; uv run python -m tests.test_plugins
   ```

   If a fix changed user-visible GUI behavior (a new control, a changed workflow), the matching
   test in `tests/` usually needs updating too — update it as part of the fix, not afterward.
   Report the actual test result; if something fails, fix it before moving on.

5. **Commit.** Land all the fixes as **one descriptive commit**, matching this repo's history for
   review work — e.g. `Fix 14 code-review findings: ROI/plugin interaction, robustness, cleanup`.
   End the message with the required footer:

   ```
   Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
   ```

   Only split into multiple commits if the fixes are genuinely independent and the user wants the
   history that way.

6. **Push if asked.** If the user wants it pushed, hand off to the `commit-push` skill, which knows
   this repo's Forgejo/Tailscale remote and its auth recovery — don't reinvent the push here.

## Why it's shaped this way

- Numbered triage + show-before-fixing keeps the user in control and makes "fix nr 3" unambiguous.
- The offscreen-test gate exists because this is a PySide6 app whose tests run headless; "looks
  right" is not evidence, a green test run is. Claiming work is fixed without running the suites is
  the failure this skill is built to prevent.
- One commit per review pass keeps history readable and matches how reviews have been landed here
  before.

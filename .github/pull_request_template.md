## Summary

<!-- What does this PR do, in one or two sentences? -->

## Changes

<!-- Bullet the notable changes so a reviewer knows where to look. -->
-
-

## How it was tested

<!-- Commands run, notebooks re-run, manual checks performed. -->
-

## Checklist

- [ ] Commits follow Conventional Commits
- [ ] `pytest tests/ -v` passes locally
- [ ] Notebooks: "Restart & Run All" completes top to bottom (outputs included)
- [ ] No transformer, encoder, scaler, or vocabulary is fitted before the train/test split
- [ ] Shared logic lives in `src/` and is imported (not copied into notebooks or `app.py`)
- [ ] Docs updated if structure, flow, or conventions changed

## Notes for reviewers

<!-- Anything to focus on, open questions, or follow-ups. Delete if not needed. -->

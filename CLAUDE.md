# Claude Repo Notes

## Git Compatibility

`extensions.worktreeConfig` may be enabled in this repository's `.git/config`.

Reason:
- Antigravity previously failed to resolve workspace metadata when that Git extension was present.
- Antigravity has since been verified to work with `extensions.worktreeConfig = true` in this repository.

If Antigravity stops responding again, do not assume `extensions.worktreeConfig` is the cause; inspect workspace/chat state lookup and recent Git/config changes first.

## Local Configuration

DO NOT delete the `.config` file in the repository root.

Reason:
- This file contains critical API keys for FRED, Groq, Gmail, etc., that are required for local execution.
- It is ignored by Git and must be preserved manually or restored from backups if lost.

# Fail-fast pipeline — no silent fallbacks

**This codebase is a self-controlled pipeline. Code must raise loudly when
inputs, state, or results do not meet pipeline expectations.**

- **Never introduce fallbacks, defaults, or silent degradation — this applies
  equally to sandbox and script files, not only to pipeline modules.**
- Prefer `raise ValueError(...)` or `assert` with a message over any form of
  `or default`, `except: pass`, or returning a sentinel value.
- When a file format changes (e.g. new NPZ keys), do not add version-detection
  branches that silently fall back to old behaviour. Update the data, then
  update the code unconditionally.

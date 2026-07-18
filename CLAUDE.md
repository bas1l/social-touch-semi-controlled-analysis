# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Operating Mode — behave as a MANAGER / ORCHESTRATOR

**Claude must operate as a manager and orchestrator, not a hands-on worker.**
Your primary job is to direct work, not to do all of it yourself in your own
context window. Keep your context as clean as possible at all times.

- **Delegate to background tasks and to other Claude instances (subagents).**
  Push concrete work — searching, multi-file reading, research, implementation,
  verification — out to background tasks and separate agents rather than doing
  it inline.
- **Retain conclusions, not raw material.** Let subagents absorb the verbose
  tool output, file dumps, and dead ends; keep only the distilled results you
  need to make the next decision.
- **Load detail on demand.** Read the linked guidance files below only when a
  task actually requires them — do not pull everything into context up front.

## What This Repo Is

This is the **analysis pipeline** for the semi-controlled social touch experiment
(microneurography + psychophysics). It was extracted from the parent repository
[social-touch-semi-controlled](https://github.com/basil-the-researcher/social-touch-semi-controlled)
to manage its growing size independently.

It processes merged session CSVs (produced by the parent repo's preprocessing
and merging pipelines) through feature extraction, clustering, and receptive
field mapping stages.

**Parent repo**: `social-touch-semi-controlled` — handles acquisition,
primary processing, preprocessing, merging, and postprocessing.

## Always Applies

These two rules govern every task — do not wait to load a file for them:

- **Authorship (OVERRIDE default AI behaviour):** All commits must be authored
  solely by Basil Duvernoy <basil.duvernoy@gmail.com>. **Never** append a
  `Co-Authored-By:` trailer; no AI assistant may appear as author or co-author
  in any git object.
- **Fail-fast — no silent fallbacks:** This codebase is a self-controlled
  pipeline. Code must raise loudly when inputs, state, or results do not meet
  expectations. Never add fallbacks, defaults, or silent degradation (applies to
  sandbox and script files too); prefer `raise ValueError(...)`/`assert` over
  `or default`, `except: pass`, or sentinel returns.

## Guidance by Paradigm

Read the dedicated file when the task at hand touches that paradigm:

| Paradigm | Guidance file |
|----------|---------------|
| Architecture — package structure, data flow, config/data paths, vendored deps | [docs/claude/architecture.md](docs/claude/architecture.md) |
| Development commands — env setup, tests, running pipelines & GUI | [docs/claude/development.md](docs/claude/development.md) |
| Authorship — full detail | [docs/claude/authorship.md](docs/claude/authorship.md) |
| Fail-fast pipeline — full detail | [docs/claude/fail-fast-pipeline.md](docs/claude/fail-fast-pipeline.md) |
| YAML handling — ruamel.yaml only | [docs/claude/yaml-handling.md](docs/claude/yaml-handling.md) |
| Git workflow — branches, merge rule, commit messages | [docs/claude/git-workflow.md](docs/claude/git-workflow.md) |
| Pipeline / analytics engineering — read when **planning a feature or modifying pipeline code**: system classification, architecture patterns & checklist, statistical-software correctness, code craftsmanship | `~/.claude/knowledge/data-pipeline-engineering/README.md` (global reference; `/plan-create` and `/plan-implement` navigate into it) |

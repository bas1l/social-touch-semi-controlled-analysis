# Git Workflow

## Branch strategy

```
main          ← production-ready (protected)
  └── dev     ← integration branch (protected)
       └── feature/*, fix/*, refactor/*, docs/*, test/*
```

## Merge rule — always `--no-ff`

## Commit messages — Conventional Commits

```
<type>(<scope>): <subject>
```

Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `style`, `perf`.

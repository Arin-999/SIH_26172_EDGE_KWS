# Contributing to ps26172-edge-kws

## Getting Started

1. **Fork** the repository and clone your fork.
2. Create a feature branch from `develop`:
   ```bash
   git checkout develop
   git checkout -b feature/your-feature-name
   ```
3. Install dependencies: `pip install -r requirements.txt`
4. Make your changes.
5. Run the test suite: `python -m pytest tests/ -m "not hardware and not model"`
6. Open a Pull Request against `develop`.

## Branch Naming

| Prefix | Usage |
|---|---|
| `feature/` | New functionality |
| `fix/` | Bug fixes |
| `docs/` | Documentation only |
| `refactor/` | Code cleanup with no behaviour change |
| `test/` | Test additions or fixes |

## Commit Messages

Use imperative, lowercase present tense:

```
add MFCC caching to feature extractor
fix debounce threshold in integration test
update FAR/FRR to remove sklearn dependency
```

Avoid: `updated`, `changes`, `final`, `test`, `misc`.

## Code Style

- Python: follow `PEP 8`. Use type hints on all public functions.
- C++: follow the existing brace style in `firmware/`.
- All new Python modules must have a module-level docstring.
- All public functions must have a NumPy-style docstring with Args / Returns.

## Testing Requirements

A feature is not complete until:

- [ ] Unit test written and passing
- [ ] Integration with existing modules verified
- [ ] `pytest -m "not hardware and not model"` passes with no new failures
- [ ] Docstring updated

For ML changes, also:

- [ ] FAR/FRR measured on test split
- [ ] Model size and latency benchmarked

## Pull Request Checklist

- [ ] Branch is up-to-date with `develop`
- [ ] All tests pass locally
- [ ] New code has docstrings and type hints
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] No debug prints left in production code
- [ ] `INFRASTRUCTURE.md` updated if adding/modifying a module

## Reporting Issues

Use the GitHub Issues tracker. Include:

1. OS and Python version
2. Full error traceback
3. Minimal reproducing example
4. Expected vs actual behaviour

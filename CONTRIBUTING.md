# Contributing to AI Forensics Python

First off, **thank you for taking the time to contribute!**  
The following guidelines help keep the project consistent and maintainable.

>NOTE: Still in draft

---

## Table of Contents
1.  [Code of Conduct](#code-of-conduct)  
2.  [Getting Started](#getting-started)  
3.  [Branching & Workflow](#branching--workflow)  
4.  [Commit Conventions](#commit-conventions)  
5.  [Developer Certificate of Origin (DCO)](#developer-certificate-of-origin-dco)  
6.  [Coding Style](#coding-style)  
7.  [Testing](#testing)  
8.  [Pull-Request Checklist](#pull-request-checklist)  
9.  [Release Process](#release-process)  

---


## Code of Conduct
We follow the [Contributor Covenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/) v2.1.  
Please be respectful and inclusive in all interactions.

---

### Getting Started (Poetry)

```bash
# clone
git clone https://github.com/interwebshack/ai-forensics-python.git
cd ai-forensics-python

# install Poetry (if not already)
curl -sSL https://install.python-poetry.org | python3 -

# create & activate venv + install deps
poetry install --with dev

*Poetry automatically selects Python ≥ 3.10 on your system or the one specified in pyproject.toml. All dev tools—ruff, black, mypy, pytest, invoke—are declared under [tool.poetry.group.dev].*  

### Prerequisites

| Tool             | Required Version | Notes |
|------------------|------------------|-------|
| **Poetry**       | 2.0 or newer     | Dependency & virtual-env manager |
| **Python**       | 3.10 < 4.x       | Poetry will auto-select within this range |
| **plantuml**     | _optional_       | Render `.puml` diagrams locally (`brew install plantuml`) |
| **Docker**       | _optional_       | Quick PlantUML preview, OCI registry tests |

## Branching & Workflow

* **`main`** — always releasable; tagged releases are cut from here.  
* **Feature branches** — name `feature/<short-topic>` (e.g., `feature/registry-push`).  
* **Bug-fix branches** — name `bugfix/<issue-#>` (e.g., `bugfix/42-null-sig-path`).  
* **Documentation-only branches** — name `docs/<topic>`.

Workflow:

1. Create an **Issue** if one doesn’t exist.  
2. Branch from `main` with the naming rules above.  
3. Work locally in a Poetry shell:  
   ```bash
   poetry shell
   git checkout -b feature/my-cool-thing
   ```
4. Keep commits small, descriptive, and signed (`git commit -s`).
5. Push to your fork / to the repo: `git push -u origin feature/my-cool-thing`.
6. Open a **pull request** against `main`. GitHub Actions will run:
  * lint → test → type-check → coverage → packaging checks.  
7. Address review comments; squash-merge once approved.  
> **Tip:** Use `git rebase -i main` to keep history clean before opening the PR.  

## Commit Conventions

We follow **Conventional Commits**:
```bash
feature(signer): add --push flag for sign-blob
bugfix(verifier): handle missing Rekor bundle gracefully
docs: update README diagrams
```

## Developer Certificate of Origin (DCO)

By submitting a PR you certify you have the right to submit the work under the project’s [MIT license](LICENSE).
Sign your commits:

```bash
git commit -s -m "bugfix: correct typo"
```
The `-s` flag adds the required `Signed-off-by` line.

## Coding Style

* **ruff** for linting (`black`, `isort`, `flake8` rules)
* Docstrings follow **Google style**.

Run all checks:

* **black** for formatting (PEP 8 compliant).
```bash
poetry run black --verbose ai_forensics
```

* **isort** your imports, so you don't have to.
```bash
poetry run isort --verbose ai_forensics
```

## Testing

* **pytest** with **pytest-asyncio** for async routines.
* 90%+ coverage target.

```bash
pytest -q
```
CI will fail if coverage drops below the threshold.  

## Pull-Request Checklist

Before requesting a review, make sure you can check **every** box below:

- [ ] **Issue linked** (e.g., “Closes #123”) or marked N/A for trivial docs fixes.  
- [ ] **Unit / integration tests** added or updated; `poetry run pytest` passes locally.  
- [ ] **Docs updated** (`README`, `docs/`, or inline docstrings) if behavior changed.  
- [ ] **Changelog entry** added under `## [Unreleased]` in `CHANGELOG.md`.  
- [ ] Required development tools pass (ruff, black, mypy, etc.).  
  - [ ] `poetry run black ai_forensics`
- [ ] All commits are **signed** (`git commit -s`) to satisfy the DCO.  
- [ ] PR title follows **Conventional Commits** (e.g., `feat(verifier): add policy flag`).  
- [ ] CI status is green (GitHub Actions).

> 🔒  PRs without a Signed-off-by line will be blocked by the DCO GitHub check.

## Release Process

1. Merge all features into `main`.
2. Bump version with `bumpver update --push-tag`.
3. GitHub Action builds sdist & wheels and uploads to PyPI.
4. Draft release notes in `CHANGELOG.md`.

Happy hacking! 💙

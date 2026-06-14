"""Dockerfile / .dockerignore structural contract.

These tests are a *static* guard, not a build: this machine may have no Docker
daemon, so they cannot prove the image builds — they pin the load-bearing
invariants of the [Dockerfile](../Dockerfile) so it cannot silently drift away
from what makes the image work or from the rest of the project:

- the base is pinned to the project's Python (3.11, matching the dev venv);
- both build targets exist and the DEFAULT (last) stage is the lean faithful
  `slim` image, not the multi-GB `ai` one;
- the entry point stays in sync with ``[project.scripts]`` in pyproject;
- the OpenCV system libs are present (``import cv2`` fails on the slim base
  without them);
- the `ai` target installs CPU-only torch (the default PyPI wheels are CUDA and
  multi-GB);
- the build context (.dockerignore) excludes venvs/VCS but keeps the files the
  build actually needs.

Pure packaging/infra: no pixels or output bytes are touched.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = REPO_ROOT / "Dockerfile"
DOCKERIGNORE = REPO_ROOT / ".dockerignore"
PYPROJECT = REPO_ROOT / "pyproject.toml"


def _dockerfile_text() -> str:
    assert DOCKERFILE.exists(), "Dockerfile is missing from the repo root"
    return DOCKERFILE.read_text(encoding="utf-8")


def _stage_names(text: str) -> list[str]:
    """Multi-stage target names, in file order (`FROM <img> AS <name>`)."""
    return re.findall(r"(?im)^FROM\s+\S+\s+AS\s+([\w-]+)", text)


def _console_script_name() -> str:
    """The single `facekeep` console-script name from [project.scripts]."""
    try:
        import tomllib

        data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
        scripts = data["project"]["scripts"]
        return next(iter(scripts))
    except ModuleNotFoundError:  # Python 3.10 has no tomllib — regex fallback
        m = re.search(
            r"(?ms)^\[project\.scripts\]\s*\n\s*([A-Za-z0-9_.-]+)\s*=",
            PYPROJECT.read_text(encoding="utf-8"),
        )
        assert m, "could not find [project.scripts] in pyproject.toml"
        return m.group(1)


def test_dockerfile_exists_and_pins_python_311():
    text = _dockerfile_text()
    # Pinned to the project's Python (the dev .venv is 3.11); a bare `FROM python`
    # would float and is not reproducible.
    assert "FROM python:3.11-slim" in text


def test_both_targets_exist_and_slim_is_default():
    names = _stage_names(_dockerfile_text())
    assert {"base", "ai", "slim"} <= set(names), names
    # The last stage is the default build target. It must be the lean faithful
    # `slim` image so a plain `docker build .` does NOT build the multi-GB `ai`.
    assert names[-1] == "slim", f"default (last) stage must be 'slim', got {names!r}"


def test_entrypoint_matches_pyproject_console_script():
    text = _dockerfile_text()
    script = _console_script_name()
    # If the console script is ever renamed, this fails until the Dockerfile's
    # ENTRYPOINT is updated to match.
    assert f'ENTRYPOINT ["{script}"]' in text, (
        f'Dockerfile ENTRYPOINT must invoke the "{script}" console script'
    )


def test_opencv_system_libs_present():
    text = _dockerfile_text()
    # opencv-python (non-headless, as pinned) needs these to `import cv2` on the
    # slim Debian base; dropping them would break every command in the container.
    assert "libgl1" in text
    assert "libglib2.0-0" in text


def test_ai_target_uses_cpu_only_torch():
    text = _dockerfile_text()
    # The default PyPI torch wheels are CUDA-enabled and multi-GB; the AI image
    # must pin the CPU index so it stays installable/runnable anywhere.
    assert "download.pytorch.org/whl/cpu" in text


def test_dockerignore_excludes_venv_and_vcs():
    assert DOCKERIGNORE.exists(), ".dockerignore is missing from the repo root"
    lines = {
        ln.strip().rstrip("/")
        for ln in DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    }
    assert ".venv" in lines
    assert ".git" in lines


def test_dockerignore_keeps_build_inputs():
    """The build COPYs these — they must NOT be ignored, or the build breaks."""
    lines = {
        ln.strip().rstrip("/")
        for ln in DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    }
    for needed in ("pyproject.toml", "README.md", "LICENSE", "facekeep"):
        assert needed not in lines, f"{needed} is required by the build; do not ignore it"

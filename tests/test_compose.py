"""docker-compose.yml structural contract (NAS deployment).

Like ``test_docker.py``, this is a *static* guard, not a build: it cannot prove
the stack runs on a NAS, but it pins the load-bearing invariants of
[docker-compose.yml](../docker-compose.yml) so it can't silently drift from the
Dockerfile it wraps or from what makes a NAS backup work:

- both services exist; the default `facekeep` service builds the lean `slim`
  target and runs a `compress` backup; the `ai` service builds the `ai` target
  and is gated behind the `ai` profile (so a plain run never pulls the multi-GB
  image);
- the build targets are real Dockerfile stages (cross-file anti-drift);
- the AI model-cache volume mounts at ``<HOME>/.cache/facekeep`` with HOME read
  from the Dockerfile (so persisted weights land where the app looks for them);
- the model-cache named volume is declared;
- neither service overrides the image's `facekeep` ENTRYPOINT;
- the source photos are mounted read-only (a backup tool never touches originals).

Pure packaging/infra: no pixels or output bytes are touched.
"""

import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
COMPOSE = REPO_ROOT / "docker-compose.yml"
DOCKERFILE = REPO_ROOT / "Dockerfile"


def _compose() -> dict:
    assert COMPOSE.exists(), "docker-compose.yml is missing from the repo root"
    data = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    assert isinstance(data, dict) and "services" in data, "compose has no services"
    return data


def _dockerfile_text() -> str:
    assert DOCKERFILE.exists(), "Dockerfile is missing from the repo root"
    return DOCKERFILE.read_text(encoding="utf-8")


def _stage_names(text: str) -> list[str]:
    return re.findall(r"(?im)^FROM\s+\S+\s+AS\s+([\w-]+)", text)


def _dockerfile_home(text: str) -> str:
    m = re.search(r"(?im)^ENV\s+HOME=(\S+)", text)
    assert m, "Dockerfile must set ENV HOME=... (the AI model-cache root)"
    return m.group(1)


def _named_volume_target(volumes: list[str], name: str) -> str | None:
    """Container path a named volume mounts to (strip an optional :ro/:rw mode).

    Parsed by prefix, not a colon-split, because other mounts use
    ``${VAR:-default}`` whose ``:-`` contains a colon.
    """
    prefix = f"{name}:"
    for v in volumes:
        if v.startswith(prefix):
            target = v[len(prefix):]
            for mode in (":ro", ":rw"):
                if target.endswith(mode):
                    target = target[: -len(mode)]
            return target
    return None


def test_compose_exists_and_has_both_services():
    assert {"facekeep", "ai"} <= set(_compose()["services"])


def test_default_service_builds_slim_and_is_a_backup():
    svc = _compose()["services"]["facekeep"]
    assert svc["build"]["context"] == "."
    assert svc["build"]["target"] == "slim"
    # The default service is the lean faithful image and runs a compress backup.
    assert svc["command"][0] == "compress"


def test_ai_service_builds_ai_target_behind_profile():
    svc = _compose()["services"]["ai"]
    assert svc["build"]["target"] == "ai"
    # Gated so a plain `docker compose up/run` never pulls the multi-GB AI image.
    assert "ai" in svc.get("profiles", [])


def test_compose_targets_are_real_dockerfile_stages():
    data = _compose()
    stages = set(_stage_names(_dockerfile_text()))
    used = {
        data["services"]["facekeep"]["build"]["target"],
        data["services"]["ai"]["build"]["target"],
    }
    assert used <= stages, f"compose targets {used} not all Dockerfile stages {stages}"


def test_ai_model_cache_mounts_at_dockerfile_home():
    svc = _compose()["services"]["ai"]
    target = _named_volume_target(svc["volumes"], "facekeep-models")
    assert target is not None, "ai service must mount the facekeep-models volume"
    home = _dockerfile_home(_dockerfile_text())
    # Weights must persist at <HOME>/.cache/facekeep, in sync with the Dockerfile;
    # drift here would silently send downloads to a non-persisted path.
    assert target == f"{home}/.cache/facekeep", target


def test_model_cache_named_volume_declared():
    assert "facekeep-models" in (_compose().get("volumes") or {})


def test_no_service_overrides_entrypoint():
    services = _compose()["services"]
    # The image ENTRYPOINT is the `facekeep` console script; commands are its
    # args. Overriding entrypoint would break that contract.
    for name in ("facekeep", "ai"):
        assert "entrypoint" not in services[name], name


def test_source_photos_mounted_read_only():
    # A backup tool must never modify originals: the source mount is read-only.
    for name in ("facekeep", "ai"):
        vols = _compose()["services"][name]["volumes"]
        photo_mounts = [v for v in vols if "/work/photos" in v]
        assert photo_mounts, f"{name} must mount source photos at /work/photos"
        assert all(v.endswith(":ro") for v in photo_mounts), photo_mounts

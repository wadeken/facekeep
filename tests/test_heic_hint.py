"""Friendly HEIC-miss behavior (ROADMAP "Friendly HEIC miss").

A ``.heic``/``.heif`` input given without the optional ``[heic]`` extra
(pillow-heif) should be a clean, actionable **SKIP** — not a scary FAILED — and
the message names the exact ``pip install`` command. Crucially, this is
HEIC-specific and gated on plugin availability, so it never masks a *real*
failure (a corrupt HEIC with the plugin installed, or any other unsupported
input) as a friendly skip.

The main ``.venv`` does not install pillow-heif (it's the ``[heic]`` extra, not
in ``[dev]``), so a real ``.heic`` naturally raises ``UnsupportedInputError``
here. The tests also monkeypatch ``_heic_plugin_available`` to stay deterministic
regardless of which extras the running environment happens to have.
"""

from pathlib import Path

from click.testing import CliRunner

from facekeep import cli
from facekeep.config import FaceKeepConfig

_HINT = 'pip install "facekeep[heic]"'


def _write(p: Path, data: bytes = b"not a real image") -> str:
    p.write_bytes(data)
    return str(p)


def test_plugin_available_returns_bool():
    assert isinstance(cli._heic_plugin_available(), bool)


def test_install_hint_names_the_pip_extra():
    assert _HINT in cli._heic_install_hint()


def test_process_one_heic_without_plugin_is_friendly_skip(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_heic_plugin_available", lambda: False)
    src = _write(tmp_path / "IMG_1234.heic")
    res = cli._process_one(src, str(tmp_path / "IMG_1234"), FaceKeepConfig(),
                           False, False)
    assert res["status"] == "skipped"
    assert _HINT in res["error"]
    assert not (tmp_path / "IMG_1234.avif").exists()


def test_process_one_heif_extension_also_friendly(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_heic_plugin_available", lambda: False)
    src = _write(tmp_path / "photo.heif")
    res = cli._process_one(src, str(tmp_path / "photo"), FaceKeepConfig(),
                           False, False)
    assert res["status"] == "skipped"
    assert _HINT in res["error"]


def test_process_one_heic_with_plugin_stays_failure(tmp_path, monkeypatch):
    # Plugin "present" -> a HEIC that still can't be read is a real failure,
    # never masked as a friendly skip.
    monkeypatch.setattr(cli, "_heic_plugin_available", lambda: True)
    src = _write(tmp_path / "broken.heic")
    res = cli._process_one(src, str(tmp_path / "broken"), FaceKeepConfig(),
                           False, False)
    assert res["status"] in ("failed", "failed-unexpected")
    assert _HINT not in res.get("error", "")


def test_process_one_non_heic_unsupported_stays_failure(tmp_path, monkeypatch):
    # The friendly path is HEIC-specific: a corrupt .jpg is still a real failure
    # even when the HEIC plugin is absent.
    monkeypatch.setattr(cli, "_heic_plugin_available", lambda: False)
    src = _write(tmp_path / "broken.jpg")
    res = cli._process_one(src, str(tmp_path / "broken"), FaceKeepConfig(),
                           False, False)
    assert res["status"] in ("failed", "failed-unexpected")


def test_cli_compress_heic_without_plugin_prints_hint(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_heic_plugin_available", lambda: False)
    src = tmp_path / "photo.heic"
    src.write_bytes(b"not a real image")
    result = CliRunner().invoke(cli.cli, ["compress", str(src), "--no-index"])
    assert result.exit_code == 0, result.output
    assert "SKIP" in result.output
    assert _HINT in result.output
    assert not (tmp_path / "photo.avif").exists()

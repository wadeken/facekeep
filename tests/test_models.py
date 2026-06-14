"""Model/weights cache — ROADMAP Phase 4 (model/weights management).

Covers facekeep.models.ensure_weights: a shared, checksum-verified cache under
~/.cache/facekeep/models with a clear, [ai]-pointing offline error. All of these
run **offline** — the network is monkeypatched out — so the suite stays green
without touching PyPI/GitHub. One @pytest.mark.real_ai test asserts the restorer
actually routes its weights through the cache (a *local path*, not a URL).
"""

import hashlib

import pytest

import facekeep.models as models
from facekeep.exceptions import ModelDownloadError

# A blob comfortably over the _MIN_VALID_BYTES floor so it passes the size check.
_BLOB = b"weights-bytes" * 2000
_BLOB_SHA = hashlib.sha256(_BLOB).hexdigest()


def _fake_urlopen(data, calls):
    """Build a urlopen replacement that yields `data` and records each call."""

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return data

    def _open(req, timeout=None):
        calls.append(getattr(req, "full_url", req))
        return _Resp()

    return _open


def test_downloads_verifies_and_caches(tmp_path, monkeypatch):
    """A cache miss downloads, verifies the checksum, and writes the file once."""
    calls = []
    monkeypatch.setattr(
        models.urllib.request, "urlopen", _fake_urlopen(_BLOB, calls)
    )
    dest = models.ensure_weights(
        "https://example/w.pth", "w.pth", sha256=_BLOB_SHA, cache_dir=tmp_path
    )
    assert dest == tmp_path / "w.pth"
    assert dest.read_bytes() == _BLOB
    assert len(calls) == 1  # downloaded exactly once
    # No leftover .part temp file from the atomic write.
    assert not (tmp_path / "w.pth.part").exists()


def test_cache_hit_does_not_download(tmp_path, monkeypatch):
    """An existing, checksum-matching file is returned without any network call."""
    (tmp_path / "w.pth").write_bytes(_BLOB)

    def _boom(*a, **k):
        raise AssertionError("urlopen must not be called on a cache hit")

    monkeypatch.setattr(models.urllib.request, "urlopen", _boom)
    dest = models.ensure_weights(
        "https://example/w.pth", "w.pth", sha256=_BLOB_SHA, cache_dir=tmp_path
    )
    assert dest.read_bytes() == _BLOB


def test_corrupt_cache_is_redownloaded(tmp_path, monkeypatch):
    """A cached file failing its checksum is treated as corrupt and re-fetched."""
    # Pre-seed a wrong-but-large file so size passes but checksum fails.
    (tmp_path / "w.pth").write_bytes(b"x" * len(_BLOB))
    calls = []
    monkeypatch.setattr(
        models.urllib.request, "urlopen", _fake_urlopen(_BLOB, calls)
    )
    dest = models.ensure_weights(
        "https://example/w.pth", "w.pth", sha256=_BLOB_SHA, cache_dir=tmp_path
    )
    assert len(calls) == 1  # the bad cache forced a download
    assert dest.read_bytes() == _BLOB  # good bytes now cached


def test_checksum_mismatch_on_download_raises_and_leaves_no_file(tmp_path, monkeypatch):
    """A download whose bytes fail the checksum errors and writes nothing."""
    calls = []
    monkeypatch.setattr(
        models.urllib.request, "urlopen", _fake_urlopen(_BLOB, calls)
    )
    with pytest.raises(ModelDownloadError, match="checksum"):
        models.ensure_weights(
            "https://example/w.pth", "w.pth",
            sha256="0" * 64, cache_dir=tmp_path,
        )
    # Neither the final file nor the temp part is left behind.
    assert not (tmp_path / "w.pth").exists()
    assert not (tmp_path / "w.pth.part").exists()


def test_offline_raises_with_ai_hint(tmp_path, monkeypatch):
    """An unreachable host raises ModelDownloadError pointing at the [ai] extra."""
    def _offline(req, timeout=None):
        raise models.urllib.error.URLError("offline")

    monkeypatch.setattr(models.urllib.request, "urlopen", _offline)
    with pytest.raises(ModelDownloadError, match=r"facekeep\[ai\]"):
        models.ensure_weights(
            "https://example/w.pth", "w.pth", sha256=_BLOB_SHA, cache_dir=tmp_path
        )


def test_too_small_download_is_rejected(tmp_path, monkeypatch):
    """A tiny payload (error page / truncation) is rejected before any checksum."""
    calls = []
    monkeypatch.setattr(
        models.urllib.request, "urlopen", _fake_urlopen(b"nope", calls)
    )
    with pytest.raises(ModelDownloadError, match="invalid"):
        models.ensure_weights(
            "https://example/w.pth", "w.pth", sha256=None, cache_dir=tmp_path
        )
    assert not (tmp_path / "w.pth").exists()


def test_no_checksum_skips_verification(tmp_path, monkeypatch):
    """sha256=None caches whatever was downloaded (verification opt-out)."""
    calls = []
    monkeypatch.setattr(
        models.urllib.request, "urlopen", _fake_urlopen(_BLOB, calls)
    )
    dest = models.ensure_weights(
        "https://example/w.pth", "w.pth", sha256=None, cache_dir=tmp_path
    )
    assert dest.read_bytes() == _BLOB


# --------------------------------------------------------------------------- #
# Restorer wiring: a download failure degrades gracefully (offline-first), and
# the real AI path routes weights through the cache (local path, not a URL).
# --------------------------------------------------------------------------- #

def test_init_upsampler_model_download_failure_falls_back(monkeypatch):
    """If ensure_weights raises, _init_upsampler degrades to bicubic (no crash)."""
    from facekeep.aggressive import restorer as _restorer
    from facekeep.config import FaceKeepConfig

    def _fail(*a, **k):
        raise ModelDownloadError("offline")

    monkeypatch.setattr(_restorer, "ensure_weights", _fail)
    r = _restorer.Restorer(FaceKeepConfig().aggressive)
    r._init_upsampler()
    assert r._upsampler is None


def test_init_face_enhancer_model_download_failure_skips(monkeypatch):
    """If ensure_weights raises, _init_face_enhancer skips GFPGAN (no crash)."""
    from facekeep.aggressive import restorer as _restorer
    from facekeep.config import FaceKeepConfig

    def _fail(*a, **k):
        raise ModelDownloadError("offline")

    monkeypatch.setattr(_restorer, "ensure_weights", _fail)
    r = _restorer.Restorer(FaceKeepConfig().aggressive)
    r._init_face_enhancer()
    assert r._face_enhancer is None


@pytest.mark.real_ai
def test_realesrgan_weights_route_through_cache(monkeypatch):
    """The real AI path hands RealESRGANer a local cached path, not the URL."""
    from facekeep.aggressive import restorer as _restorer
    from facekeep.config import FaceKeepConfig

    _restorer._ensure_torchvision_compat()
    pytest.importorskip("realesrgan", reason="[ai] extra not installed")

    captured = {}

    # ensure_weights must be called with the configured URL + checksum and its
    # returned local path is what gets loaded.
    real_ensure = _restorer.ensure_weights

    def _spy(url, filename, *, sha256=None, **k):
        captured["url"] = url
        captured["sha256"] = sha256
        return real_ensure(url, filename, sha256=sha256, **k)

    monkeypatch.setattr(_restorer, "ensure_weights", _spy)

    # Capture the model_path actually passed to RealESRGANer without running it.
    import realesrgan

    def _fake_realesrganer(*a, model_path=None, **k):
        captured["model_path"] = model_path
        return object()

    # _init_upsampler does `from realesrgan import RealESRGANer` each call, so
    # patching the module attribute is picked up (no real model is constructed).
    monkeypatch.setattr(realesrgan, "RealESRGANer", _fake_realesrganer)

    cfg = FaceKeepConfig()
    cfg.aggressive.face_enhance = False
    r = _restorer.Restorer(cfg.aggressive)
    try:
        r._init_upsampler()
    except ModelDownloadError:
        pytest.skip("weights unavailable (offline) — cannot verify cache routing")

    assert captured.get("url", "").startswith("https://")
    assert captured.get("sha256")  # a checksum was supplied
    mp = captured.get("model_path")
    assert mp is not None
    assert not str(mp).startswith("http")  # a LOCAL path, not a URL

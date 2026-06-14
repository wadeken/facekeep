"""Quality-targeted aggressive mode — ROADMAP Phase 4.

Instead of compressing every photo at one fixed ``bg_scale``, opt-in
quality-targeting walks ``aggressive.quality_scale_candidates`` and picks the
*most aggressive* (smallest) scale whose reconstructed background still meets a
target perceptual quality — measured with **LPIPS** (learned perceptual
distance; lower = more similar), the right metric for a hallucinated-but-
plausible background (SSIM is the wrong tool). So each photo is compressed as
hard as it can be without looking wrong.

Design pinned here:

* the search picks the smallest candidate with LPIPS <= ``quality_target``;
* if no candidate qualifies, it falls back to the candidate that came *closest*
  (lowest LPIPS), never below the content-aware/no-face floor;
* it is opt-in (``quality_target=None`` => off, the fixed-scale behavior, byte-
  for-byte unchanged) and degrades gracefully when LPIPS is unavailable
  (offline-first: returns the floor, warns, never crashes);
* it composes with content-aware conservatism — it only ever raises the scale
  (never picks one *more* aggressive than the conservative floor);
* the new output-affecting fields bust the incremental-index fingerprint
  (aggressive only; faithful untouched);
* ``validate()`` range-checks the fields and they survive a YAML round-trip.

LPIPS is **faked** throughout (a deterministic monotone function of the scale)
so these tests need no torch/lpips and stay offline, matching the repo's
fake-model convention in tests/test_lpips.py.
"""

import cv2
import numpy as np
import pytest

import facekeep.aggressive.compressor as compressor_mod
from facekeep.aggressive.compressor import _search_bg_scale, compress_photo
from facekeep.aggressive.format import read_fkeep_info, write_fkeep
from facekeep.config import AggressiveConfig, FaceKeepConfig
from facekeep.detector import FaceRegion
from facekeep.exceptions import ConfigError
from facekeep.index import settings_fingerprint


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _image(h=600, w=900) -> np.ndarray:
    """A deterministic image; the search downsamples/upscales it but the fake
    LPIPS ignores the pixels, so the content only needs to be a valid frame."""
    rng = np.random.default_rng(7)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _write_jpg(tmp_path, name, img) -> str:
    path = tmp_path / name
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return str(path)


def _face(x1, y1, x2, y2, conf=0.9) -> FaceRegion:
    return FaceRegion(id=0, bbox=(x1, y1, x2, y2),
                      padded_bbox=(x1, y1, x2, y2), confidence=conf)


def _patch_detector(monkeypatch, faces):
    class _Fixed:
        def detect(self, image):
            return list(faces)

    monkeypatch.setattr(compressor_mod, "create_detector", lambda **kw: _Fixed())


def _patch_lpips(monkeypatch, dist_fn, available=True):
    """Inject a fake LPIPS into the metrics module the compressor imports.

    ``dist_fn(a, b)`` returns the distance; the compressor reconstructs ``b`` by
    bicubic-upscaling a downsample at the candidate scale. To make the fake
    deterministic *per scale*, we recover the scale from b's downsample is hard,
    so instead the tests drive the search by patching with closures that key off
    a mutable list of returned values (FIFO over the ascending candidate order).
    When ``available`` is False, ``lpips_available`` reports unavailable.
    """
    import facekeep.metrics as metrics_mod

    monkeypatch.setattr(metrics_mod, "lpips_available", lambda: available)
    monkeypatch.setattr(metrics_mod, "lpips_distance", dist_fn)


def _scale_keyed_lpips(mapping, default=1.0):
    """A fake lpips_distance that returns a value based on the recovered scale.

    The compressor calls ``lpips_distance(original, recon)`` where ``recon`` is
    the original downsampled-then-upscaled. We can't read the scale off ``recon``
    directly, so this factory instead returns values from ``mapping`` consumed in
    ascending-candidate order (the order the search evaluates them). Each call
    pops the next expected distance.
    """
    seq = list(mapping)

    def _fn(a, b):
        return seq.pop(0) if seq else default

    return _fn


# --------------------------------------------------------------------------- #
# A. The pure search: _search_bg_scale
# --------------------------------------------------------------------------- #

def test_search_disabled_returns_floor(monkeypatch):
    """quality_target=None => no search, returns the floor unchanged."""
    cfg = AggressiveConfig()  # quality_target defaults to None
    scale, reason = _search_bg_scale(cfg, _image(), floor_scale=0.25)
    assert scale == 0.25 and reason is None


def test_search_picks_smallest_qualifying_scale(monkeypatch):
    """Ascending candidates [0.125, 0.1667, 0.25, ...]; the smallest with
    LPIPS <= target wins (most aggressive that still looks right)."""
    cfg = AggressiveConfig(
        quality_target=0.15,
        quality_scale_candidates=[0.125, 0.25, 0.5],
    )
    # Distances in ascending-candidate order: 0.125 too high, 0.25 meets target.
    _patch_lpips(monkeypatch, _scale_keyed_lpips([0.30, 0.12, 0.05]))
    scale, reason = _search_bg_scale(cfg, _image(), floor_scale=0.05)
    assert scale == 0.25
    assert reason is not None and "0.25" in reason


def test_search_picks_most_aggressive_when_first_qualifies(monkeypatch):
    """If the most aggressive candidate already meets the target, take it and
    stop (don't keep walking to gentler scales)."""
    cfg = AggressiveConfig(
        quality_target=0.5,
        quality_scale_candidates=[0.125, 0.25, 0.5],
    )
    # First candidate (0.125) already qualifies; later ones never evaluated.
    _patch_lpips(monkeypatch, _scale_keyed_lpips([0.10]))
    scale, reason = _search_bg_scale(cfg, _image(), floor_scale=0.05)
    assert scale == 0.125


def test_search_falls_back_to_closest_when_none_qualify(monkeypatch):
    """If no candidate meets the target, choose the one with the lowest LPIPS
    (closest to acceptable) rather than failing."""
    cfg = AggressiveConfig(
        quality_target=0.05,
        quality_scale_candidates=[0.125, 0.25, 0.5],
    )
    # None <= 0.05; 0.5 is the closest (0.20 is the min distance).
    _patch_lpips(monkeypatch, _scale_keyed_lpips([0.40, 0.30, 0.20]))
    scale, reason = _search_bg_scale(cfg, _image(), floor_scale=0.05)
    assert scale == 0.5
    assert reason is not None and "unmet" in reason


def test_search_never_below_floor(monkeypatch):
    """Candidates at/below the floor are skipped; the result never drops below it.

    With a floor of 0.30, only 0.5 is a real candidate (0.125/0.25 <= floor are
    skipped), so even if the fake says it qualifies the result is >= floor.
    """
    cfg = AggressiveConfig(
        quality_target=0.5,
        quality_scale_candidates=[0.125, 0.25, 0.5],
    )
    # Only 0.5 survives the floor filter; it qualifies.
    _patch_lpips(monkeypatch, _scale_keyed_lpips([0.10]))
    scale, _ = _search_bg_scale(cfg, _image(), floor_scale=0.30)
    assert scale == 0.5
    assert scale >= 0.30


def test_search_floor_when_no_candidate_above_floor(monkeypatch):
    """If every candidate is at/below the floor, the search has nothing to do and
    returns the floor (the conservative decision already made elsewhere)."""
    cfg = AggressiveConfig(
        quality_target=0.5,
        quality_scale_candidates=[0.125, 0.25],
    )
    called = {"n": 0}

    def _fn(a, b):
        called["n"] += 1
        return 0.0

    _patch_lpips(monkeypatch, _fn)
    scale, reason = _search_bg_scale(cfg, _image(), floor_scale=0.5)
    assert scale == 0.5 and reason is None
    assert called["n"] == 0  # no candidate scored


def test_search_unavailable_lpips_degrades(monkeypatch):
    """LPIPS not installed => search skipped, floor used, no crash (offline-first)."""
    cfg = AggressiveConfig(quality_target=0.15)

    def _should_not_be_called(a, b):  # pragma: no cover - must not run
        raise AssertionError("lpips_distance called despite being unavailable")

    _patch_lpips(monkeypatch, _should_not_be_called, available=False)
    scale, reason = _search_bg_scale(cfg, _image(), floor_scale=0.25)
    assert scale == 0.25 and reason is None


def test_search_skips_candidates_that_error(monkeypatch):
    """A candidate whose LPIPS returns None (inference error) is skipped; the
    search uses the candidates that did score."""
    cfg = AggressiveConfig(
        quality_target=0.15,
        quality_scale_candidates=[0.125, 0.25, 0.5],
    )
    # 0.125 errors (None), 0.25 qualifies.
    _patch_lpips(monkeypatch, _scale_keyed_lpips([None, 0.10, 0.05]))
    scale, _ = _search_bg_scale(cfg, _image(), floor_scale=0.05)
    assert scale == 0.25


def test_search_all_errored_returns_floor(monkeypatch):
    """If every candidate's LPIPS errors, fall back to the floor (nothing scored)."""
    cfg = AggressiveConfig(
        quality_target=0.15,
        quality_scale_candidates=[0.125, 0.25],
    )
    _patch_lpips(monkeypatch, _scale_keyed_lpips([None, None]))
    scale, reason = _search_bg_scale(cfg, _image(), floor_scale=0.2)
    assert scale == 0.2 and reason is None


# --------------------------------------------------------------------------- #
# B. End-to-end through compress_photo (detector + LPIPS mocked)
# --------------------------------------------------------------------------- #

def test_compress_quality_target_sets_manifest_scale(tmp_path, monkeypatch):
    """The searched scale flows into effective_bg_scale and the manifest."""
    path = _write_jpg(tmp_path, "q.jpg", _image(800, 1000))
    _patch_detector(monkeypatch, [_face(340, 190, 660, 610)])

    cfg = FaceKeepConfig()
    cfg.aggressive.quality_target = 0.15
    cfg.aggressive.quality_scale_candidates = [0.125, 0.25, 0.5]
    cfg.aggressive.content_aware = False  # isolate the quality-target decision
    # 0.125 too high, 0.25 qualifies.
    _patch_lpips(monkeypatch, _scale_keyed_lpips([0.30, 0.10, 0.05]))

    photo = compress_photo(path, cfg)
    assert photo.effective_bg_scale == 0.25

    fkeep = tmp_path / "out.fkeep"
    write_fkeep(photo, str(fkeep))
    assert read_fkeep_info(str(fkeep))["settings"]["bg_scale"] == 0.25


def test_compress_quality_target_off_uses_fixed_scale(tmp_path, monkeypatch):
    """quality_target=None (default): the configured fixed bg_scale is used
    and LPIPS is never touched (anti-regression on the default path)."""
    path = _write_jpg(tmp_path, "fixed.jpg", _image(800, 1000))
    _patch_detector(monkeypatch, [_face(340, 190, 660, 610)])

    def _should_not_be_called(a, b):  # pragma: no cover
        raise AssertionError("LPIPS used despite quality_target=None")

    _patch_lpips(monkeypatch, _should_not_be_called, available=True)

    cfg = FaceKeepConfig()  # quality_target None, bg_scale 0.25
    cfg.aggressive.content_aware = False
    photo = compress_photo(path, cfg)
    assert photo.effective_bg_scale == 0.25


def test_compress_quality_target_respects_conservative_floor(tmp_path, monkeypatch):
    """Quality-targeting composes with content-aware: the search runs on top of
    the conservative floor and never picks a scale below it.

    A text-heavy photo raises the floor to conservative_bg_scale (0.5); even
    though the most aggressive candidate would qualify, the result stays >= 0.5.
    """
    from test_content_aware import _dense_text  # reuse the risky-content builder

    img = _dense_text(800, 1000)
    cv2.ellipse(img, (500, 400), (160, 210), 0, 0, 360, (180, 170, 165), -1)
    path = _write_jpg(tmp_path, "text.jpg", img)
    _patch_detector(monkeypatch, [_face(340, 190, 660, 610)])

    cfg = FaceKeepConfig()  # content_aware on -> floor becomes 0.5 on text
    cfg.aggressive.quality_target = 0.5
    cfg.aggressive.quality_scale_candidates = [0.125, 0.25, 0.5]
    # Only 0.5 survives the 0.5 floor filter; it qualifies.
    _patch_lpips(monkeypatch, _scale_keyed_lpips([0.10]))

    photo = compress_photo(path, cfg)
    assert photo.effective_bg_scale >= cfg.aggressive.conservative_bg_scale
    assert photo.effective_bg_scale == 0.5


# --------------------------------------------------------------------------- #
# C. Incremental-index fingerprint
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("mutate", [
    lambda c: setattr(c.aggressive, "quality_target", 0.2),
    lambda c: setattr(c.aggressive, "quality_scale_candidates", [0.2, 0.4]),
])
def test_fingerprint_busts_on_quality_target_change(mutate):
    base = FaceKeepConfig(mode="aggressive")
    changed = FaceKeepConfig(mode="aggressive")
    mutate(changed)
    assert settings_fingerprint(base) != settings_fingerprint(changed)


def test_faithful_fingerprint_unaffected_by_quality_target():
    base = FaceKeepConfig(mode="faithful")
    changed = FaceKeepConfig(mode="faithful")
    changed.aggressive.quality_target = 0.2
    changed.aggressive.quality_scale_candidates = [0.3]
    assert settings_fingerprint(base) == settings_fingerprint(changed)


# --------------------------------------------------------------------------- #
# D. Validation + YAML round-trip
# --------------------------------------------------------------------------- #

def test_validate_accepts_none_and_valid():
    FaceKeepConfig().validate()  # quality_target None by default
    cfg = FaceKeepConfig()
    cfg.aggressive.quality_target = 0.15
    cfg.aggressive.quality_scale_candidates = [0.1, 0.25, 0.5]
    cfg.validate()


@pytest.mark.parametrize("field,value", [
    ("quality_target", 0.0),       # must be > 0 when set
    ("quality_target", -0.1),
    ("quality_scale_candidates", [0.0]),   # below 0.05
    ("quality_scale_candidates", [1.5]),   # above 1.0
    ("quality_scale_candidates", [0.25, 2.0]),
])
def test_validate_rejects_out_of_range(field, value):
    cfg = FaceKeepConfig()
    setattr(cfg.aggressive, field, value)
    with pytest.raises(ConfigError):
        cfg.validate()


def test_yaml_round_trip_preserves_quality_fields(tmp_path):
    cfg = FaceKeepConfig()
    cfg.aggressive.quality_target = 0.12
    cfg.aggressive.quality_scale_candidates = [0.1, 0.2, 0.4]
    p = tmp_path / "facekeep.yaml"
    cfg.save(p)

    loaded = FaceKeepConfig.load(p)
    assert loaded.aggressive.quality_target == 0.12
    assert loaded.aggressive.quality_scale_candidates == [0.1, 0.2, 0.4]

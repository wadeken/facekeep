"""Detection caching — ROADMAP Phase 6.

Detection of an image under a fixed set of detector settings is deterministic,
so its result can be cached by ``(content hash, detector fingerprint)`` and
reused on a re-run. This is a *pure speed* feature, like the incremental index:
it must never change which faces a pipeline acts on or the output bytes. These
tests pin both halves of that contract:

* the fingerprint is stable for the same settings and busts on any change;
* the cache round-trips a FaceRegion list (tuples preserved), misses correctly,
  and survives a corrupt row / schema bump without crashing;
* ``detect_cached`` reuses a hit *without* calling the detector, records a miss,
  and is a plain passthrough when ``cache=None`` (the parallel / opt-out path);
* end-to-end (faithful and aggressive) a cached re-run reuses detection yet
  produces byte-identical output — the no-observable-effect guarantee.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest
from click.testing import CliRunner

from facekeep import detector as det
from facekeep import encoders
from facekeep.cli import cli
from facekeep.config import FaceKeepConfig
from facekeep.detector import (
    DetectionCache,
    FaceRegion,
    HaarDetector,
    create_detector,
    detect_cached,
    detector_fingerprint,
)

requires_avif = pytest.mark.skipif(
    not encoders.codec_available("avif"), reason="AVIF encoder not installed"
)


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #


def _make_photo(path: Path, seed: int = 3):
    """Write a compressible synthetic JPEG with a Haar-detectable face."""
    rng = np.random.default_rng(seed)
    H, W = 600, 800
    bg = cv2.resize(
        rng.normal(128, 25, (H // 10, W // 10, 3)).astype(np.float32),
        (W, H), interpolation=cv2.INTER_CUBIC,
    )
    img = np.clip(bg, 0, 255).astype(np.uint8)
    cx, cy, fw = 400, 300, 200
    fh = int(fw * 1.3)
    cv2.ellipse(img, (cx, cy), (fw // 2, fh // 2), 0, 0, 360, (180, 170, 165), -1)
    cv2.ellipse(img, (cx, cy - fh // 6), (fw // 2 - 5, fh // 4), 0, 0, 360,
                (195, 185, 180), -1)
    ew = fw // 7
    cv2.ellipse(img, (cx - fw // 5, cy - fh // 10), (ew, ew // 2), 0, 0, 360,
                (60, 55, 55), -1)
    cv2.ellipse(img, (cx + fw // 5, cy - fh // 10), (ew, ew // 2), 0, 0, 360,
                (60, 55, 55), -1)
    cv2.line(img, (cx, cy - fh // 10), (cx, cy + fh // 12), (150, 140, 135), 5)
    cv2.ellipse(img, (cx, cy + fh // 4), (fw // 5, fh // 18), 0, 0, 180,
                (120, 90, 90), -1)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])


def _sample_faces():
    return [
        FaceRegion(id=0, bbox=(10, 20, 110, 150),
                   padded_bbox=(0, 5, 130, 180), confidence=0.91),
        FaceRegion(id=1, bbox=(200, 60, 260, 140),
                   padded_bbox=(180, 40, 280, 160), confidence=1.0),
    ]


class _SpyDetector(det.FaceDetector):
    """A FaceDetector that records call count and returns a fixed list."""

    _backend = "haar"

    def __init__(self, faces):
        self._faces = faces
        self.calls = 0

    def detect(self, image):
        self.calls += 1
        return list(self._faces)


# --------------------------------------------------------------------------- #
# detector_fingerprint
# --------------------------------------------------------------------------- #


def _fp(**over):
    base = dict(backend="haar", confidence=0.6, padding=1.5, nms_iou=0.3,
                min_size_ratio=0.05, max_aspect_ratio=1.6, roi="face")
    base.update(over)
    return detector_fingerprint(**base)


def test_fingerprint_stable_for_same_settings():
    assert _fp() == _fp()


@pytest.mark.parametrize("field,value", [
    ("backend", "yunet"),
    ("confidence", 0.5),
    ("padding", 1.2),
    ("nms_iou", 0.5),
    ("min_size_ratio", 0.02),
    ("max_aspect_ratio", 2.0),
    ("roi", "person"),
])
def test_fingerprint_busts_on_each_field(field, value):
    assert _fp() != _fp(**{field: value})


def test_detector_fingerprint_matches_index_detector_fields():
    """The detector fingerprint covers exactly the fields index.py treats as
    output-affecting, so the two stay consistent."""
    d = create_detector(backend="haar")
    assert d.fingerprint() == detector_fingerprint(
        backend="haar", confidence=d.confidence, padding=d.padding,
        nms_iou=d.nms_iou, min_size_ratio=d.min_size_ratio,
        max_aspect_ratio=d.max_aspect_ratio, roi=d.roi,
    )


def test_detector_fingerprint_reflects_backend_and_roi():
    haar_face = create_detector(backend="haar", roi="face")
    haar_person = create_detector(backend="haar", roi="person")
    assert haar_face.fingerprint() != haar_person.fingerprint()
    # Haar pins confidence to 1.0 regardless of the (yunet-only) confidence arg,
    # so passing a different confidence to Haar does not change its fingerprint.
    haar_a = create_detector(backend="haar", confidence=0.6)
    haar_b = create_detector(backend="haar", confidence=0.3)
    assert haar_a.fingerprint() == haar_b.fingerprint()


# --------------------------------------------------------------------------- #
# DetectionCache round-trip / miss / resilience
# --------------------------------------------------------------------------- #


def test_cache_roundtrip_preserves_faceregions(tmp_path):
    faces = _sample_faces()
    with DetectionCache(tmp_path / "d.sqlite") as cache:
        assert cache.lookup("h1", "fp1") is None  # cold
        cache.record("h1", "fp1", faces)
        got = cache.lookup("h1", "fp1")
    assert got == faces  # equal, including tuple bboxes
    # bboxes must come back as tuples (the rest of the system reads them so)
    assert isinstance(got[0].bbox, tuple)
    assert isinstance(got[0].padded_bbox, tuple)


def test_cache_roundtrip_empty_list(tmp_path):
    with DetectionCache(tmp_path / "d.sqlite") as cache:
        cache.record("h", "fp", [])
        assert cache.lookup("h", "fp") == []  # a real hit, not a miss


def test_cache_miss_on_different_hash_or_fingerprint(tmp_path):
    faces = _sample_faces()
    with DetectionCache(tmp_path / "d.sqlite") as cache:
        cache.record("h1", "fp1", faces)
        assert cache.lookup("h2", "fp1") is None  # different content
        assert cache.lookup("h1", "fp2") is None  # different settings
        assert cache.lookup("h1", "fp1") == faces


def test_cache_record_overwrites(tmp_path):
    with DetectionCache(tmp_path / "d.sqlite") as cache:
        cache.record("h", "fp", _sample_faces())
        cache.record("h", "fp", [])  # re-detect with a newer result
        assert cache.lookup("h", "fp") == []


def test_cache_persists_across_connections(tmp_path):
    db = tmp_path / "d.sqlite"
    with DetectionCache(db) as cache:
        cache.record("h", "fp", _sample_faces())
    with DetectionCache(db) as cache:
        assert cache.lookup("h", "fp") == _sample_faces()


def test_cache_corrupt_row_is_a_miss(tmp_path):
    db = tmp_path / "d.sqlite"
    with DetectionCache(db) as cache:
        cache.record("h", "fp", _sample_faces())
        # Corrupt the stored JSON directly.
        cache._conn.execute(
            "UPDATE detections SET faces_json = ? WHERE content_hash = ?",
            ("not json", "h"),
        )
        cache._conn.commit()
        assert cache.lookup("h", "fp") is None  # swallowed, treated as miss


def test_cache_schema_bump_wipes_old_rows(tmp_path, monkeypatch):
    db = tmp_path / "d.sqlite"
    with DetectionCache(db) as cache:
        cache.record("h", "fp", _sample_faces())
    # Simulate an incompatible schema version on the next open.
    monkeypatch.setattr(det, "_DETECTION_SCHEMA_VERSION", 99)
    with DetectionCache(db) as cache:
        assert cache.lookup("h", "fp") is None  # wiped, not crashed


# --------------------------------------------------------------------------- #
# detect_cached
# --------------------------------------------------------------------------- #


def test_detect_cached_none_is_passthrough():
    spy = _SpyDetector(_sample_faces())
    img = np.zeros((10, 10, 3), np.uint8)
    out = detect_cached(spy, img, "h", None)
    assert out == _sample_faces()
    assert spy.calls == 1  # cache=None: just detect


def test_detect_cached_miss_then_hit(tmp_path):
    spy = _SpyDetector(_sample_faces())
    img = np.zeros((10, 10, 3), np.uint8)
    with DetectionCache(tmp_path / "d.sqlite") as cache:
        first = detect_cached(spy, img, "h", cache)
        assert spy.calls == 1  # miss -> detector ran
        second = detect_cached(spy, img, "h", cache)
        assert spy.calls == 1  # hit -> detector NOT called again
    assert first == second == _sample_faces()


def test_detect_cached_different_image_misses(tmp_path):
    spy = _SpyDetector(_sample_faces())
    img = np.zeros((10, 10, 3), np.uint8)
    with DetectionCache(tmp_path / "d.sqlite") as cache:
        detect_cached(spy, img, "h1", cache)
        detect_cached(spy, img, "h2", cache)  # different content hash
    assert spy.calls == 2


def test_detect_cached_different_fingerprint_misses(tmp_path):
    """Same image+hash but a different detector fingerprint must miss, so a
    settings change is never served a stale detection."""
    img = np.zeros((10, 10, 3), np.uint8)
    face_spy = _SpyDetector(_sample_faces())
    person_spy = _SpyDetector(_sample_faces())
    # Give them different effective settings so their fingerprints differ.
    face_spy.roi = "face"
    person_spy.roi = "person"
    with DetectionCache(tmp_path / "d.sqlite") as cache:
        detect_cached(face_spy, img, "h", cache)
        detect_cached(person_spy, img, "h", cache)
    assert face_spy.calls == 1
    assert person_spy.calls == 1  # different fingerprint -> did not hit face's row


# --------------------------------------------------------------------------- #
# End-to-end: no observable effect (byte-identical output)
# --------------------------------------------------------------------------- #


@requires_avif
def test_faithful_cache_byte_identical_and_reuses_detection(tmp_path, monkeypatch):
    from facekeep import faithful

    src = tmp_path / "a.jpg"
    _make_photo(src)
    cfg = FaceKeepConfig()
    cfg.faithful.auto_tune = False  # deterministic, fast

    # Reference: no cache.
    ref = faithful.compress(str(src), str(tmp_path / "ref"), cfg)
    ref_bytes = ref.output_path.read_bytes()

    # Spy on the real detector to prove the 2nd cached run skips detection.
    calls = {"n": 0}
    real_detect = HaarDetector.detect

    def _spy(self, image):
        calls["n"] += 1
        return real_detect(self, image)

    monkeypatch.setattr(HaarDetector, "detect", _spy)

    with DetectionCache(tmp_path / "d.sqlite") as cache:
        r1 = faithful.compress(str(src), str(tmp_path / "o1"), cfg,
                               detection_cache=cache)
        r2 = faithful.compress(str(src), str(tmp_path / "o2"), cfg,
                               detection_cache=cache)

    assert calls["n"] == 1  # detection ran once; second run hit the cache
    assert r1.faces_detected == r2.faces_detected == ref.faces_detected
    # Pure speed feature: cached output is byte-identical to the no-cache run.
    assert r1.output_path.read_bytes() == ref_bytes
    assert r2.output_path.read_bytes() == ref_bytes


def test_aggressive_cache_reuses_detection(tmp_path, monkeypatch):
    from facekeep.aggressive import compressor

    src = tmp_path / "a.jpg"
    _make_photo(src)
    cfg = FaceKeepConfig()
    cfg.mode = "aggressive"

    calls = {"n": 0}
    real_detect = HaarDetector.detect

    def _spy(self, image):
        calls["n"] += 1
        return real_detect(self, image)

    monkeypatch.setattr(HaarDetector, "detect", _spy)

    with DetectionCache(tmp_path / "d.sqlite") as cache:
        p1 = compressor.compress_photo(str(src), cfg, detection_cache=cache)
        p2 = compressor.compress_photo(str(src), cfg, detection_cache=cache)

    assert calls["n"] == 1  # second run reused the cached detection
    assert len(p1.faces) == len(p2.faces)
    assert len(p1.faces) >= 1  # the synthetic face is found


# --------------------------------------------------------------------------- #
# CLI flag
# --------------------------------------------------------------------------- #


@requires_avif
def test_cli_no_detect_cache_flag_runs(tmp_path):
    src = tmp_path / "a.jpg"
    _make_photo(src)
    runner = CliRunner()
    result = runner.invoke(cli, ["compress", str(src), "--no-detect-cache",
                                 "--no-index", "-o", str(tmp_path / "out")])
    assert result.exit_code == 0, result.output

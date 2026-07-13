"""iPhone HDR gain-map preservation through aggressive mode (ROADMAP 9.2).

Compress stores the gain map ``imageio.load`` extracted (9.1) as a
``gainmap.jpg`` member + a ``gain_map_preserved`` manifest flag (schema
1.10.0, hdrgm params 1.11.0); ``verify_fkeep`` requires a *declared* gain map to decode; restore
re-attaches it into a backward-compatible HDR AVIF via the external
``avifgainmaputil`` binary when the output is ``.avif`` and the binary is
locatable, or (ROADMAP 9.3) into an **Ultra HDR JPEG** via pure Pillow when
the output is ``.jpg``/``.jpeg`` (the default — no binary needed) — every
other combination falls back to the normal SDR write with a warning
(offline-first, never a hard fail).

Gating mirrors the high-bit tests: the real HDR re-attach test skips unless
``avifgainmaputil`` is locatable (set ``FACEKEEP_AVIFENC`` — the sibling lookup
finds the whole libavif tool family). Everything else is offline/synthetic;
conftest's autouse fixture keeps restore on the bicubic no-AI path.
"""

import json
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from facekeep import encoders
from facekeep.aggressive.compressor import compress_photo
from facekeep.aggressive.format import read_fkeep, verify_fkeep, write_fkeep
from facekeep.aggressive.restorer import Restorer
from facekeep.config import ConfigError, FaceKeepConfig
from facekeep.exceptions import EncodingError
from facekeep.index import settings_fingerprint

APPLE_GAINMAP_XMP = (
    b'<x:xmpmeta xmlns:x="adobe:ns:meta/" '
    b'xmlns:HDRGainMap="http://ns.apple.com/HDRGainMap/1.0/">'
    b"<apdi:AuxiliaryImageType>urn:com:apple:photo:2020:aux:hdrgainmap"
    b"</apdi:AuxiliaryImageType></x:xmpmeta>"
)

GAIN_MAP = np.tile(np.linspace(0, 200, 128, dtype=np.uint8), (96, 1))


def _write_gainmap_jpeg(path: Path) -> np.ndarray:
    """Author an iPhone-style HDR JPEG: MPF second frame + gain-map XMP.

    Returns the authored gain-map array (what the loader should extract).
    """
    rng = np.random.default_rng(7)
    base = rng.integers(30, 220, (192, 256, 3), np.uint8).astype(np.uint8)
    gm = Image.fromarray(GAIN_MAP, "L")
    gm.encoderinfo = {"xmp": APPLE_GAINMAP_XMP}
    Image.fromarray(base, "RGB").save(
        str(path), format="MPO", save_all=True, append_images=[gm]
    )
    return GAIN_MAP


def _compress_to_fkeep(src: Path, tmp_path: Path, **agg_overrides) -> Path:
    cfg = FaceKeepConfig(mode="aggressive")
    for k, v in agg_overrides.items():
        setattr(cfg.aggressive, k, v)
    photo = compress_photo(str(src), cfg)
    fkeep = tmp_path / (src.stem + ".fkeep")
    write_fkeep(photo, str(fkeep))
    return fkeep


@pytest.fixture()
def hdr_jpeg(tmp_path):
    src = tmp_path / "hdr.jpg"
    _write_gainmap_jpeg(src)
    return src


# ------------------------------------------------------------- compress side


def test_fkeep_stores_gain_map(hdr_jpeg, tmp_path):
    """A gain-map source produces gainmap.jpg + the manifest flag (schema now 1.11.0)."""
    fkeep = _compress_to_fkeep(hdr_jpeg, tmp_path)

    with zipfile.ZipFile(fkeep) as zf:
        names = set(zf.namelist())
        manifest = json.loads(zf.read("manifest.json"))
    assert "gainmap.jpg" in names
    assert manifest["gain_map_preserved"] is True
    assert manifest["version"] == "1.11.0"


def test_gain_map_roundtrip_values(hdr_jpeg, tmp_path):
    """read_fkeep returns the stored gain map close to the source's (JPEG-lossy)."""
    fkeep = _compress_to_fkeep(hdr_jpeg, tmp_path)

    data = read_fkeep(str(fkeep))
    gm = data["gain_map"]
    assert gm is not None
    assert gm.ndim == 2  # grayscale member stays single-channel
    assert gm.shape == GAIN_MAP.shape
    assert float(np.mean(np.abs(gm.astype(int) - GAIN_MAP.astype(int)))) < 4.0


def test_no_gain_map_source_has_no_member(tmp_path):
    """A plain JPEG packs no gainmap member and a False flag."""
    import cv2

    src = tmp_path / "plain.jpg"
    cv2.imwrite(str(src), np.full((192, 256, 3), 120, np.uint8))
    fkeep = _compress_to_fkeep(src, tmp_path)

    with zipfile.ZipFile(fkeep) as zf:
        names = set(zf.namelist())
        manifest = json.loads(zf.read("manifest.json"))
    assert "gainmap.jpg" not in names
    assert manifest["gain_map_preserved"] is False


def test_preserve_gain_map_opt_out(hdr_jpeg, tmp_path):
    """preserve_gain_map=False drops the member even on a gain-map source."""
    fkeep = _compress_to_fkeep(hdr_jpeg, tmp_path, preserve_gain_map=False)

    with zipfile.ZipFile(fkeep) as zf:
        names = set(zf.namelist())
        manifest = json.loads(zf.read("manifest.json"))
    assert "gainmap.jpg" not in names
    assert manifest["gain_map_preserved"] is False


def test_dry_run_size_matches_real(hdr_jpeg, tmp_path):
    """The gain-map member packs identically in dry-run and real writes."""
    cfg = FaceKeepConfig(mode="aggressive")
    photo = compress_photo(str(hdr_jpeg), cfg)
    estimated = write_fkeep(photo, str(tmp_path / "a.fkeep"), dry_run=True)
    real = write_fkeep(photo, str(tmp_path / "a.fkeep"))
    assert estimated == real


# ---------------------------------------------------------------- verify side


def test_verify_ok_with_gain_map(hdr_jpeg, tmp_path):
    fkeep = _compress_to_fkeep(hdr_jpeg, tmp_path)

    report = verify_fkeep(str(fkeep))
    assert report.ok
    assert report.gain_map_declared is True
    assert report.gain_map_ok is True


def test_verify_flags_missing_gain_map_member(hdr_jpeg, tmp_path):
    """A declared-but-missing gainmap.jpg is a reported problem, not a crash."""
    fkeep = _compress_to_fkeep(hdr_jpeg, tmp_path)
    tampered = tmp_path / "tampered.fkeep"
    with zipfile.ZipFile(fkeep) as src, zipfile.ZipFile(tampered, "w") as dst:
        for item in src.infolist():
            if item.filename != "gainmap.jpg":
                dst.writestr(item, src.read(item.filename))

    report = verify_fkeep(str(tampered))
    assert not report.ok
    assert report.gain_map_declared is True
    assert report.gain_map_ok is False
    assert any("gainmap.jpg" in p for p in report.problems)


def test_verify_gain_map_less_file_unchanged(tmp_path):
    import cv2

    src = tmp_path / "plain.jpg"
    cv2.imwrite(str(src), np.full((192, 256, 3), 120, np.uint8))
    fkeep = _compress_to_fkeep(src, tmp_path)

    report = verify_fkeep(str(fkeep))
    assert report.ok
    assert report.gain_map_declared is False
    assert report.gain_map_ok is False


# ---------------------------------------------------- config and fingerprint


def test_validate_headroom_range():
    ok = FaceKeepConfig()
    ok.aggressive.gain_map_headroom = 3.5
    ok.validate()
    for bad in (0.0, -1.0, 7.0):
        cfg = FaceKeepConfig()
        cfg.aggressive.gain_map_headroom = bad
        with pytest.raises(ConfigError):
            cfg.validate()


def test_fingerprint_busts_on_preserve_gain_map():
    base = FaceKeepConfig(mode="aggressive")
    changed = FaceKeepConfig(mode="aggressive")
    changed.aggressive.preserve_gain_map = False
    assert settings_fingerprint(base) != settings_fingerprint(changed)


def test_fingerprint_ignores_restore_only_headroom():
    base = FaceKeepConfig(mode="aggressive")
    changed = FaceKeepConfig(mode="aggressive")
    changed.aggressive.gain_map_headroom = 2.0
    assert settings_fingerprint(base) == settings_fingerprint(changed)


def test_faithful_fingerprint_ignores_gain_map_fields():
    base = FaceKeepConfig()  # faithful
    changed = FaceKeepConfig()
    changed.aggressive.preserve_gain_map = False
    assert settings_fingerprint(base) == settings_fingerprint(changed)


# ---------------------------------------------------------------- restore side


def test_restore_jpg_carries_gain_map(hdr_jpeg, tmp_path, caplog):
    """The default .jpg restore output is an Ultra HDR JPEG (ROADMAP 9.3):
    the gain map rides as the MPF second frame, no warning, and FaceKeep's own
    loader re-extracts it (self-round-trip)."""
    from facekeep import imageio

    fkeep = _compress_to_fkeep(hdr_jpeg, tmp_path)
    out = tmp_path / "restored.jpg"

    with caplog.at_level("WARNING", logger="facekeep.aggressive.restorer"):
        Restorer(FaceKeepConfig().aggressive).restore(str(fkeep), str(out))

    assert out.exists() and out.stat().st_size > 0
    assert not any("gain map" in r.message.lower() for r in caplog.records)
    loaded = imageio.load(str(out))
    assert loaded.gain_map is not None
    assert loaded.gain_map_meta["source"] == "jpeg-mpf"
    # The stored member bytes ride verbatim, so the values match read_fkeep's
    # decode of gainmap.jpg exactly (same JPEG, decoded twice).
    stored = read_fkeep(str(fkeep))["gain_map"]
    gm = loaded.gain_map
    assert gm.shape == stored.shape
    assert np.array_equal(gm, stored)
    # The full Ultra HDR flavor: hdrgm + GContainer on the primary (what the
    # user-validated Chrome render keys on).
    raw = out.read_bytes()
    assert b'hdrgm:Version="1.0"' in raw
    assert b"Item:Semantic=\"GainMap\"" in raw


def test_restore_png_warns_and_writes_sdr(hdr_jpeg, tmp_path, caplog):
    """A format that can't carry the gain map (.png) still warns + writes SDR."""
    fkeep = _compress_to_fkeep(hdr_jpeg, tmp_path)
    out = tmp_path / "restored.png"

    with caplog.at_level("WARNING", logger="facekeep.aggressive.restorer"):
        Restorer(FaceKeepConfig().aggressive).restore(str(fkeep), str(out))

    assert out.exists() and out.stat().st_size > 0
    assert any("gain map" in r.message.lower() for r in caplog.records)


def test_restore_jpg_authoring_failure_falls_back_sdr(
    hdr_jpeg, tmp_path, caplog, monkeypatch
):
    """An Ultra HDR authoring failure degrades to the plain SDR JPEG write."""
    fkeep = _compress_to_fkeep(hdr_jpeg, tmp_path)
    out = tmp_path / "restored.jpg"

    def boom(*a, **k):
        raise EncodingError("synthetic authoring failure")

    monkeypatch.setattr(encoders, "encode_gainmap_jpeg", boom)
    with caplog.at_level("WARNING", logger="facekeep.aggressive.restorer"):
        Restorer(FaceKeepConfig().aggressive).restore(str(fkeep), str(out))

    assert out.exists() and out.stat().st_size > 0
    assert any("gain-map JPEG authoring failed" in r.message
               for r in caplog.records)
    # The fallback is a plain SDR JPEG: no MPF second frame.
    with Image.open(out) as pil:
        assert pil.format == "JPEG"
        assert getattr(pil, "n_frames", 1) == 1


def test_restore_avif_without_binary_falls_back_sdr(
    hdr_jpeg, tmp_path, caplog, monkeypatch
):
    """.avif output without avifgainmaputil degrades to a plain SDR AVIF."""
    fkeep = _compress_to_fkeep(hdr_jpeg, tmp_path)
    out = tmp_path / "restored.avif"
    monkeypatch.setattr(encoders, "avifgainmaputil_available", lambda: False)

    with caplog.at_level("WARNING", logger="facekeep.aggressive.restorer"):
        Restorer(FaceKeepConfig().aggressive).restore(str(fkeep), str(out))

    assert out.exists() and out.stat().st_size > 0
    assert any("avifgainmaputil" in r.message for r in caplog.records)
    # The fallback is a real (SDR) AVIF the normal decoder can open.
    decoded = encoders.decode(out.read_bytes())
    assert decoded.shape[:2] == (192, 256)


def test_preview_skips_gain_map_silently(hdr_jpeg, tmp_path, caplog):
    """preview() never re-attaches (speed) and never warns about it either."""
    fkeep = _compress_to_fkeep(hdr_jpeg, tmp_path)
    out = tmp_path / "preview.jpg"

    with caplog.at_level("WARNING", logger="facekeep.aggressive.restorer"):
        Restorer(FaceKeepConfig().aggressive).preview(str(fkeep), str(out))

    assert out.exists() and out.stat().st_size > 0
    assert not any("gain map" in r.message.lower() for r in caplog.records)


def test_encode_gainmap_avif_raises_without_binary(monkeypatch):
    monkeypatch.setattr(encoders, "_find_avifgainmaputil", lambda: None)
    with pytest.raises(EncodingError):
        encoders.encode_gainmap_avif(
            np.zeros((16, 16, 3), np.uint8), np.zeros((8, 8), np.uint8)
        )


@pytest.mark.skipif(
    not encoders.avifgainmaputil_available(),
    reason="avifgainmaputil binary not found (set FACEKEEP_AVIFENC or put the "
    "libavif tools on PATH)",
)
def test_restore_avif_reattaches_gain_map(hdr_jpeg, tmp_path):
    """The real HDR path: restore -f avif emits an AVIF with an embedded gain map."""
    fkeep = _compress_to_fkeep(hdr_jpeg, tmp_path)
    out = tmp_path / "restored_hdr.avif"

    Restorer(FaceKeepConfig().aggressive).restore(str(fkeep), str(out), quality=80)

    assert out.exists() and out.stat().st_size > 0
    binary = encoders._find_avifgainmaputil()
    proc = subprocess.run(
        [binary, "printmetadata", str(out)],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    # printmetadata only succeeds on a gain-map AVIF; sanity-check the fields.
    assert "Alternate headroom" in proc.stdout
    assert "Gain Map Max" in proc.stdout

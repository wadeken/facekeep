"""Faithful-mode HDR gain-map carry (ROADMAP 9.6).

A real phone HDR photo is an 8-bit SDR base plus an HDR *gain map*; faithful
mode used to keep only the base (SDR output). Now a gain-map-bearing source
whose concrete output codec is AVIF is written as a backward-compatible
**gain-map (HDR) AVIF** via ``encoders.encode_gainmap_avif`` (the external
``avifgainmaputil combine``); every other combination — JXL/WebP output,
lossless mode, a deep-color uint16 source, a missing binary, or
``faithful.preserve_gain_map: false`` — keeps today's SDR bytes (warned where
HDR is really lost, silent where the user opted out or has no map).

Gating mirrors the aggressive gain-map tests: the real combine round-trip
skips unless ``avifgainmaputil`` is locatable (set ``FACEKEEP_AVIFENC`` — the
sibling lookup finds the whole libavif tool family). Everything else is
offline/synthetic.
"""

import subprocess

import numpy as np
import pytest
from PIL import Image

from facekeep import encoders, faithful, imageio
from facekeep.config import ConfigError, FaceKeepConfig
from facekeep.index import settings_fingerprint

APPLE_GAINMAP_XMP = (
    b'<x:xmpmeta xmlns:x="adobe:ns:meta/" '
    b'xmlns:HDRGainMap="http://ns.apple.com/HDRGainMap/1.0/">'
    b"<apdi:AuxiliaryImageType>urn:com:apple:photo:2020:aux:hdrgainmap"
    b"</apdi:AuxiliaryImageType></x:xmpmeta>"
)

GAIN_MAP = np.tile(np.linspace(0, 200, 64, dtype=np.uint8), (48, 1))


def _smooth_base(h=96, w=128) -> np.ndarray:
    """A compressible RGB gradient so AVIF beats the JPEG source comfortably."""
    x = np.linspace(40, 210, w, dtype=np.uint8)
    base = np.tile(x, (h, 1))
    return np.dstack([base, base[::-1], np.full((h, w), 90, np.uint8)])


def _write_gainmap_jpeg(path) -> None:
    """Author an iPhone-style HDR JPEG: MPF second frame + gain-map XMP."""
    gm = Image.fromarray(GAIN_MAP, "L")
    gm.encoderinfo = {"xmp": APPLE_GAINMAP_XMP}
    Image.fromarray(_smooth_base(), "RGB").save(
        str(path), format="MPO", save_all=True, append_images=[gm]
    )


def _write_plain_jpeg(path) -> None:
    Image.fromarray(_smooth_base(), "RGB").save(str(path), quality=90)


@pytest.fixture()
def hdr_jpeg(tmp_path):
    src = tmp_path / "hdr.jpg"
    _write_gainmap_jpeg(src)
    return src


def _cfg(**faithful_overrides) -> FaceKeepConfig:
    cfg = FaceKeepConfig()
    for k, v in faithful_overrides.items():
        setattr(cfg.faithful, k, v)
    return cfg


# ------------------------------------------------------- degraded paths (offline)


def test_mapless_source_never_touches_the_hdr_path(tmp_path, monkeypatch):
    """A source with no gain map never calls the combine encoder (pin: the
    default path is untouched — byte-identical output by construction)."""
    src = tmp_path / "plain.jpg"
    _write_plain_jpeg(src)

    def _boom(*a, **k):  # pragma: no cover - the assert is that it never runs
        raise AssertionError("encode_gainmap_avif must not be called")

    monkeypatch.setattr(encoders, "encode_gainmap_avif", _boom)
    monkeypatch.setattr(encoders, "avifgainmaputil_available", lambda: True)

    res = faithful.compress(str(src), str(tmp_path / "out"), _cfg())

    assert res.gain_map_carried is False
    assert res.output_path.suffix == ".avif"


def test_gain_map_without_binary_falls_back_sdr(hdr_jpeg, tmp_path, caplog, monkeypatch):
    """No avifgainmaputil -> today's SDR AVIF + a warning (offline-first)."""
    monkeypatch.setattr(encoders, "avifgainmaputil_available", lambda: False)

    with caplog.at_level("WARNING", logger="facekeep.faithful"):
        res = faithful.compress(str(hdr_jpeg), str(tmp_path / "out"), _cfg())

    assert res.gain_map_carried is False
    assert any("avifgainmaputil" in r.message for r in caplog.records)
    # The fallback is a real SDR AVIF the normal decoder opens at full size.
    decoded = encoders.decode(res.output_path.read_bytes())
    assert decoded.shape[:2] == (96, 128)


def test_non_avif_codec_warns_and_stays_sdr(hdr_jpeg, tmp_path, caplog):
    """A JXL output cannot carry the map -> SDR + a warning naming avif."""
    with caplog.at_level("WARNING", logger="facekeep.faithful"):
        res = faithful.compress(
            str(hdr_jpeg), str(tmp_path / "out"), _cfg(codec="jxl")
        )

    assert res.gain_map_carried is False
    assert res.output_path.suffix == ".jxl"
    assert any("only an AVIF output" in r.message for r in caplog.records)


def test_preserve_gain_map_off_is_silent(hdr_jpeg, tmp_path, caplog, monkeypatch):
    """Opting out is not a degradation: no gain-map warning at all."""
    monkeypatch.setattr(encoders, "avifgainmaputil_available", lambda: True)

    with caplog.at_level("WARNING", logger="facekeep.faithful"):
        res = faithful.compress(
            str(hdr_jpeg), str(tmp_path / "out"), _cfg(preserve_gain_map=False)
        )

    assert res.gain_map_carried is False
    assert not any("gain map" in r.message for r in caplog.records)


def test_lossless_keeps_bit_exact_promise(hdr_jpeg, tmp_path, caplog):
    """Lossless mode never re-encodes its base for HDR -> SDR + a warning."""
    with caplog.at_level("WARNING", logger="facekeep.faithful"):
        res = faithful.compress(
            str(hdr_jpeg), str(tmp_path / "out"),
            _cfg(codec="jxl", lossless=True),
        )

    assert res.gain_map_carried is False
    assert any("lossless" in r.message for r in caplog.records)


def test_highbit_source_keeps_deepcolor_path(tmp_path, caplog, monkeypatch):
    """A uint16 source with a gain map keeps the 10/12-bit path (warned)."""
    src = tmp_path / "deep.png"
    _write_plain_jpeg(src)  # content irrelevant; load is faked below

    loaded = imageio.LoadedImage(
        image=(np.random.default_rng(3).integers(
            0, 65535, (32, 40, 3), np.uint16
        ).astype(np.uint16)),
        exif=None, original_orientation=1, width=40, height=32,
        source_bit_depth=16,
        gain_map=np.full((16, 20), 128, np.uint8),
        gain_map_meta={"source": "test"},
    )
    monkeypatch.setattr(imageio, "load", lambda *a, **k: loaded)
    monkeypatch.setattr(encoders, "avifgainmaputil_available", lambda: True)

    with caplog.at_level("WARNING", logger="facekeep.faithful"):
        res = faithful.compress(
            str(src), str(tmp_path / "out"), _cfg(auto_tune=False)
        )

    assert res.gain_map_carried is False
    assert any("deep-color" in r.message for r in caplog.records)


def test_carry_failure_falls_back_sdr(hdr_jpeg, tmp_path, caplog, monkeypatch):
    """A combine failure degrades to the SDR bytes — never a failed compress."""
    monkeypatch.setattr(encoders, "avifgainmaputil_available", lambda: True)

    def _fail(*a, **k):
        raise encoders.EncodingError("combine exploded")

    monkeypatch.setattr(encoders, "encode_gainmap_avif", _fail)

    with caplog.at_level("WARNING", logger="facekeep.faithful"):
        res = faithful.compress(str(hdr_jpeg), str(tmp_path / "out"), _cfg())

    assert res.gain_map_carried is False
    assert any("carry failed" in r.message for r in caplog.records)
    decoded = encoders.decode(res.output_path.read_bytes())
    assert decoded.shape[:2] == (96, 128)


def test_skip_if_larger_sees_the_final_hdr_size(hdr_jpeg, tmp_path, monkeypatch):
    """The size decision runs on the combined (HDR) bytes: an HDR file bigger
    than the source keeps the original — which still carries its own map."""
    monkeypatch.setattr(encoders, "avifgainmaputil_available", lambda: True)
    huge = b"x" * (hdr_jpeg.stat().st_size + 4096)
    monkeypatch.setattr(encoders, "encode_gainmap_avif", lambda *a, **k: huge)

    # verify=False: the faked combine bytes aren't decodable, and this test
    # pins the size decision, not the round-trip check.
    res = faithful.compress(
        str(hdr_jpeg), str(tmp_path / "out"), _cfg(verify=False)
    )

    assert res.skipped is True
    assert res.gain_map_carried is False
    assert res.output_path.read_bytes() == hdr_jpeg.read_bytes()


# ----------------------------------------------------------- config / fingerprint


def test_fingerprint_busts_on_both_fields():
    base = settings_fingerprint(FaceKeepConfig())
    off = FaceKeepConfig()
    off.faithful.preserve_gain_map = False
    tuned = FaceKeepConfig()
    tuned.faithful.gain_map_headroom = 2.5
    assert settings_fingerprint(off) != base
    assert settings_fingerprint(tuned) != base
    assert settings_fingerprint(off) != settings_fingerprint(tuned)


def test_validate_rejects_bad_headroom():
    for bad in (0, -1.0, 6.5):
        cfg = FaceKeepConfig()
        cfg.faithful.gain_map_headroom = bad
        with pytest.raises(ConfigError):
            cfg.validate()


# ------------------------------------------------------------------ real binary


@pytest.mark.skipif(
    not encoders.avifgainmaputil_available(),
    reason="avifgainmaputil binary not found (set FACEKEEP_AVIFENC or put the "
    "libavif tools on PATH)",
)
def test_faithful_avif_carries_gain_map(hdr_jpeg, tmp_path):
    """The real HDR path: a faithful AVIF output embeds the source's gain map."""
    res = faithful.compress(
        str(hdr_jpeg), str(tmp_path / "out"), _cfg(skip_if_larger=False)
    )

    assert res.gain_map_carried is True
    assert res.output_path.suffix == ".avif"
    # The base decodes at full size through the normal (Pillow) decoder, so
    # verify_roundtrip ran against the real output bytes.
    decoded = encoders.decode(res.output_path.read_bytes())
    assert decoded.shape[:2] == (96, 128)
    # printmetadata only succeeds on a gain-map AVIF; sanity-check the fields.
    binary = encoders._find_avifgainmaputil()
    proc = subprocess.run(
        [binary, "printmetadata", str(res.output_path)],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Alternate headroom" in proc.stdout
    assert "Gain Map Max" in proc.stdout


@pytest.mark.skipif(
    not encoders.avifgainmaputil_available(),
    reason="avifgainmaputil binary not found (set FACEKEEP_AVIFENC or put the "
    "libavif tools on PATH)",
)
def test_dry_run_reports_hdr_without_writing(hdr_jpeg, tmp_path):
    """A dry run computes the real (combined) size + carried flag, writes nothing."""
    out = tmp_path / "out"
    res = faithful.compress(
        str(hdr_jpeg), str(out), _cfg(skip_if_larger=False), dry_run=True
    )

    assert res.gain_map_carried is True
    assert not res.output_path.exists()
    assert res.compressed_size > 0

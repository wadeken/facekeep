"""Per-image codec choice (faithful mode `codec="both"`).

ROADMAP Phase 5: with `codec: both`, faithful mode trial-encodes each image with
*both* AVIF and JXL (each at its own auto-tuned/configured quality) and keeps the
smaller output. These tests pin:

* config: `validate()` accepts "both", rejects unknown codecs;
* the pure selection logic (`faithful._encode_best_codec`) — smaller-wins, the
  reported codec is the concrete winner, single-codec is unchanged, and graceful
  degradation when one/both plugins are missing — driven with a *fake* encoder so
  the choice is deterministic and offline (no dependence on real codec sizes);
* both + auto-tune runs the search once per codec;
* the index fingerprint busts on "both" vs a single codec;
* an end-to-end CLI run with the real plugins (skipped if either is absent).
"""

import numpy as np
import pytest
from click.testing import CliRunner

from facekeep import encoders, faithful, imageio
from facekeep.cli import cli
from facekeep.config import FaceKeepConfig
from facekeep.exceptions import ConfigError, EncodingError
from facekeep.index import settings_fingerprint


# --------------------------------------------------------------------------- #
# config.validate()
# --------------------------------------------------------------------------- #

def test_validate_accepts_both():
    cfg = FaceKeepConfig()
    cfg.faithful.codec = "both"
    cfg.validate()  # must not raise


@pytest.mark.parametrize("codec", ["avif", "jxl", "webp", "both"])
def test_validate_accepts_known_codecs(codec):
    cfg = FaceKeepConfig()
    cfg.faithful.codec = codec
    cfg.validate()


def test_validate_rejects_unknown_codec():
    cfg = FaceKeepConfig()
    cfg.faithful.codec = "heic"  # not a faithful output codec
    with pytest.raises(ConfigError):
        cfg.validate()


# --------------------------------------------------------------------------- #
# _encode_best_codec — pure selection logic with a fake encoder
# --------------------------------------------------------------------------- #

def _cfg(codec="both", auto_tune=False):
    cfg = FaceKeepConfig()
    cfg.faithful.codec = codec
    cfg.faithful.auto_tune = auto_tune
    return cfg.faithful


def _fake_encode_factory(sizes, calls=None):
    """Return a fake ``encoders.encode`` that yields ``sizes[codec]`` bytes.

    The returned bytes encode the codec name (so a test can assert *which* bytes
    won), padded to the requested size. Records each call's codec in ``calls``.
    """
    def fake_encode(image, codec, quality, speed, chroma, has_faces,
                    exif=None, icc=None, bit_depth=8, output_bit_depth=10):
        if calls is not None:
            calls.append(codec)
        tag = codec.encode()
        return tag + b"\x00" * (max(sizes[codec], len(tag)) - len(tag))
    return fake_encode


def test_both_picks_smaller_codec(monkeypatch):
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    # jxl smaller -> jxl wins.
    monkeypatch.setattr(encoders, "codec_available", lambda c: c in ("avif", "jxl"))
    monkeypatch.setattr(
        encoders, "encode", _fake_encode_factory({"avif": 500, "jxl": 300})
    )
    data, q, codec_used = faithful._encode_best_codec(
        img, [], _cfg("both"), False, None, None
    )
    assert codec_used == "jxl"
    assert len(data) == 300
    assert data.startswith(b"jxl")


def test_both_picks_avif_when_smaller(monkeypatch):
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    monkeypatch.setattr(encoders, "codec_available", lambda c: c in ("avif", "jxl"))
    monkeypatch.setattr(
        encoders, "encode", _fake_encode_factory({"avif": 200, "jxl": 350})
    )
    data, q, codec_used = faithful._encode_best_codec(
        img, [], _cfg("both"), False, None, None
    )
    assert codec_used == "avif"
    assert len(data) == 200
    assert data.startswith(b"avif")


def test_both_trial_encodes_both_codecs(monkeypatch):
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    calls = []
    monkeypatch.setattr(encoders, "codec_available", lambda c: c in ("avif", "jxl"))
    monkeypatch.setattr(
        encoders, "encode", _fake_encode_factory({"avif": 200, "jxl": 350}, calls)
    )
    faithful._encode_best_codec(img, [], _cfg("both"), False, None, None)
    assert set(calls) == {"avif", "jxl"}  # both were actually tried


def test_single_codec_unchanged(monkeypatch):
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    calls = []
    monkeypatch.setattr(encoders, "codec_available", lambda c: c in ("avif", "jxl"))
    monkeypatch.setattr(
        encoders, "encode", _fake_encode_factory({"avif": 200, "jxl": 350}, calls)
    )
    data, q, codec_used = faithful._encode_best_codec(
        img, [], _cfg("avif"), False, None, None
    )
    assert codec_used == "avif"
    assert calls == ["avif"]  # only the one codec, no trial of the other


def test_both_falls_back_to_only_available_codec(monkeypatch):
    """codec=both but only jxl installed -> use jxl, warn, no avif attempt."""
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    calls = []
    monkeypatch.setattr(encoders, "codec_available", lambda c: c == "jxl")
    monkeypatch.setattr(
        encoders, "encode", _fake_encode_factory({"avif": 100, "jxl": 999}, calls)
    )
    data, q, codec_used = faithful._encode_best_codec(
        img, [], _cfg("both"), False, None, None
    )
    assert codec_used == "jxl"  # not avif, even though avif's fake size is smaller
    assert calls == ["jxl"]


def test_both_with_neither_codec_raises(monkeypatch):
    """codec=both with no plugins -> defer to encode(), which raises EncodingError."""
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    monkeypatch.setattr(encoders, "codec_available", lambda c: False)

    def boom(*a, **k):
        raise EncodingError("no codec")

    monkeypatch.setattr(encoders, "encode", boom)
    with pytest.raises(EncodingError):
        faithful._encode_best_codec(img, [], _cfg("both"), False, None, None)


# --------------------------------------------------------------------------- #
# both + auto-tune: the search runs once per codec
# --------------------------------------------------------------------------- #

def test_both_runs_autotune_per_codec(monkeypatch):
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    seen = []

    def fake_auto_tune(image, faces, cfg, has_faces, codec, exif=None, icc=None,
                       bit_depth=8):
        seen.append(codec)
        size = {"avif": 400, "jxl": 250}[codec]
        return codec.encode() + b"\x00" * (size - len(codec)), 71

    monkeypatch.setattr(encoders, "codec_available", lambda c: c in ("avif", "jxl"))
    monkeypatch.setattr(faithful, "_auto_tune_quality", fake_auto_tune)

    data, q, codec_used = faithful._encode_best_codec(
        img, [], _cfg("both", auto_tune=True), False, None, None
    )
    assert sorted(seen) == ["avif", "jxl"]  # auto-tune ran for each codec
    assert codec_used == "jxl"  # the smaller one
    assert q == 71


# --------------------------------------------------------------------------- #
# index fingerprint
# --------------------------------------------------------------------------- #

def test_fingerprint_busts_on_both():
    avif = FaceKeepConfig()
    both = FaceKeepConfig()
    both.faithful.codec = "both"
    assert settings_fingerprint(avif) != settings_fingerprint(both)


def test_fingerprint_stable_for_both():
    a = FaceKeepConfig()
    a.faithful.codec = "both"
    b = FaceKeepConfig()
    b.faithful.codec = "both"
    assert settings_fingerprint(a) == settings_fingerprint(b)


# --------------------------------------------------------------------------- #
# end-to-end through compress() with the real codecs
# --------------------------------------------------------------------------- #

_HAVE_BOTH = encoders.codec_available("avif") and encoders.codec_available("jxl")


@pytest.mark.skipif(not _HAVE_BOTH, reason="needs both avif and jxl plugins")
def test_compress_both_writes_decodable_file(plain_image, tmp_path):
    cfg = FaceKeepConfig()
    cfg.faithful.codec = "both"
    cfg.faithful.auto_tune = False  # keep it a single encode per codec, fast
    out = tmp_path / "out"
    res = faithful.compress(str(plain_image), str(out), cfg)

    # The chosen codec is one of the two concrete codecs, never "both".
    assert res.codec in ("avif", "jxl")
    # Output extension matches the chosen codec.
    assert res.output_path.suffix == encoders.CODEC_EXTENSION[res.codec]
    # And it decodes back at the right size.
    decoded = encoders.decode(res.output_path.read_bytes())
    loaded_shape = imageio.load(str(plain_image)).image.shape[:2]
    assert decoded.shape[:2] == loaded_shape


@pytest.mark.skipif(not _HAVE_BOTH, reason="needs both avif and jxl plugins")
def test_compress_both_is_no_larger_than_either_alone(plain_image, tmp_path):
    """'both' must produce output <= each single-codec output (it picks the min)."""
    sizes = {}
    for codec in ("avif", "jxl", "both"):
        cfg = FaceKeepConfig()
        cfg.faithful.codec = codec
        cfg.faithful.auto_tune = False
        cfg.faithful.skip_if_larger = False  # measure the real encode size
        res = faithful.compress(str(plain_image), str(tmp_path / codec), cfg)
        sizes[codec] = res.compressed_size
    assert sizes["both"] <= min(sizes["avif"], sizes["jxl"])


@pytest.mark.skipif(not _HAVE_BOTH, reason="needs both avif and jxl plugins")
def test_cli_codec_both(plain_image, tmp_path):
    out_dir = tmp_path / "outdir"
    out_dir.mkdir()  # an existing dir -> the writer appends <stem>.<ext> inside it
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["compress", str(plain_image), "-o", str(out_dir),
         "--codec", "both", "--no-auto-tune", "--no-progress"],
    )
    assert result.exit_code == 0, result.output
    produced = [p for p in out_dir.glob("*") if p.suffix in (".avif", ".jxl")]
    assert len(produced) == 1, list(out_dir.glob("*"))
    encoders.decode(produced[0].read_bytes())  # decodes without error

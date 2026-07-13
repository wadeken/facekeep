"""Residual layer — opt-in "middle mode" (ROADMAP Phase 8.5).

Aggressive mode's ceiling is information-theoretic: detail the downsample
discarded can only be *invented* back. The residual layer stores what was lost
— ``residual = original - bicubic(decoded background)`` — as one extra member
(``residual.jxl``, half-res, offset-encoded uint8), so restore can add it back
and the background becomes *real (lossy) data* instead of a hallucination. On
that path the AI upscale and GFPGAN are skipped (both exist to make
hallucination plausible; repainting real data would violate "never replace real
pixels with a hallucination"), while 8.2's grain still applies.

What these tests pin:

* the offset transform (``clip(r/2 + 128)`` <-> ``x*2 - 256``) round-trips up to
  its quantization step;
* the member lands as ``residual.jxl`` (or the warned ``residual.jpg`` fallback
  when the JXL plugin is unavailable), at ``residual_scale`` resolution, and the
  manifest bumps to 1.6.0 with the presence flag + knobs recorded;
* end-to-end restore WITH the residual is strictly closer to the original than
  without (SSIM/PSNR — for once the right metric: the background is real data
  again);
* the AI upsampler and GFPGAN are provably NOT called on the residual path
  (spies), while the normal path still uses them;
* ``preview()`` applies the residual too (it is stored *data*, not an
  enhancement — and the bench bicubic proxy is preview-based, so skipping it
  would hide the fidelity win from the numbers);
* grain (8.2) still runs on the residual path; restore stays deterministic;
* ``.fkeep`` round-trip + verify + backward compat (a no-residual file is
  unchanged; a v1.5.0 file still reads/verifies/restores);
* the three new config fields are validated, YAML round-trip, fingerprint-bust
  (aggressive only), and the CLI ``--residual`` flag wires through;
* dry-run size parity holds (the shared ``_write_archive`` rule).
"""

import json
import zipfile

import cv2
import numpy as np
import pytest
from click.testing import CliRunner

from facekeep import encoders, metrics
from facekeep.aggressive import restorer as restorer_mod
from facekeep.aggressive.compressor import CompressedPhoto, compress_photo
from facekeep.aggressive.format import (
    _fkeep_path,
    _offset_decode_residual,
    _offset_encode_residual,
    read_fkeep,
    read_fkeep_info,
    verify_fkeep,
    write_fkeep,
)
from facekeep.aggressive.restorer import Restorer, _apply_residual
from facekeep.cli import cli
from facekeep.config import FaceKeepConfig
from facekeep.exceptions import ConfigError
from facekeep.index import settings_fingerprint

requires_jxl = pytest.mark.skipif(
    not encoders.codec_available("jxl"), reason="JXL encoder not installed"
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _detailed_original(w: int = 960, h: int = 640) -> np.ndarray:
    """A detailed synthetic scene whose high frequencies a downsample destroys.

    Text-like strokes, a fine grid, and hard edges on a smooth gradient — the
    structured content aggressive mode honestly cannot reconstruct (the whole
    point of the residual). Deterministic, no per-pixel noise.
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.zeros((h, w, 3), np.uint8)
    img[..., 0] = (110 + 50 * np.sin(xx / 170)).clip(0, 255)
    img[..., 1] = (120 + 40 * np.cos(yy / 140)).clip(0, 255)
    img[..., 2] = (100 + 45 * np.sin((xx + yy) / 200)).clip(0, 255)
    # Fine 1px grid (regular structure: SR's worst case).
    img[::7, :] = (40, 40, 45)
    img[:, ::9] = (210, 205, 200)
    # Text-like strokes.
    for i, y in enumerate(range(40, h - 40, 90)):
        cv2.putText(img, "FACEKEEP 8.5 RESIDUAL", (20 + 5 * i, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (15, 15, 20), 2)
    return img


def _photo_with_residual(original: np.ndarray, *, residual: bool = True,
                         bg_scale: float = 0.25,
                         crops: bool = False, **agg_overrides) -> CompressedPhoto:
    """Build a CompressedPhoto around a controlled original, bypassing detection.

    ``background`` is the real INTER_AREA downsample of ``original`` (exactly
    what compress_photo produces), and ``original_image`` is attached iff
    ``residual`` — the same gating compress_photo applies. ``crops=True`` adds
    one noisy fake face crop so grain-estimation paths have real pixels.
    """
    h, w = original.shape[:2]
    cfg = FaceKeepConfig()
    cfg.aggressive.residual = residual
    for k, v in agg_overrides.items():
        setattr(cfg.aggressive, k, v)
    bw, bh = max(1, int(w * bg_scale)), max(1, int(h * bg_scale))
    background = cv2.resize(original, (bw, bh), interpolation=cv2.INTER_AREA)

    face_crops, face_masks, faces = [], [], []
    if crops:
        from facekeep.aggressive.blender import create_soft_mask
        from facekeep.detector import FaceRegion

        rng = np.random.default_rng(7)
        crop = original[40:168, 40:168].astype(np.float32)
        crop += rng.normal(0, 4.0, crop.shape).astype(np.float32)  # real grain
        face_crops = [np.clip(crop, 0, 255).astype(np.uint8)]
        face_masks = [create_soft_mask((128, 128), margin=16)]
        faces = [FaceRegion(id=0, bbox=(60, 60, 148, 148),
                            padded_bbox=(40, 40, 168, 168), confidence=0.9)]

    return CompressedPhoto(
        original_filename="p.jpg", original_width=w, original_height=h,
        original_size_bytes=999, original_hash="0" * 64, original_orientation=1,
        exif=None,
        background=background,
        face_crops=face_crops, face_masks=face_masks, faces=faces,
        thumbnail=np.full((128, 128, 3), 128, np.uint8),
        effective_bg_scale=bg_scale, config=cfg.aggressive,
        original_image=original if residual else None,
    )


def _write(photo, tmp_path, name):
    p = _fkeep_path(str(tmp_path / name))
    write_fkeep(photo, str(tmp_path / name))
    assert p.exists()
    return p


def _rewrite_members(fkeep, *, drop=(), replace=None, edit_manifest=None):
    """Rewrite a .fkeep in place: drop/replace members and/or edit the manifest."""
    with zipfile.ZipFile(fkeep) as zf:
        members = {n: zf.read(n) for n in zf.namelist() if n not in set(drop)}
    if replace:
        members.update(replace)
    if edit_manifest is not None:
        manifest = json.loads(members["manifest.json"])
        edit_manifest(manifest)
        members["manifest.json"] = json.dumps(manifest)
    with zipfile.ZipFile(fkeep, "w", zipfile.ZIP_DEFLATED) as zf:
        for n, b in members.items():
            zf.writestr(n, b)
    return str(fkeep)


def _residual_member(fkeep):
    with zipfile.ZipFile(fkeep) as zf:
        return next((n for n in zf.namelist() if n.startswith("residual.")), None)


# --------------------------------------------------------------------------- #
# the offset transform
# --------------------------------------------------------------------------- #

def test_offset_transform_roundtrips_within_quantization():
    """encode->decode reproduces the signed residual within the /2 step (<=1.0)."""
    rng = np.random.default_rng(0)
    residual = rng.uniform(-255, 255, (64, 64, 3)).astype(np.float32)
    out = _offset_decode_residual(_offset_encode_residual(residual))
    assert out.dtype == np.float32
    assert float(np.max(np.abs(out - residual))) <= 1.0 + 1e-5


def test_offset_encode_clips_to_uint8_range():
    """Out-of-range values (|r| > 256) clip instead of wrapping."""
    residual = np.array([[[-512.0, 0.0, 512.0]]], dtype=np.float32)
    enc = _offset_encode_residual(residual)
    assert enc.dtype == np.uint8
    assert enc.flatten().tolist() == [0, 128, 255]


# --------------------------------------------------------------------------- #
# compress side: member, manifest, scale, fallback, dry-run
# --------------------------------------------------------------------------- #

@requires_jxl
def test_residual_member_and_manifest(tmp_path):
    """residual=True stores residual.jxl; the manifest flags it (the residual
    keys landed at 1.6.0; the current writer schema is 1.11.0)."""
    photo = _photo_with_residual(_detailed_original())
    fkeep = _write(photo, tmp_path, "res")

    assert _residual_member(fkeep) == "residual.jxl"
    info = read_fkeep_info(str(fkeep))
    assert info["version"] == "1.11.0"
    assert info["settings"]["residual"] is True
    assert info["settings"]["residual_scale"] == 0.5
    assert info["settings"]["residual_quality"] == 60


def test_default_writes_no_residual_member(tmp_path):
    """The default (residual off) writes no member and flags False — but the
    schema version is still 1.11.0 (it describes the writer, like 1.5.0 did)."""
    photo = _photo_with_residual(_detailed_original(480, 320), residual=False)
    fkeep = _write(photo, tmp_path, "plain")

    assert _residual_member(fkeep) is None
    info = read_fkeep_info(str(fkeep))
    assert info["version"] == "1.11.0"
    assert info["settings"]["residual"] is False


@requires_jxl
def test_residual_member_is_at_residual_scale(tmp_path):
    """The stored residual decodes at residual_scale x the original dims."""
    original = _detailed_original(800, 600)
    photo = _photo_with_residual(original, residual_scale=0.5)
    fkeep = _write(photo, tmp_path, "scale")

    data = read_fkeep(str(fkeep))
    assert data["residual"] is not None
    assert data["residual"].shape[:2] == (300, 400)


def test_jxl_unavailable_falls_back_to_jpg(tmp_path, monkeypatch, caplog):
    """No JXL plugin -> warned residual.jpg fallback; the reader still finds it.

    format.py lazy-imports encoders inside the function, so patch the codec
    probe on the encoders module itself.
    """
    monkeypatch.setattr(encoders, "codec_available", lambda c: False)
    photo = _photo_with_residual(_detailed_original(480, 320))
    with caplog.at_level("WARNING", logger="facekeep.aggressive.format"):
        fkeep = _write(photo, tmp_path, "fallback")

    assert _residual_member(fkeep) == "residual.jpg"
    assert any("residual" in r.message.lower() for r in caplog.records)
    data = read_fkeep(str(fkeep))
    assert data["residual"] is not None


@requires_jxl
def test_dry_run_size_parity_with_residual(tmp_path, monkeypatch):
    """Dry-run size equals the real written size with a residual member.

    The packing guarantee is "deterministic within a second" (the manifest
    carries a seconds-precision created_at), so freeze the clock across the two
    packs — otherwise the pair can straddle a second boundary and the DEFLATEd
    manifest may differ by a byte, a rare flake unrelated to the residual.
    """
    import datetime as _dt

    import facekeep.aggressive.format as format_mod

    frozen = _dt.datetime(2026, 6, 11, 12, 0, 0, tzinfo=_dt.timezone.utc)

    class _FrozenDatetime:
        @staticmethod
        def now(tz=None):
            return frozen

    monkeypatch.setattr(format_mod, "datetime", _FrozenDatetime)

    photo = _photo_with_residual(_detailed_original(480, 320))
    est = write_fkeep(photo, str(tmp_path / "x"), dry_run=True)
    real = write_fkeep(photo, str(tmp_path / "x"))
    assert est == real


@requires_jxl
def test_compress_photo_attaches_original_only_when_enabled(face_image):
    """compress_photo gates original_image on cfg.residual (memory honesty)."""
    cfg = FaceKeepConfig()
    photo = compress_photo(str(face_image), cfg)
    assert photo.original_image is None

    cfg.aggressive.residual = True
    photo = compress_photo(str(face_image), cfg)
    assert photo.original_image is not None
    assert photo.original_image.dtype == np.uint8
    assert photo.original_image.shape[:2] == (
        photo.original_height, photo.original_width
    )


# --------------------------------------------------------------------------- #
# restore side: fidelity, skipped AI/GFPGAN, preview, grain, determinism
# --------------------------------------------------------------------------- #

@requires_jxl
def test_restore_with_residual_strictly_closer_to_original(tmp_path):
    """The residual path beats the plain bicubic restore on SSIM and PSNR.

    The background is real (lossy) data again, so fidelity metrics are — for
    once in aggressive mode — the right yardstick.
    """
    original = _detailed_original()
    with_res = _write(_photo_with_residual(original), tmp_path, "with")
    without = _write(_photo_with_residual(original, residual=False),
                     tmp_path, "without")

    r = Restorer()
    out_with = r.restore(str(with_res))
    out_without = r.restore(str(without))

    def mse(a, b):  # PSNR is monotone in MSE, so this is the PSNR assertion
        return float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))

    assert metrics.ssim(original, out_with) > metrics.ssim(original, out_without)
    assert mse(original, out_with) < mse(original, out_without)


@requires_jxl
def test_ai_and_gfpgan_not_called_on_residual_path(tmp_path, monkeypatch):
    """Spies prove the residual path never upscales with AI nor runs GFPGAN."""
    original = _detailed_original(480, 320)
    with_res = _write(_photo_with_residual(original), tmp_path, "spy")
    without = _write(_photo_with_residual(original, residual=False),
                     tmp_path, "spy_plain")

    calls = {"upscale": 0, "enhance": 0}
    real_upscale = Restorer._upscale_background

    def spy_upscale(self, *a, **k):
        calls["upscale"] += 1
        return real_upscale(self, *a, **k)

    def spy_enhance(self, bg):
        calls["enhance"] += 1
        return bg

    monkeypatch.setattr(Restorer, "_upscale_background", spy_upscale)
    monkeypatch.setattr(Restorer, "_enhance_background_faces", spy_enhance)

    Restorer().restore(str(with_res))
    assert calls == {"upscale": 0, "enhance": 0}

    # Anti-false-green: the normal path does go through both.
    Restorer().restore(str(without))
    assert calls["upscale"] == 1
    assert calls["enhance"] == 1


@requires_jxl
def test_preview_applies_residual(tmp_path):
    """preview() composites the residual too — it is stored *data*, and the
    bench bicubic proxy is preview-based, so skipping it would hide the win."""
    original = _detailed_original()
    with_res = _write(_photo_with_residual(original), tmp_path, "prev")
    without = _write(_photo_with_residual(original, residual=False),
                     tmp_path, "prev_plain")

    r = Restorer()
    assert (metrics.ssim(original, r.preview(str(with_res)))
            > metrics.ssim(original, r.preview(str(without))))


@requires_jxl
def test_cropless_preview_matches_restore(tmp_path):
    """With no crops (grain skipped) the residual restore and preview agree —
    they are the same bicubic + residual + (empty) composite pipeline."""
    fkeep = _write(_photo_with_residual(_detailed_original(480, 320)),
                   tmp_path, "agree")
    r = Restorer()
    assert np.array_equal(r.restore(str(fkeep)), r.preview(str(fkeep)))


@requires_jxl
def test_grain_still_applies_on_residual_path(tmp_path, monkeypatch):
    """8.2's grain runs on the residual path when the file has real crops."""
    fkeep = _write(_photo_with_residual(_detailed_original(), crops=True),
                   tmp_path, "grain")

    applied = {"n": 0}
    real_apply = restorer_mod._apply_grain

    def spy_apply(bg, sigma):
        applied["n"] += 1
        return real_apply(bg, sigma)

    monkeypatch.setattr(restorer_mod, "_apply_grain", spy_apply)
    Restorer().restore(str(fkeep))
    assert applied["n"] == 1


@requires_jxl
def test_residual_restore_deterministic(tmp_path):
    """Two restores of the same residual .fkeep are byte-identical."""
    fkeep = _write(_photo_with_residual(_detailed_original(), crops=True),
                   tmp_path, "det")
    r = Restorer()
    assert np.array_equal(r.restore(str(fkeep)), r.restore(str(fkeep)))


def test_apply_residual_helper_math():
    """The pure helper adds the decoded delta on top of the bicubic upscale."""
    bg = np.full((40, 60, 3), 100, np.uint8)
    # A flat +30 delta, stored at half resolution.
    residual = _offset_encode_residual(np.full((40, 60, 3), 30, np.float32))
    residual = cv2.resize(residual, (30, 20), interpolation=cv2.INTER_AREA)
    out = _apply_residual(bg, residual, 120, 80)
    assert out.shape == (80, 120, 3) and out.dtype == np.uint8
    assert abs(float(out.mean()) - 130.0) <= 1.5


# --------------------------------------------------------------------------- #
# verify + backward compat
# --------------------------------------------------------------------------- #

@requires_jxl
def test_verify_ok_with_residual(tmp_path):
    fkeep = _write(_photo_with_residual(_detailed_original(480, 320)),
                   tmp_path, "v_ok")
    rep = verify_fkeep(str(fkeep))
    assert rep.ok, rep.problems


@requires_jxl
def test_verify_catches_missing_residual_member(tmp_path):
    """A manifest that declares a residual whose member is gone is a problem."""
    fkeep = _write(_photo_with_residual(_detailed_original(480, 320)),
                   tmp_path, "v_miss")
    _rewrite_members(fkeep, drop=("residual.jxl",))

    rep = verify_fkeep(str(fkeep))
    assert rep.ok is False
    assert any("residual" in p for p in rep.problems)


@requires_jxl
def test_verify_reports_undecodable_residual(tmp_path):
    fkeep = _write(_photo_with_residual(_detailed_original(480, 320)),
                   tmp_path, "v_garb")
    _rewrite_members(fkeep, replace={"residual.jxl": b"not a jxl"})

    rep = verify_fkeep(str(fkeep))
    assert rep.ok is False
    assert any("residual" in p and "decode" in p for p in rep.problems)


def test_v150_file_reads_verifies_restores(tmp_path):
    """A downgraded v1.5.0 manifest (no residual keys) still works end-to-end."""
    original = _detailed_original(240, 160)
    fkeep = _write(_photo_with_residual(original, residual=False),
                   tmp_path, "old")

    def _downgrade(manifest):
        manifest["version"] = "1.5.0"
        for key in ("residual", "residual_scale", "residual_quality"):
            manifest["settings"].pop(key, None)

    _rewrite_members(fkeep, edit_manifest=_downgrade)

    data = read_fkeep(str(fkeep))
    assert data["residual"] is None
    rep = verify_fkeep(str(fkeep))
    assert rep.ok, rep.problems
    out = Restorer().restore(str(fkeep))
    assert out.shape[:2] == (160, 240)


# --------------------------------------------------------------------------- #
# config validation + YAML + fingerprint + CLI
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("field,value", [
    ("residual_scale", 0.0),
    ("residual_scale", 1.5),
    ("residual_quality", 0),
    ("residual_quality", 101),
])
def test_validate_rejects_out_of_range(field, value):
    cfg = FaceKeepConfig()
    setattr(cfg.aggressive, field, value)
    with pytest.raises(ConfigError, match=field):
        cfg.validate()


def test_validate_accepts_defaults_and_enabled():
    cfg = FaceKeepConfig()
    cfg.validate()
    cfg.aggressive.residual = True
    cfg.aggressive.residual_scale = 1.0
    cfg.aggressive.residual_quality = 100
    cfg.validate()  # must not raise


def test_yaml_roundtrip_residual(tmp_path):
    cfg = FaceKeepConfig()
    cfg.mode = "aggressive"
    cfg.aggressive.residual = True
    cfg.aggressive.residual_scale = 0.25
    cfg.aggressive.residual_quality = 80
    p = tmp_path / "c.yaml"
    cfg.save(p)
    loaded = FaceKeepConfig.load(p)
    assert loaded.aggressive.residual is True
    assert loaded.aggressive.residual_scale == 0.25
    assert loaded.aggressive.residual_quality == 80


@pytest.mark.parametrize("field,value", [
    ("residual", True),
    ("residual_scale", 0.25),
    ("residual_quality", 80),
])
def test_fingerprint_busts_on_each_residual_field(field, value):
    base = FaceKeepConfig(mode="aggressive")
    changed = FaceKeepConfig(mode="aggressive")
    setattr(changed.aggressive, field, value)
    assert settings_fingerprint(base) != settings_fingerprint(changed)


def test_faithful_fingerprint_ignores_residual():
    base = FaceKeepConfig()  # faithful
    changed = FaceKeepConfig()
    changed.aggressive.residual = True
    assert settings_fingerprint(base) == settings_fingerprint(changed)


@requires_jxl
def test_cli_residual_flag_wires_through(face_image, tmp_path):
    """compress --residual -m aggressive produces a .fkeep with the member."""
    target = tmp_path / "out" / "photo.fkeep"
    result = CliRunner().invoke(cli, [
        "compress", str(face_image), "-m", "aggressive", "--residual",
        "-o", str(target),
    ])
    assert result.exit_code == 0, result.output
    fkeep = target
    assert _residual_member(fkeep) == "residual.jxl"
    assert read_fkeep_info(str(fkeep))["settings"]["residual"] is True

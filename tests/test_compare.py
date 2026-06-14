"""`facekeep compare` — the before/after HTML comparison tool.

A read-only visualization command: it loads an original and a compressed artifact
(faithful image, an aggressive .fkeep restored on the fly, or an already-restored
image), and writes a single self-contained HTML report with a before/after
slider, a difference heatmap, and SSIM/PSNR. It changes no output pixels, so the
contract these tests pin is purely the report-building logic: the pure helpers
(diff map, data-URI embedding, the HTML string), the "after"-image dispatch, the
end-to-end render, and the CLI wiring.
"""

import base64

import cv2
import numpy as np
import pytest
from click.testing import CliRunner

from facekeep import compare as compare_mod
from facekeep.cli import cli
from facekeep.config import FaceKeepConfig
from facekeep.exceptions import FaceKeepError


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _img(h=120, w=160, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _write(path, img):
    assert cv2.imwrite(str(path), img)
    return path


def _decode_data_uri(uri):
    head, b64 = uri.split(",", 1)
    data = base64.b64decode(b64)
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return head, arr


# --------------------------------------------------------------------------- #
# diff_map
# --------------------------------------------------------------------------- #

def test_diff_map_identical_is_zero():
    x = _img()
    d = compare_mod.diff_map(x, x, colormap=False)
    assert d.shape == x.shape
    assert d.dtype == np.uint8
    assert not d.any()  # identical images -> all-zero difference


def test_diff_map_difference_is_nonzero():
    a, b = _img(seed=1), _img(seed=2)
    assert compare_mod.diff_map(a, b, colormap=False).any()


def test_diff_map_amplify_increases_magnitude():
    a = _img(seed=3)
    b = a.copy()
    b[:, :, :] = np.clip(b.astype(int) + 2, 0, 255).astype(np.uint8)  # tiny delta
    low = compare_mod.diff_map(a, b, amplify=1.0, colormap=False).mean()
    high = compare_mod.diff_map(a, b, amplify=20.0, colormap=False).mean()
    assert high > low


def test_diff_map_colormap_is_bgr_3channel():
    a, b = _img(seed=4), _img(seed=5)
    d = compare_mod.diff_map(a, b, colormap=True)
    assert d.shape == a.shape and d.dtype == np.uint8


def test_diff_map_dtype_safe_uint16():
    a = (_img(seed=6).astype(np.uint16) * 257)  # 16-bit view of the same content
    b = (_img(seed=7).astype(np.uint16) * 257)
    d = compare_mod.diff_map(a, b, colormap=False)
    assert d.dtype == np.uint8 and d.any()


# --------------------------------------------------------------------------- #
# _embed (data URIs)
# --------------------------------------------------------------------------- #

def test_embed_jpeg_data_uri():
    uri = compare_mod._embed(_img(), fmt="jpeg")
    assert uri.startswith("data:image/jpeg;base64,")
    head, arr = _decode_data_uri(uri)
    assert arr is not None and arr.shape[2] == 3


def test_embed_png_data_uri():
    uri = compare_mod._embed(_img(), fmt="png")
    assert uri.startswith("data:image/png;base64,")


def test_embed_max_side_downscales_preview():
    big = _img(h=400, w=800)
    uri = compare_mod._embed(big, fmt="jpeg", max_side=100)
    _head, arr = _decode_data_uri(uri)
    assert max(arr.shape[:2]) <= 101  # longest side capped (allow rounding)


def test_embed_full_res_keeps_size():
    img = _img(h=300, w=200)
    _head, arr = _decode_data_uri(compare_mod._embed(img, fmt="png", max_side=None))
    assert arr.shape[:2] == (300, 200)


# --------------------------------------------------------------------------- #
# align
# --------------------------------------------------------------------------- #

def test_align_noop_when_same_shape():
    x = _img()
    out = compare_mod.align(x, x)
    assert out is x  # returned unchanged (no copy/resize)


def test_align_resizes_to_reference():
    after = _img(h=50, w=70, seed=8)
    like = _img(h=120, w=160, seed=9)
    out = compare_mod.align(after, like)
    assert out.shape[:2] == like.shape[:2]


# --------------------------------------------------------------------------- #
# build_html (pure)
# --------------------------------------------------------------------------- #

def test_build_html_is_self_contained_and_has_markers():
    doc = compare_mod.build_html(
        before_uri="data:image/jpeg;base64,AAAA",
        after_uri="data:image/jpeg;base64,BBBB",
        diff_uri="data:image/png;base64,CCCC",
        metric_rows=[("SSIM", "0.9876", "higher = better")],
        meta_rows=[("Original", "photo.jpg", "160x120, 1.0 KB")],
        title="FaceKeep compare — photo.avif",
        before_label="Original", after_label="Compressed (decoded .avif)",
        diff_caption="Mean absolute difference, amplified 8x.",
    )
    # Self-contained: the images are inlined data URIs, no external asset refs.
    assert "data:image/jpeg;base64,AAAA" in doc
    assert "data:image/png;base64,CCCC" in doc
    assert 'src="http' not in doc and "<link" not in doc
    # Interactive slider hooks + both image layers are present.
    assert 'id="cmp-range"' in doc and 'id="cmp-after"' in doc
    assert doc.count("ba-img") >= 2
    # The numbers and labels made it in.
    assert "0.9876" in doc and "SSIM" in doc
    assert "photo.jpg" in doc and "photo.avif" in doc


def test_build_html_escapes_text():
    doc = compare_mod.build_html(
        before_uri="data:,", after_uri="data:,", diff_uri="data:,",
        metric_rows=[("SSIM", "1.0", "x")],
        meta_rows=[("Original", "a<b>&.jpg", "1x1")],
        title="t", before_label="<L>", after_label="<R>", diff_caption="c",
    )
    assert "a<b>&.jpg" not in doc  # raw angle brackets escaped
    assert "a&lt;b&gt;&amp;.jpg" in doc


# --------------------------------------------------------------------------- #
# _default_output (dotted-filename safety)
# --------------------------------------------------------------------------- #

def test_default_output_simple():
    out = compare_mod._default_output("/x/y/out.avif")
    assert out.name == "out_compare.html"


def test_default_output_dotted_name_preserved():
    out = compare_mod._default_output("/x/2024.05.20_trip.avif")
    assert out.name == "2024.05.20_trip_compare.html"


# --------------------------------------------------------------------------- #
# load_after (dispatch)
# --------------------------------------------------------------------------- #

def test_load_after_decodes_standard_image(tmp_path):
    src = _write(tmp_path / "c.png", _img(seed=10))
    arr, kind = compare_mod.load_after(str(src), FaceKeepConfig().aggressive)
    assert arr.shape[2] == 3 and kind == "decoded .png"


def test_load_after_fkeep_uses_restore(tmp_path, monkeypatch):
    known = _img(seed=11)

    class _FakeRestorer:
        def __init__(self, cfg):
            self.cfg = cfg

        def restore(self, path, out, *, quality=70):
            return known

        def preview(self, path, out, *, quality=70):
            raise AssertionError("preview must not be called without --preview")

    monkeypatch.setattr("facekeep.aggressive.restorer.Restorer", _FakeRestorer)
    arr, kind = compare_mod.load_after(str(tmp_path / "x.fkeep"), None)
    assert kind == "restore" and np.array_equal(arr, known)


def test_load_after_fkeep_preview(tmp_path, monkeypatch):
    known = _img(seed=12)

    class _FakeRestorer:
        def __init__(self, cfg):
            pass

        def preview(self, path, out, *, quality=70):
            return known

        def restore(self, path, out, *, quality=70):
            raise AssertionError("restore must not be called with --preview")

    monkeypatch.setattr("facekeep.aggressive.restorer.Restorer", _FakeRestorer)
    arr, kind = compare_mod.load_after(str(tmp_path / "x.fkeep"), None, preview=True)
    assert kind == "bicubic preview" and np.array_equal(arr, known)


# --------------------------------------------------------------------------- #
# load_after — per-file preset auto-apply (the "after" matches `facekeep restore`)
# --------------------------------------------------------------------------- #

def _spy_restorer(monkeypatch):
    """Patch Restorer to capture the agg config it is built with (no real restore)."""
    captured = {}

    class _Spy:
        def __init__(self, cfg):
            captured["cfg"] = cfg

        def restore(self, path, out, *, quality=70):
            return _img(seed=7)

        def preview(self, path, out, *, quality=70):
            return _img(seed=7)

    monkeypatch.setattr("facekeep.aggressive.restorer.Restorer", _Spy)
    return captured


def _fake_manifest(monkeypatch, manifest):
    monkeypatch.setattr("facekeep.aggressive.format.read_fkeep_info",
                        lambda path: manifest)


def test_load_after_applies_fkeep_manifest_preset(tmp_path, monkeypatch):
    # A .fkeep recorded with preset "pretty" drives its restore-side knobs
    # (face_enhance_fidelity 0.5, not the 0.7 default), so the compare "after"
    # matches what `facekeep restore` actually produces — the consistency fix.
    captured = _spy_restorer(monkeypatch)
    _fake_manifest(monkeypatch, {"settings": {"preset": "pretty"}})
    base = FaceKeepConfig().aggressive
    assert base.face_enhance_fidelity == 0.7  # guard the premise
    compare_mod.load_after(str(tmp_path / "x.fkeep"), base)
    assert captured["cfg"].face_enhance_fidelity == 0.5  # pretty's value applied
    assert captured["cfg"] is not base  # a copy, the passed config is not mutated


def test_load_after_preview_ignores_preset(tmp_path, monkeypatch):
    # preview never enhances faces, so the preset must NOT be applied — the base
    # config is used as-is (mirrors cli.restore keeping preview on base_restorer).
    captured = _spy_restorer(monkeypatch)
    _fake_manifest(monkeypatch, {"settings": {"preset": "pretty"}})
    base = FaceKeepConfig().aggressive
    compare_mod.load_after(str(tmp_path / "x.fkeep"), base, preview=True)
    assert captured["cfg"] is base


def test_load_after_explicit_key_beats_manifest_preset(tmp_path, monkeypatch):
    # An explicitly-set restore knob wins over the manifest preset (the same
    # precedence as compress and `facekeep restore`).
    captured = _spy_restorer(monkeypatch)
    _fake_manifest(monkeypatch, {"settings": {"preset": "pretty"}})
    base = FaceKeepConfig().aggressive
    compare_mod.load_after(
        str(tmp_path / "x.fkeep"), base,
        explicit_keys=frozenset({"aggressive.face_enhance_fidelity"}))
    assert captured["cfg"].face_enhance_fidelity == 0.7  # explicit default kept


def test_load_after_no_preset_uses_base_config(tmp_path, monkeypatch):
    # A presetless .fkeep (no settings.preset) -> no overrides, base config as-is.
    captured = _spy_restorer(monkeypatch)
    _fake_manifest(monkeypatch, {"settings": {}})
    base = FaceKeepConfig().aggressive
    compare_mod.load_after(str(tmp_path / "x.fkeep"), base)
    assert captured["cfg"] is base


def test_load_after_unreadable_manifest_falls_back(tmp_path, monkeypatch):
    # Manifest unreadable (here a missing file -> OSError inside the helper) ->
    # fall back to the base config and let restore() report the real problem.
    captured = _spy_restorer(monkeypatch)
    base = FaceKeepConfig().aggressive
    compare_mod.load_after(str(tmp_path / "missing.fkeep"), base)
    assert captured["cfg"] is base


# --------------------------------------------------------------------------- #
# render_comparison (end-to-end, image branch — no codec/AI needed)
# --------------------------------------------------------------------------- #

def test_render_comparison_writes_html_and_summary(tmp_path):
    orig = _write(tmp_path / "orig.png", _img(seed=20))
    comp = _write(tmp_path / "comp.png", _img(seed=21))
    out = tmp_path / "report.html"
    summary = compare_mod.render_comparison(
        str(orig), str(comp), str(out), agg_config=FaceKeepConfig().aggressive,
    )
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "data:image/jpeg;base64," in text and "data:image/png;base64," in text
    assert "SSIM" in text
    # Summary carries real numbers for the CLI to print.
    assert summary["output_path"] == str(out)
    assert 0.0 <= summary["ssim"] <= 1.0
    assert summary["original_bytes"] > 0 and summary["compressed_bytes"] > 0
    assert summary["after_kind"] == "decoded .png"
    assert summary["lpips"] is None and summary["ssimulacra2"] is None


def test_render_comparison_identical_is_high_ssim(tmp_path):
    img = _img(seed=22)
    orig = _write(tmp_path / "a.png", img)
    comp = _write(tmp_path / "b.png", img)
    summary = compare_mod.render_comparison(
        str(orig), str(comp), str(tmp_path / "r.html"),
        agg_config=FaceKeepConfig().aggressive,
    )
    assert summary["ssim"] > 0.999  # same pixels -> ~1.0


def test_render_comparison_default_output_path(tmp_path):
    orig = _write(tmp_path / "orig.png", _img(seed=23))
    comp = _write(tmp_path / "shot.png", _img(seed=24))
    summary = compare_mod.render_comparison(
        str(orig), str(comp), None, agg_config=FaceKeepConfig().aggressive,
    )
    assert summary["output_path"].endswith("shot_compare.html")
    assert (tmp_path / "shot_compare.html").exists()


def test_render_comparison_aligns_mismatched_sizes(tmp_path):
    orig = _write(tmp_path / "o.png", _img(h=120, w=160, seed=25))
    comp = _write(tmp_path / "c.png", _img(h=60, w=80, seed=26))
    # Must not raise despite differing dimensions (align resizes the "after").
    summary = compare_mod.render_comparison(
        str(orig), str(comp), str(tmp_path / "r.html"),
        agg_config=FaceKeepConfig().aggressive,
    )
    assert "ssim" in summary


def test_render_comparison_bad_input_raises_facekeep_error(tmp_path):
    orig = _write(tmp_path / "o.png", _img(seed=27))
    bad = tmp_path / "broken.png"
    bad.write_bytes(b"not an image")
    with pytest.raises(FaceKeepError):
        compare_mod.render_comparison(
            str(orig), str(bad), str(tmp_path / "r.html"),
            agg_config=FaceKeepConfig().aggressive,
        )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def test_cli_compare_writes_report(tmp_path):
    orig = _write(tmp_path / "orig.png", _img(seed=30))
    comp = _write(tmp_path / "comp.png", _img(seed=31))
    out = tmp_path / "cmp.html"
    res = CliRunner().invoke(
        cli, ["compare", str(orig), str(comp), "-o", str(out)]
    )
    assert res.exit_code == 0, res.output
    assert out.exists()
    assert "SSIM" in res.output and "Wrote" in res.output


def test_cli_compare_default_output(tmp_path):
    orig = _write(tmp_path / "orig.png", _img(seed=32))
    comp = _write(tmp_path / "comp.png", _img(seed=33))
    res = CliRunner().invoke(cli, ["compare", str(orig), str(comp)])
    assert res.exit_code == 0, res.output
    assert (tmp_path / "comp_compare.html").exists()


def test_cli_compare_undecodable_compressed_fails_cleanly(tmp_path):
    orig = _write(tmp_path / "orig.png", _img(seed=34))
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"junk")
    res = CliRunner().invoke(cli, ["compare", str(orig), str(bad)])
    assert res.exit_code == 1
    assert "Compare failed" in res.output


def test_cli_compare_lpips_hint_when_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr("facekeep.metrics.lpips_available", lambda: False)
    orig = _write(tmp_path / "orig.png", _img(seed=35))
    comp = _write(tmp_path / "comp.png", _img(seed=36))
    res = CliRunner().invoke(
        cli, ["compare", str(orig), str(comp), "-o", str(tmp_path / "r.html"),
              "--lpips"]
    )
    assert res.exit_code == 0, res.output
    assert "LPIPS unavailable" in res.output  # hint, not a silent blank

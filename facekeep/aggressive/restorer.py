"""Aggressive-mode restoration: AI super-resolution + face compositing."""

import logging
import struct
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..config import AggressiveConfig
from ..exceptions import ModelDownloadError, RestoreError
from ..models import ensure_weights
from .blender import blend_face_onto_background
from .format import _offset_decode_residual, read_fkeep

logger = logging.getLogger("facekeep.aggressive.restorer")

# Real-ESRGAN weights by model name: (url, filename, sha256). ``aggressive.model``
# selects which. Rather than hand RealESRGANer the ``https://`` URL (which would
# download into the realesrgan package's own ``weights/`` dir, unverified), we
# fetch + checksum-verify the file via ``models.ensure_weights`` into the shared
# FaceKeep cache and pass RealESRGANer the resulting *local path* — it then loads
# our verified file instead of downloading. SHA-256 computed from the official
# release weights.
_REALESRGAN_WEIGHTS = {
    "realesrgan-x4plus": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/"
        "v0.1.0/RealESRGAN_x4plus.pth",
        "RealESRGAN_x4plus.pth",
        "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1",
    ),
}

# GFPGAN face-restoration weights: (url, filename, sha256). Same handling as
# above — fetched + verified through ``ensure_weights`` and passed to GFPGANer as
# a local path. v1.4 is the current "clean" arch release.
_GFPGAN_WEIGHTS = (
    "https://github.com/TencentARC/GFPGAN/releases/download/"
    "v1.3.4/GFPGANv1.4.pth",
    "GFPGANv1.4.pth",
    "e2cd4703ab14f4d01fd1383a8a8b266f9a5833dacee8e6a79d3bf21a1b6be5ad",
)

# CodeFormer face-restoration weights (the opt-in `face_enhance_backend:
# codeformer`): (url, filename, sha256). Same handling as above — fetched +
# verified through ``ensure_weights``, loaded from the local path. NOTE:
# CodeFormer's code and weights are S-Lab License 1.0 (non-commercial); the
# arch comes from the `codeformer-pip` package ([codeformer] extra) and is
# never vendored into this repo — installing the extra is the user accepting
# that license. SHA-256 computed from a real download of the official release
# asset (2026-06-11).
_CODEFORMER_WEIGHTS = (
    "https://github.com/sczhou/CodeFormer/releases/download/"
    "v0.1.0/codeformer.pth",
    "codeformer.pth",
    "1009e537e0c2a07d4cabce6355f53cb66767cd4b4297ec7a4a64ca4b8a5684b7",
)


def _ensure_torchvision_compat() -> None:
    """Shim the module BasicSR imports but modern torchvision no longer ships.

    Real-ESRGAN and GFPGAN both import ``basicsr``, which at import time does
    ``from torchvision.transforms.functional_tensor import rgb_to_grayscale``.
    ``torchvision.transforms.functional_tensor`` was **removed in torchvision
    0.17**; ``rgb_to_grayscale`` now lives in ``torchvision.transforms.functional``.
    Without this shim, installing the ``[ai]`` extra against a current torchvision
    still raises ``ModuleNotFoundError`` deep inside basicsr — which our
    ``except ImportError`` would silently swallow, degrading to bicubic so the user
    never actually gets AI restore despite having installed it.

    We register an alias module pointing at the new location *before* basicsr is
    imported. It only runs on the AI restore path (never on the default offline
    faithful path), is a no-op if ``functional_tensor`` already exists or if
    torchvision isn't installed, and only adds a missing legacy name — it never
    overrides an existing module.
    """
    if "torchvision.transforms.functional_tensor" in sys.modules:
        return
    try:
        import importlib

        importlib.import_module("torchvision.transforms.functional_tensor")
        return  # genuinely present (older torchvision) — nothing to shim
    except ImportError:
        pass
    try:
        functional = importlib.import_module("torchvision.transforms.functional")
    except ImportError:
        return  # torchvision absent entirely; let the real import fail/skip
    sys.modules["torchvision.transforms.functional_tensor"] = functional


def realesrgan_available() -> bool:
    """True iff the Real-ESRGAN stack (the ``[ai]`` extra) can be imported.

    A cheap **import-level** probe (no model construction, no weight download),
    mirroring ``metrics.lpips_available`` / ``ssimulacra2_available``: it lets a
    caller (the GUI's opt-in "real AI restore" toggle) decide up front whether a
    *genuine* AI restore is possible, so it can warn honestly instead of silently
    running the bicubic fallback and mislabeling it. It does **not** guarantee the
    weights will download — an offline machine still degrades to bicubic at
    restore time (``_init_upsampler``'s ``ModelDownloadError`` branch) — it only
    reports that the package is installed. Returns ``False`` (never raises) when
    the extra is absent.
    """
    try:
        _ensure_torchvision_compat()
        from basicsr.archs.rrdbnet_arch import RRDBNet  # noqa: F401
        from realesrgan import RealESRGANer  # noqa: F401

        return True
    except ImportError:
        return False


# --- Low-frequency anchoring (ROADMAP 8.1) ---------------------------------- #
# Real-ESRGAN is trained for perceptual realism and drifts in color/brightness/
# low-frequency structure vs the real photo. But the stored background.jpg *is* a
# real measurement: every spatial frequency below its Nyquist is data, not
# guesswork. So after the AI upscale we swap the result's low band for the
# reference's. The Gaussian's sigma is derived from the upscale factor
# (~1/bg_scale): with sigma = factor/bg_scale the blur's pass-band sits safely
# *inside* the band the stored background genuinely measured, so only
# certainly-real frequencies are transplanted (erring toward "swap less" — the
# safe direction; a wider band would start pulling the stored JPEG's q85
# artifacts back in). Module constants, not config (ROADMAP 8.1).
_ANCHOR_SIGMA_FACTOR = 1.0

# Back-projection step size for the optional mid-band consistency iterations
# (aggressive.restore_backproject_iters, default 0 = off). 0.5 is the gentle
# half-step the ROADMAP item suggests experimenting with.
_BACKPROJECT_LAMBDA = 0.5


def _anchor_sigma(bg_scale: float) -> float:
    """Gaussian sigma for low-frequency anchoring, from the upscale factor.

    Larger upscales (smaller ``bg_scale``) mean the stored background measured a
    narrower band of real frequencies, so the anchor blur must be wider (bigger
    sigma) to stay inside it.
    """
    return _ANCHOR_SIGMA_FACTOR / float(bg_scale)


def _anchor_low_frequencies(upscaled: np.ndarray, bg: np.ndarray,
                            bg_scale: float) -> np.ndarray:
    """Replace ``upscaled``'s low-frequency band with the real background's.

    ``ref = bicubic(bg)`` carries the photo's true low-frequency signal (color,
    brightness, large structure); the AI output keeps only its high-frequency
    detail on top of it: ``out = sr - blur(sr) + blur(ref)``. Pure NumPy/OpenCV,
    float32 math, clipped back to uint8 — shape and dtype preserved.
    """
    h, w = upscaled.shape[:2]
    sigma = _anchor_sigma(bg_scale)
    ref = cv2.resize(bg, (w, h), interpolation=cv2.INTER_CUBIC)
    sr = upscaled.astype(np.float32)
    # (0, 0) kernel: OpenCV derives the kernel size from sigma.
    out = (
        sr
        - cv2.GaussianBlur(sr, (0, 0), sigma)
        + cv2.GaussianBlur(ref.astype(np.float32), (0, 0), sigma)
    )
    return np.clip(out, 0, 255).astype(np.uint8)


# --- Grain matching (ROADMAP 8.2) ------------------------------------------- #
# The composite mixes *real* pixels (face crops / region patches, carrying
# natural sensor noise and JPEG texture) with a GAN/bicubic background that is
# too smooth ("plastic") — so even a perfectly feathered paste is findable by
# texture discontinuity. Standard SR-pipeline fix: estimate the grain level
# from the real crops and add matched grain to the reconstructed background
# before compositing. Module constants, not config (the 8.1 precedent).
_GRAIN_RESIDUAL_SIGMA = 1.5  # blur whose residual isolates the grain band
_GRAIN_SOFTEN_SIGMA = 0.5  # light blur so noise reads as grain, not salt-and-pepper
_GRAIN_SEED = 0  # fixed seed -> the same .fkeep restores to the same bytes
_MAD_TO_SIGMA = 1.4826  # MAD -> Gaussian sigma (the standard consistency factor)


def _estimate_grain_sigma(crops) -> Optional[float]:
    """Estimate the grain level (Gaussian sigma) carried by the real crops.

    Per crop: take the luma's high-frequency residual ``luma - blur(luma)`` and
    measure its spread with the robust **median absolute deviation** scaled to a
    Gaussian sigma — NOT a std, because real edges (eyes, hairlines) are sparse
    outliers that would inflate a std far above the true noise floor. The
    per-crop estimates are then medianed. Returns ``None`` for an empty list
    (no real-pixel patches -> no texture mismatch to hide).
    """
    sigmas = []
    for crop in crops:
        luma = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        residual = luma - cv2.GaussianBlur(luma, (0, 0), _GRAIN_RESIDUAL_SIGMA)
        mad = float(np.median(np.abs(residual - np.median(residual))))
        sigmas.append(mad * _MAD_TO_SIGMA)
    if not sigmas:
        return None
    return float(np.median(sigmas))


def _apply_grain(bg: np.ndarray, sigma: float) -> np.ndarray:
    """Add seeded, deterministic mono grain of strength ``sigma`` to ``bg``.

    One Gaussian noise field, lightly blurred so it reads as photographic grain
    rather than salt-and-pepper, added identically to all three BGR channels —
    luma-only grain; chroma noise looks wrong. Seeded RNG keeps restore
    deterministic: the same ``.fkeep`` restores to the same bytes every run.
    """
    if sigma <= 0:
        return bg
    h, w = bg.shape[:2]
    rng = np.random.default_rng(_GRAIN_SEED)
    noise = rng.standard_normal((h, w), dtype=np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), _GRAIN_SOFTEN_SIGMA)
    # The soften blur attenuates the field's energy (~0.6x std); renormalize so
    # the applied grain strength actually equals the estimated sigma.
    std = float(noise.std())
    if std <= 0:
        return bg
    noise *= sigma / std
    out = bg.astype(np.float32) + noise[:, :, None]
    return np.clip(out, 0, 255).astype(np.uint8)


# --- Residual layer (ROADMAP 8.5) ------------------------------------------- #
# When a .fkeep carries a residual member (aggressive.residual at compress
# time), the background is reconstructed from *real data*: a plain bicubic
# upscale plus the stored high-frequency delta. On that path the AI upscale and
# GFPGAN are deliberately skipped — both exist to make hallucination plausible,
# and repainting real data would violate "never replace real pixels with a
# hallucination" (soft-but-real beats fake-but-sharp). 8.1's anchor is moot
# (the low band already IS the stored background's); 8.2's grain still applies.


def _apply_residual(bg: np.ndarray, residual: np.ndarray,
                    width: int, height: int) -> np.ndarray:
    """Reconstruct the background from real data: bicubic upscale + residual.

    ``residual`` is the offset-encoded uint8 member as stored (see
    ``format._offset_encode_residual``); its signed delta is resized to full
    resolution with INTER_CUBIC — the same interpolation the compress side used
    to compute it (a pinned contract, see ``format._encode_residual``) — and
    added on top of the bicubic upscale, clipped back to uint8.
    """
    up = cv2.resize(bg, (width, height), interpolation=cv2.INTER_CUBIC)
    delta = _offset_decode_residual(residual)
    if delta.shape[:2] != (height, width):
        delta = cv2.resize(delta, (width, height), interpolation=cv2.INTER_CUBIC)
    return np.clip(up.astype(np.float32) + delta, 0, 255).astype(np.uint8)


def _back_project(upscaled: np.ndarray, bg: np.ndarray, iters: int) -> np.ndarray:
    """Gentle iterative back-projection pinning the mid band to the stored bg.

    Each step nudges the restore toward downsample-consistency with the real
    stored background: ``x <- x + lambda * up(bg - down(x))``, where ``down`` is
    INTER_AREA to the stored background's *exact* size (matching how the
    background was produced at compress time) and ``up`` is bicubic back. Off by
    default (``aggressive.restore_backproject_iters = 0``): the stored bg carries
    JPEG q85 artifacts that strict consistency would pull back in.
    """
    if iters <= 0:
        return upscaled
    h, w = upscaled.shape[:2]
    bh, bw = bg.shape[:2]
    bg_f = bg.astype(np.float32)
    x = upscaled.astype(np.float32)
    for _ in range(iters):
        down = cv2.resize(x, (bw, bh), interpolation=cv2.INTER_AREA)
        residual = cv2.resize(bg_f - down, (w, h), interpolation=cv2.INTER_CUBIC)
        x += _BACKPROJECT_LAMBDA * residual
    return np.clip(x, 0, 255).astype(np.uint8)


class _CodeFormerEnhancer:
    """CodeFormer face restorer with a GFPGANer-shaped surface.

    Duck-types the three things ``_enhance_background_faces`` reads off
    ``GFPGANer`` — ``enhance(...) -> (cropped, restored, None)``, ``face_helper``
    (for the inverse affines), and ``upscale`` — so the enhancement + bounded
    self-paste path stays a *single* code path for both backends.

    Detection/alignment uses facexlib's ``FaceRestoreHelper`` (the exact helper
    GFPGANer wraps, installed by the [ai] extra), NOT the helper vendored inside
    codeformer-pip — the vendored one downloads its detection weights into a
    CWD-relative ``weights/`` directory, which a library must never do. From
    codeformer-pip we import only the network arch. ``use_parse=False`` because
    our paste (``Restorer._paste_restored_face``) mirrors GFPGAN's *non-parse*
    feathered-square blend, so the parsing model would be a dead ~85 MB download.

    ``net`` / ``face_helper`` are injectable for offline tests; the default
    construction imports codeformer-pip and loads the checksum-verified weights
    from ``model_path``.
    """

    upscale = 1  # we enhance the already-upscaled background in place

    def __init__(self, model_path, fidelity, net=None, face_helper=None):
        import torch

        self._torch = torch
        self.fidelity = float(fidelity)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if net is None:
            # Import only the arch module — codeformer-pip's `app` module
            # downloads ~700 MB of models into the CWD at import time and must
            # never be touched. The arch import is side-effect-free (verified).
            from codeformer.basicsr.archs.codeformer_arch import CodeFormer

            net = CodeFormer(
                dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
                connect_list=["32", "64", "128", "256"],
            ).to(self.device)
            ckpt = torch.load(model_path, map_location=self.device)["params_ema"]
            net.load_state_dict(ckpt)
            net.eval()
        self.net = net
        if face_helper is None:
            from facexlib.utils.face_restoration_helper import FaceRestoreHelper

            face_helper = FaceRestoreHelper(
                upscale_factor=1, face_size=512, crop_ratio=(1, 1),
                det_model="retinaface_resnet50", save_ext="png",
                use_parse=False, device=self.device,
            )
        self.face_helper = face_helper

    def enhance(self, img, has_aligned=False, only_center_face=False,
                paste_back=False):
        """Align + restore each face to a 512x512 crop; never paste.

        Mirrors ``GFPGANer.enhance(paste_back=False)``: returns
        ``(cropped_faces, restored_faces, None)`` and leaves the helper's
        ``affine_matrices`` populated so the caller can compute the inverse
        affines for its own bounded paste. No full-frame buffer is built here.
        BGR in / BGR out (the repo convention); the BGR<->RGB flip happens only
        at the model boundary below, like the MediaPipe detect path.
        """
        torch = self._torch
        helper = self.face_helper
        helper.clean_all()
        helper.read_image(img)
        helper.get_face_landmarks_5(
            only_center_face=only_center_face, eye_dist_threshold=5
        )
        helper.align_warp_face()
        for cropped in helper.cropped_faces:
            # BGR uint8 -> RGB CHW float in [-1, 1] (CodeFormer's input space).
            t = torch.from_numpy(
                cropped[:, :, ::-1].transpose(2, 0, 1).copy()
            ).float() / 255.0
            t = ((t - 0.5) / 0.5).unsqueeze(0).to(self.device)
            try:
                with torch.no_grad():
                    out = self.net(t, w=self.fidelity, adain=True)[0]
                out = out.squeeze(0).float().clamp_(-1, 1).cpu().numpy()
                restored = (out.transpose(1, 2, 0)[:, :, ::-1] + 1.0) * 0.5 * 255.0
                restored = restored.round().astype(np.uint8)
            except RuntimeError as e:
                # GFPGAN parity: one failed face (OOM, torch error) keeps its
                # un-restored crop instead of failing the whole pass.
                logger.warning(
                    "CodeFormer inference failed for a face (%s); keeping the "
                    "un-restored crop.", e,
                )
                restored = cropped
            helper.add_restored_face(restored)
        return helper.cropped_faces, helper.restored_faces, None


class Restorer:
    """Restore aggressive-mode .fkeep photos using AI super-resolution.

    Tries Real-ESRGAN if installed; otherwise falls back to bicubic upscaling
    (much lower quality, but keeps the tool usable without the AI extras).
    """

    def __init__(self, config: Optional[AggressiveConfig] = None):
        self.config = config or AggressiveConfig()
        self._upsampler = None
        self._tried_init = False
        self._face_enhancer = None
        self._tried_face_init = False

    def _init_upsampler(self):
        self._tried_init = True
        model_name = self.config.model
        weights = _REALESRGAN_WEIGHTS.get(model_name)
        if weights is None:
            logger.warning(
                "Unknown Real-ESRGAN model %r; known: %s. Using bicubic upscaling.",
                model_name, ", ".join(sorted(_REALESRGAN_WEIGHTS)),
            )
            self._upsampler = None
            return
        url, filename, sha256 = weights
        try:
            _ensure_torchvision_compat()
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer

            # Fetch + checksum-verify into the shared FaceKeep model cache, then
            # hand RealESRGANer the *local path* (not the URL) so it loads our
            # verified file instead of downloading unverified into site-packages.
            model_path = str(ensure_weights(url, filename, sha256=sha256))
            model = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=23, num_grow_ch=32, scale=4,
            )
            # Tile the upscale so a large background never builds a full-resolution
            # intermediate at once (memory bounded on 24MP+). Sizes are config knobs
            # (aggressive.tile / tile_pad), defaulting to 512 / 10 as before.
            self._upsampler = RealESRGANer(
                scale=4, model_path=model_path, model=model,
                tile=self.config.tile, tile_pad=self.config.tile_pad,
                pre_pad=0, half=False, gpu_id=None,
            )
        except ImportError:
            logger.warning(
                "Real-ESRGAN not installed; restore will use bicubic upscaling. "
                "Install with: pip install facekeep[ai]"
            )
            self._upsampler = None
        except (ModelDownloadError, RuntimeError, OSError, ValueError) as e:
            # Weights download / checksum / load failure (offline, corrupt cache,
            # bad URL): degrade to bicubic instead of failing the whole restore.
            logger.warning(
                "Could not load Real-ESRGAN weights (%s); using bicubic upscaling.", e
            )
            self._upsampler = None

    def _upscale_background(self, bg, target_w, target_h, bg_scale):
        """Upscale ``bg`` to the original size; returns ``(out, used_ai)``.

        ``used_ai`` reports whether the Real-ESRGAN path actually produced the
        output — low-frequency anchoring (ROADMAP 8.1) keys off it, because the
        bicubic fallback is consistent with ``bg`` by construction (anchoring it
        would be an identity up to float rounding) and skipping keeps the bicubic
        path byte-identical.
        """
        if not self._tried_init:
            self._init_upsampler()

        if self._upsampler is not None:
            try:
                out, _ = self._upsampler.enhance(bg, outscale=1.0 / bg_scale)
                if out.shape[:2] != (target_h, target_w):
                    out = cv2.resize(out, (target_w, target_h),
                                     interpolation=cv2.INTER_LANCZOS4)
                return out, True
            # Only fall back to bicubic for genuine inference-time failures
            # (CUDA/CPU OOM and torch errors raise RuntimeError; bad buffers raise
            # cv2.error/ValueError). Programming errors (AttributeError, TypeError,
            # ZeroDivisionError from a bad bg_scale, ...) are our bugs — let them
            # propagate instead of being silently masked as "use bicubic".
            except (RuntimeError, ValueError, cv2.error, MemoryError) as e:
                logger.warning("AI upscale failed (%s); using bicubic.", e)

        out = cv2.resize(bg, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        return out, False

    def _init_face_enhancer(self):
        """Build the configured face-enhance backend (gfpgan | codeformer).

        Both backends degrade the same way: missing package or weights -> warn
        and skip background-face enhancement (``self._face_enhancer = None``).
        A selected-but-unavailable codeformer never silently falls back to
        gfpgan — the user asked for a specific model and silently substituting
        another would misreport what restored their photo.
        """
        self._tried_face_init = True
        if self.config.face_enhance_backend == "codeformer":
            self._init_codeformer_enhancer()
        else:
            self._init_gfpgan_enhancer()

    def _init_codeformer_enhancer(self):
        url, filename, sha256 = _CODEFORMER_WEIGHTS
        try:
            _ensure_torchvision_compat()
            # Fail on a missing package *before* fetching ~360 MB of weights.
            import codeformer  # noqa: F401
            import facexlib  # noqa: F401

            model_path = str(ensure_weights(url, filename, sha256=sha256))
            self._face_enhancer = _CodeFormerEnhancer(
                model_path, self.config.face_enhance_fidelity
            )
        except ImportError:
            logger.warning(
                "CodeFormer backend selected but not installed; skipping "
                "background-face restore. Install with: pip install "
                "facekeep[ai] facekeep[codeformer]"
            )
            self._face_enhancer = None
        except (ModelDownloadError, RuntimeError, OSError, ValueError, KeyError) as e:
            # Weights download / checksum / load failure (offline, corrupt
            # cache, malformed checkpoint): skip face enhancement instead of
            # failing the whole restore.
            logger.warning(
                "Could not load CodeFormer weights (%s); skipping "
                "background-face restore.", e,
            )
            self._face_enhancer = None

    def _init_gfpgan_enhancer(self):
        url, filename, sha256 = _GFPGAN_WEIGHTS
        try:
            _ensure_torchvision_compat()
            from gfpgan import GFPGANer

            # upscale=1: we already upscaled the background; GFPGAN only needs to
            # restore faces in place. bg_upsampler=None: it must not re-upscale or
            # repaint the non-face background — we keep the Real-ESRGAN/bicubic
            # background and only swap in restored faces (see _enhance...).
            # Fetch + verify into the shared cache and pass the local path (passing
            # None crashes GFPGANer on .startswith; the URL would download unverified).
            model_path = str(ensure_weights(url, filename, sha256=sha256))
            self._face_enhancer = GFPGANer(
                model_path=model_path, upscale=1, arch="clean",
                channel_multiplier=2, bg_upsampler=None,
            )
        except ImportError:
            logger.warning(
                "GFPGAN not installed; reconstructed background faces will not be "
                "restored. Install with: pip install facekeep[ai]"
            )
            self._face_enhancer = None
        except (ModelDownloadError, RuntimeError, OSError, ValueError) as e:
            # Weights download / checksum / load failure (offline, corrupt cache):
            # skip face enhancement instead of failing the whole restore.
            logger.warning(
                "Could not load GFPGAN weights (%s); skipping background-face "
                "restore.", e
            )
            self._face_enhancer = None

    def _enhance_background_faces(self, bg):
        """Restore faces the *detector missed* in the reconstructed background.

        A face that detection missed at compress time was downsampled with the
        background and upscaled here by Real-ESRGAN/bicubic, which tends to melt
        faces into something uncanny — the worst failure for a family tool. The
        configured backend (``face_enhance_backend``: GFPGAN by default,
        CodeFormer opt-in with a ``face_enhance_fidelity`` dial) re-synthesizes
        plausible face detail; both expose the same GFPGANer-shaped surface, so
        everything below is backend-agnostic. We blend only the enhancer's *own*
        detected face regions back with a soft mask (scaled by
        ``face_enhance_strength`` — sub-1.0 softens the "too-perfect face"
        look), leaving the rest of the background (sky/foliage/etc.) exactly as
        the super-resolver produced it — the enhancer must not repaint non-face
        content. Detected faces are real crops composited on top afterward (in
        _composite), so they are never replaced by a hallucination.

        **Bounded memory (ROADMAP backlog).** We call ``enhance(paste_back=False)``,
        which returns only the small 512x512 aligned/restored face crops — GFPGAN's
        own ``paste_back=True`` warps every face into a *full-frame* float buffer
        (observed: a 433 MiB ``(H,W,3) float64`` OOM on a 3840x4929 photo), which the
        ``except`` below would swallow, silently turning this missed-face safety net
        OFF on exactly the large frames where it matters. Instead we paste each
        restored face back *ourselves*, one face at a time, into a buffer sized to
        that face's destination box (``_paste_restored_face``) — peak memory is one
        face, not the frame, so it never OOMs regardless of image size. GFPGAN has no
        tiling API, so this self-paste is the only way to bound it.

        Gated by aggressive.face_enhance; a no-op (returns bg unchanged) when the
        flag is off, ``face_enhance_strength`` is 0, or the selected backend is
        not installed, so restore degrades gracefully.
        """
        if not self.config.face_enhance:
            return bg
        strength = float(self.config.face_enhance_strength)
        if strength <= 0:
            # A zero blend alpha makes the paste a no-op by construction, so
            # skip model init/inference entirely — byte-identical, just cheaper.
            return bg
        if not self._tried_face_init:
            self._init_face_enhancer()
        if self._face_enhancer is None:
            return bg

        try:
            # paste_back=False: GFPGAN aligns + restores each face to a 512x512 crop
            # but does NOT composite (no full-frame buffer). restored_img is None.
            _cropped, restored, _restored_img = self._face_enhancer.enhance(
                bg, has_aligned=False, only_center_face=False, paste_back=False,
            )
        except (RuntimeError, ValueError, cv2.error, MemoryError) as e:
            logger.warning("Face restore failed (%s); using bg as-is.", e)
            return bg

        if not restored:
            # No faces found in the reconstructed background — nothing to restore.
            return bg

        # Inverse-affine matrices map each 512x512 restored crop back to its place
        # in the original frame. GFPGAN computes them in get_inverse_affine (only
        # populated by the paste_back path we skipped), so we trigger it ourselves.
        helper = getattr(self._face_enhancer, "face_helper", None)
        if helper is None:
            logger.warning(
                "Face-enhancer face_helper unavailable; skipping background-face "
                "restore."
            )
            return bg
        try:
            if not getattr(helper, "inverse_affine_matrices", None):
                helper.get_inverse_affine(None)
            inverse_affines = helper.inverse_affine_matrices
        except (AttributeError, cv2.error, ValueError) as e:
            logger.warning(
                "Could not map restored faces back (%s); using bg as-is.", e
            )
            return bg

        if len(inverse_affines) != len(restored):
            logger.warning(
                "Face enhancer returned %d faces but %d affine matrices; "
                "using bg as-is.",
                len(restored), len(inverse_affines),
            )
            return bg

        result = bg
        upscale = float(getattr(self._face_enhancer, "upscale", 1) or 1)
        for restored_face, inverse_affine in zip(restored, inverse_affines):
            result = self._paste_restored_face(
                result, restored_face, inverse_affine, upscale, strength
            )
        return result

    @staticmethod
    def _paste_restored_face(bg, restored_face, inverse_affine, upscale,
                             strength=1.0):
        """Warp + feather one 512x512 restored face back into ``bg`` in place.

        Memory is bounded to the face's destination box: we transform the crop's
        corners through ``inverse_affine`` to find that box, warp only into a
        box-sized buffer (offsetting the affine by the box origin), build a feathered
        square mask the same way (mirroring GFPGAN's own non-parse paste), and blend
        only that slice of ``bg``. No full-frame buffer is ever allocated, so a
        large frame can never OOM here. Best-effort: a degenerate/out-of-frame box
        is skipped (returns ``bg`` unchanged), never raised.
        """
        h, w = bg.shape[:2]
        fh, fw = restored_face.shape[:2]

        # extra_offset matches GFPGAN's paste for sub-pixel back-alignment.
        inv = inverse_affine.astype(np.float64).copy()
        if upscale > 1:
            inv[:, 2] += 0.5 * upscale

        # Destination box = bbox of the four crop corners mapped by the affine.
        corners = np.array(
            [[0, 0], [fw, 0], [fw, fh], [0, fh]], dtype=np.float64
        )
        dst = (corners @ inv[:, :2].T) + inv[:, 2]
        x1 = int(np.floor(dst[:, 0].min()))
        y1 = int(np.floor(dst[:, 1].min()))
        x2 = int(np.ceil(dst[:, 0].max()))
        y2 = int(np.ceil(dst[:, 1].max()))
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return bg  # face maps entirely outside the frame — nothing to paste

        box_w, box_h = x2 - x1, y2 - y1
        # Shift the affine so it warps into the box-local coordinate system: only a
        # box_h x box_w buffer is allocated, not the full frame.
        local = inv.copy()
        local[0, 2] -= x1
        local[1, 2] -= y1

        try:
            inv_restored = cv2.warpAffine(restored_face, local, (box_w, box_h))
            ones = np.ones((fh, fw), dtype=np.float32)
            inv_mask = cv2.warpAffine(ones, local, (box_w, box_h))
        except cv2.error:
            return bg

        # Mirror GFPGAN's non-parse paste: erode the warped mask to drop the warp's
        # black border, then feather the edge with a Gaussian sized to the face area.
        erode_k = max(1, int(2 * upscale))
        inv_mask_erosion = cv2.erode(inv_mask, np.ones((erode_k, erode_k), np.uint8))
        total_face_area = float(inv_mask_erosion.sum())
        if total_face_area <= 0:
            return bg
        w_edge = int(total_face_area ** 0.5) // 20
        erosion_radius = max(1, w_edge * 2)
        inv_mask_center = cv2.erode(
            inv_mask_erosion, np.ones((erosion_radius, erosion_radius), np.uint8)
        )
        blur_size = w_edge * 2
        soft = cv2.GaussianBlur(inv_mask_center, (blur_size + 1, blur_size + 1), 0)
        # aggressive.face_enhance_strength scales the blend alpha: 1.0 keeps
        # full enhancement (multiplying by exactly 1.0 is an IEEE identity, so
        # the default stays byte-identical); ~0.6-0.8 lerps the restored face
        # toward the un-enhanced pixels to soften the "too-perfect face" look.
        soft3 = soft[:, :, None] * float(strength)

        bg_region = bg[y1:y2, x1:x2].astype(np.float32)
        blended = soft3 * inv_restored.astype(np.float32) + (1.0 - soft3) * bg_region
        bg[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
        return bg

    def _composite(self, upscaled_bg, data):
        result = upscaled_bg.copy()
        m = data["manifest"]
        # Region-local conservatism patches first: they restore sharp *background*
        # detail (the area around a small/distant face) over the AI/bicubic
        # upscale, so they go *under* the faces. Absent on older (<1.3.0) files —
        # then regions / region_crops are empty and this loop is a no-op. Reuses
        # the same feathered compositor as faces (a region patch is just a
        # non-face crop), resized to its bbox by the blender.
        for i, region_info in enumerate(m.get("regions", []) or []):
            result = blend_face_onto_background(
                background=result,
                face_crop=data["region_crops"][i],
                face_mask=data["region_masks"][i],
                padded_bbox=tuple(region_info["bbox"]),
                mode=self.config.blend_mode,
            )
        for i, face_info in enumerate(m["faces"]):
            result = blend_face_onto_background(
                background=result,
                face_crop=data["face_crops"][i],
                face_mask=data["face_masks"][i],
                padded_bbox=tuple(face_info["padded_bbox"]),
                mode=self.config.blend_mode,
            )
        return result

    def _write(self, result, output_path, exif, *, icc=None, has_faces=False,
               quality=70):
        """Write the restored image to a standard file, format chosen by suffix.

        Restore is meant to be the "never a dead end" escape hatch (ROADMAP
        Phase 4), so it emits a *standard* image and preserves both EXIF and the
        ICC color profile (e.g. Display P3) on every format that can carry them —
        without the profile a wide-gamut photo restores duller (viewers fall back
        to sRGB), the aggressive-mode counterpart of faithful's ICC preservation.

        * ``.jpg``/``.jpeg``/``.png``/``.webp`` -> **Pillow** ``save`` (not
          ``cv2.imwrite``, which drops ICC): a single encode carrying ``exif=``
          and ``icc_profile=`` together. This *replaces* the OpenCV write, so it
          is not an extra JPEG generation.
        * ``.avif``/``.jxl`` -> the faithful-mode codec (``encoders.encode`` ->
          ``write_encoded``), which produces a *real* AVIF/JXL (OpenCV can't
          write these here) and embeds EXIF *and* ICC through the encoder.
          ``has_faces`` drives ``auto`` chroma to 4:4:4 so restored skin/lips stay
          crisp — the same one face-aware decision faithful mode makes.

        BGR stays internal: the only BGR->RGB conversions are at the PIL boundary
        here and inside ``encoders.encode``, per the repo convention.
        """
        suffix = Path(output_path).suffix.lower()
        if suffix in (".avif", ".jxl"):
            from .. import encoders

            codec = "avif" if suffix == ".avif" else "jxl"
            data = encoders.encode(
                result, codec=codec, quality=quality,
                chroma="auto", has_faces=has_faces, exif=exif, icc=icc,
            )
            encoders.write_encoded(data, output_path, codec)
            return

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        # Pillow write so EXIF *and* the ICC profile are embedded in one save
        # (OpenCV's imwrite carries neither). BGR->RGB at this boundary.
        from PIL import Image

        pil = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        save_kwargs = {}
        if suffix in (".jpg", ".jpeg"):
            # Match the previous cv2.imwrite default JPEG quality (95) so restored
            # file size/quality stays in the same ballpark; this is independent of
            # the avif/jxl `quality` knob above.
            save_kwargs["quality"] = 95
        if icc:
            save_kwargs["icc_profile"] = icc
        if exif:
            save_kwargs["exif"] = exif
        try:
            pil.save(output_path, **save_kwargs)
        except (ValueError, struct.error, OSError) as e:
            # A malformed EXIF/ICC blob must not lose the just-restored pixels:
            # drop the metadata and write the image, warning (consistent with the
            # prior "don't lose pixels on a bad re-embed" stance). Keep only the
            # quality kwarg (safe for JPEG; ignored otherwise).
            logger.warning(
                "Could not embed metadata into %s (%s); writing without it.",
                output_path, e,
            )
            bare_kwargs = {k: save_kwargs[k] for k in ("quality",) if k in save_kwargs}
            pil.save(output_path, **bare_kwargs)

    def restore(self, fkeep_path: str, output_path: Optional[str] = None,
                *, quality: int = 70) -> np.ndarray:
        """Restore a .fkeep file to full resolution using AI super-resolution.

        When ``output_path`` ends in ``.avif``/``.jxl`` the restored image is
        written through the faithful codec at ``quality`` (else OpenCV writes the
        suffix-named JPEG/PNG/WebP). See ``_write``.
        """
        try:
            data = read_fkeep(fkeep_path)
        except Exception as e:  # noqa: BLE001
            raise RestoreError(f"Cannot read {fkeep_path}: {e}") from e

        m = data["manifest"]
        ow, oh = m["original"]["width"], m["original"]["height"]
        bg_scale = m["settings"]["bg_scale"]

        if data.get("residual") is not None:
            # Residual layer (ROADMAP 8.5): the background is real (lossy) data
            # again — bicubic upscale + the stored delta. Skip the AI upscale
            # AND GFPGAN (both exist to make hallucination plausible; a face in
            # real data must not be repainted), which also makes 8.1's anchor /
            # back-projection moot (the low band already is the stored
            # background's). Grain (8.2) below still applies — the residual is
            # half-res + lossy, so the background stays smoother than the crops.
            upscaled = _apply_residual(data["background"], data["residual"],
                                       ow, oh)
        else:
            upscaled, used_ai = self._upscale_background(
                data["background"], ow, oh, bg_scale
            )
            # Low-frequency anchoring (ROADMAP 8.1): only when the AI upsampler
            # actually ran — the bicubic fallback already carries the real low
            # band (anchoring it is an identity up to float rounding), and
            # gating keeps that path byte-identical. Anchor before GFPGAN so the
            # background-face restore sees the corrected tones. Both knobs are
            # restore-only and not in index.settings_fingerprint.
            if used_ai and self.config.restore_anchor:
                upscaled = _anchor_low_frequencies(
                    upscaled, data["background"], bg_scale
                )
            if used_ai and self.config.restore_backproject_iters > 0:
                upscaled = _back_project(
                    upscaled, data["background"],
                    self.config.restore_backproject_iters,
                )
            # Restore any faces the detector missed in the reconstructed
            # background *before* compositing the real face crops, so the
            # original-quality crops land on top and are never replaced by a
            # GFPGAN hallucination.
            upscaled = self._enhance_background_faces(upscaled)
        # Grain matching (ROADMAP 8.2): the real crops carry sensor noise + JPEG
        # texture; the upscale (AI *or* bicubic — both too smooth) does not, and
        # that texture discontinuity is the paste's biggest visible tell. Runs
        # after GFPGAN (its restored faces are smooth the same way) and before
        # compositing (the feather then blends grainy bg into grainy crops).
        # Prefer face crops for the estimate, fall back to region patches; with
        # neither there is no mismatch to hide. Restore-only, not fingerprinted.
        if self.config.restore_grain:
            crops = data["face_crops"] or data["region_crops"]
            grain_sigma = _estimate_grain_sigma(crops)
            if grain_sigma:
                upscaled = _apply_grain(upscaled, grain_sigma)
        result = self._composite(upscaled, data)

        if output_path:
            self._write(result, output_path, data.get("exif"),
                        icc=data.get("icc"),
                        has_faces=bool(m["faces"]), quality=quality)
        return result

    def preview(self, fkeep_path: str, output_path: Optional[str] = None,
                *, quality: int = 70) -> np.ndarray:
        """Quick preview using bicubic upscaling (no AI, fast).

        A stored residual layer IS applied here (unlike GFPGAN/grain, which
        preview skips for speed): the residual is *data* the file carries, not
        an enhancement — a preview that omitted it would misrepresent the
        file's content, and the bench bicubic proxy (preview-based) would hide
        the fidelity win. It costs one resize + add, nothing model-sized.
        """
        data = read_fkeep(fkeep_path)
        m = data["manifest"]
        ow, oh = m["original"]["width"], m["original"]["height"]
        if data.get("residual") is not None:
            upscaled = _apply_residual(data["background"], data["residual"],
                                       ow, oh)
        else:
            upscaled = cv2.resize(data["background"], (ow, oh),
                                  interpolation=cv2.INTER_CUBIC)
        result = self._composite(upscaled, data)
        if output_path:
            self._write(result, output_path, data.get("exif"),
                        icc=data.get("icc"),
                        has_faces=bool(m["faces"]), quality=quality)
        return result

"""Custom exception hierarchy for FaceKeep."""


class FaceKeepError(Exception):
    """Base exception for all FaceKeep errors."""


class ConfigError(FaceKeepError):
    """Raised when configuration is invalid."""


class DetectionError(FaceKeepError):
    """Raised when face/ROI detection fails."""


class EncodingError(FaceKeepError):
    """Raised when image encoding fails (e.g. codec unavailable)."""


class CompressionError(FaceKeepError):
    """Raised when compression fails."""


class RestoreError(FaceKeepError):
    """Raised when restoration fails."""


class ModelDownloadError(FaceKeepError):
    """Raised when a model/weights file cannot be fetched or verified.

    Covers a failed download (offline), a checksum mismatch, or a corrupt
    cache. Callers on the AI restore path catch this and degrade gracefully
    (bicubic upscale / skip face-enhance) — it is never allowed to crash a
    restore, matching the offline-first contract."""


class FormatError(FaceKeepError):
    """Raised when a .fkeep file is malformed or unreadable."""


class SkipFileError(FaceKeepError):
    """Raised to signal a file should be skipped (not a hard error)."""


class VideoError(FaceKeepError):
    """Raised when video probing or re-encoding fails.

    Also raised when the external ffmpeg/ffprobe binaries are unavailable —
    with an install hint, so the CLI surfaces it as a message, not a crash.
    Photos never touch the video path, so they are unaffected."""


class UnsupportedInputError(FaceKeepError):
    """Raised when an input file format is not supported."""

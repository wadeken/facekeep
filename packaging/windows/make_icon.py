"""Generate FaceKeep.ico from the tray icon drawing.

The icon is drawn programmatically (facekeep.app._tray_image), so no binary
asset lives in git — build.ps1 runs this before PyInstaller and the .ico is
gitignored.
"""

from pathlib import Path

from facekeep.app import _tray_image


def main() -> None:
    out = Path(__file__).with_name("FaceKeep.ico")
    img = _tray_image(256)
    img.save(out, sizes=[(16, 16), (24, 24), (32, 32), (48, 48),
                         (64, 64), (128, 128), (256, 256)])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

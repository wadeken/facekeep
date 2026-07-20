# Packaging FaceKeep for Windows (ROADMAP 11.3)

This directory builds the **single-install desktop app**: a tray icon that
keeps an inbox folder compressed into an archive (the `facekeep watch` engine),
opens the drag-and-drop GUI, offers start-with-Windows, and raises done/failed
notifications — so a non-Python user can run FaceKeep.

## Build

```powershell
# From this directory, in Windows PowerShell:
.\build.ps1 -FfmpegDir C:\path\to\ffmpeg\bin [-LibavifDir C:\path\to\libavif\bin]
```

The script:

1. creates `.venv-package/` at the repo root and installs
   `facekeep[app,heic,progress]` + `pyinstaller` + `ssimulacra2` into it —
   **never torch / the `[ai]` extra** (see guardrails below);
2. stages the optional external tools under `stage/tools/` (gitignored);
3. generates `FaceKeep.ico` (`make_icon.py` — drawn with PIL, no binary asset
   in git) and runs PyInstaller on `FaceKeep.spec`;
4. fails the build if any `torch*` artifact appears in the dist;
5. runs the frozen exe's own headless smoke test (`FaceKeep.exe --selftest`,
   report in `%USERPROFILE%\.cache\facekeep\app.log`) and fails the build on a
   non-zero exit;
6. zips `dist/FaceKeep/` into `dist/FaceKeep-win64.zip`.

Optionally build a proper installer from the same dist with
[Inno Setup](https://jrsoftware.org/isinfo.php): `iscc installer.iss` →
`dist/FaceKeep-Setup.exe` (per-user install, no admin).

## What the app is

`FaceKeep.exe` runs `facekeep.app.main` — the tray app (`facekeep app` for pip
users, the `[app]` extra). It is a thin shell over the library:

- **Watch** — the tray toggle drives `cli._watch_cycles`, the *same* loop
  engine as `facekeep watch` (stability guard, metadata-only idle cycles,
  failure memo). Folders are shared with the GUI Backup tab
  (`~/.cache/facekeep/gui_state.json`).
- **Open FaceKeep GUI** (default action) — serves the local Gradio GUI and
  opens the browser. Local-only, sharing/telemetry off.
- **Start with Windows** — an HKCU `Run` registry value (per-user, no admin).
- **Notifications** — one per watch cycle that compressed or failed files;
  idle cycles never notify.
- The windowed exe has no console: all output goes to
  `%USERPROFILE%\.cache\facekeep\app.log`.

## Bundled external tools

The app resolves external binaries exactly like the CLI (`$FACEKEEP_FFMPEG` /
`$FACEKEEP_AVIFENC` → PATH → absent). When frozen, `facekeep.app.
wire_bundled_tools` points those env vars at `tools/` inside the bundle **iff
the user hasn't set them** — bundling changes availability, never precedence.

- **ffmpeg** (`-FfmpegDir`, bundles `ffmpeg.exe` + `ffprobe.exe`): enables
  video compression. **Licensing (Phase-11 guardrail 4): prefer an LGPL
  build.** [BtbN's `ffmpeg-master-latest-win64-lgpl` builds](https://github.com/BtbN/FFmpeg-Builds/releases)
  include everything FaceKeep uses — `libsvtav1` (BSD) and `libvmaf` (BSD) —
  with no GPL components, so the packaged app stays MIT + LGPL (ship the
  build's license text beside the binaries; build.ps1 copies it when found).
  If you bundle a GPL build instead (e.g. gyan.dev's, which adds x264/x265
  that FaceKeep never uses), **the distributed package as a whole must comply
  with the GPL** — do that knowingly or not at all. Omitting `-FfmpegDir`
  builds a photos-only app: videos are skipped with an install hint.
- **libavif CLI** (`-LibavifDir`, optional: `avifenc.exe`/`avifdec.exe`/
  `avifgainmaputil.exe`, BSD): enables 10/12-bit AVIF output, lossless AVIF,
  and HDR gain-map AVIF authoring. Without it those paths degrade exactly as
  a plain install does (warned 8-bit/SDR fallbacks).

## Guardrails (from ROADMAP Phase 11)

- **Never torch.** The `[ai]` extra (aggressive-mode AI restore) stays a
  pip-time opt-in; the build venv never installs it, the spec excludes it,
  and build.ps1 fails if a torch artifact reaches the dist anyway. The tray
  backup flow is faithful-only — aggressive mode lives in the GUI's Compress
  tab where its trade-off is explained.
- **Sources are never deleted or modified**; the guardrail-2 honesty note
  (visually lossless ≠ bit-exact, with a Lossless menu toggle) is raised when
  watching starts.

## Known facts / troubleshooting

- The exe is **unsigned**: SmartScreen will warn on first run ("More info" →
  "Run anyway"). Code signing is a distribution follow-up, not a build step.
- Gradio needs its data files and source collected (`collect_data_files` +
  `module_collection_mode={"gradio": "py"}` in the spec); a missing-asset
  regression is exactly what the `--selftest` gate catches (it builds the
  real Blocks).
- macOS packaging (signing + notarization need Apple credentials) is scoped
  as its own follow-up once this Windows shape is proven — see the ROADMAP.

# Build the packaged FaceKeep Windows tray app (ROADMAP 11.3).
#
#   .\build.ps1 -FfmpegDir C:\path\to\ffmpeg\bin [-LibavifDir C:\path\to\libavif]
#
# Produces dist\FaceKeep\FaceKeep.exe (+ dist\FaceKeep-win64.zip) and gates the
# result on the frozen exe's own headless smoke test (FaceKeep.exe --selftest).
# See README.md in this directory for the ffmpeg licensing note (prefer an
# LGPL build) and the full story. Windows PowerShell 5.1 compatible.

param(
    # Directory containing ffmpeg.exe + ffprobe.exe to bundle (guardrail 4:
    # prefer an LGPL build — see README.md). Omit to build without video
    # support (videos are then skipped with an install hint; photos work).
    [string]$FfmpegDir = "",
    # Optional: directory containing avifenc.exe/avifdec.exe/avifgainmaputil.exe
    # (libavif CLI tools) to enable 10/12-bit AVIF + HDR gain-map output.
    [string]$LibavifDir = "",
    # Base interpreter used to create the build venv.
    [string]$Python = "python",
    # Reuse an existing .venv-package instead of rebuilding it.
    [switch]$SkipVenv
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = (Resolve-Path (Join-Path $here "..\..")).Path
$venv = Join-Path $repo ".venv-package"
$py = Join-Path $venv "Scripts\python.exe"

# 1. Build venv: exactly the [app] surface (+ heic input + progress bar) +
#    pyinstaller + the ssimulacra2 auto-tune metric. NEVER torch/[ai]
#    (guardrail 4): this venv structurally cannot leak it into the bundle,
#    and the spec's excludes are the second line of defense.
if (-not $SkipVenv -or -not (Test-Path $py)) {
    & $Python -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }
    & $py -m pip install --upgrade pip
    & $py -m pip install "$repo[app,heic,progress]" pyinstaller ssimulacra2
    if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
}

# 2. Stage the bundled tools (optional, gitignored).
$stage = Join-Path $here "stage"
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
if ($FfmpegDir) {
    $dst = Join-Path $stage "tools\ffmpeg"
    New-Item -ItemType Directory -Force $dst | Out-Null
    Copy-Item (Join-Path $FfmpegDir "ffmpeg.exe") $dst
    Copy-Item (Join-Path $FfmpegDir "ffprobe.exe") $dst
    # Ship the build's license text beside the binaries when present.
    foreach ($lic in "LICENSE", "LICENSE.txt", "COPYING.LGPLv2.1", "COPYING.GPLv2") {
        $p = Join-Path (Split-Path -Parent $FfmpegDir) $lic
        if (Test-Path $p) { Copy-Item $p $dst }
    }
}
if ($LibavifDir) {
    $dst = Join-Path $stage "tools\libavif"
    New-Item -ItemType Directory -Force $dst | Out-Null
    foreach ($t in "avifenc.exe", "avifdec.exe", "avifgainmaputil.exe") {
        $p = Join-Path $LibavifDir $t
        if (Test-Path $p) { Copy-Item $p $dst }
    }
}

# 3. Icon (generated, gitignored) + the PyInstaller build.
& $py (Join-Path $here "make_icon.py")
if ($LASTEXITCODE -ne 0) { throw "icon generation failed" }
Push-Location $here
try {
    & $py -m PyInstaller --noconfirm --clean FaceKeep.spec
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }
} finally {
    Pop-Location
}
$dist = Join-Path $here "dist\FaceKeep"

# 4. Guardrail 4 check: never torch in the shipped bundle.
$torch = Get-ChildItem -Recurse $dist -Filter "torch*" -ErrorAction SilentlyContinue |
    Select-Object -First 1
if ($torch) { throw "guardrail 4 violated: torch found in dist ($($torch.FullName))" }

# 5. Smoke test: the frozen exe must pass its own selftest (it exercises the
#    tray menu build, the Gradio Blocks build — which catches missing bundled
#    assets — and reports the wired tools). The windowed exe has no console;
#    its report lands in ~\.cache\facekeep\app.log.
$exe = Join-Path $dist "FaceKeep.exe"
$proc = Start-Process -FilePath $exe -ArgumentList "--selftest" -Wait -PassThru
$log = Join-Path $env:USERPROFILE ".cache\facekeep\app.log"
if (Test-Path $log) { Get-Content $log | Write-Host }
if ($proc.ExitCode -ne 0) {
    throw "FaceKeep.exe --selftest failed (exit $($proc.ExitCode)) - see $log"
}

# 6. Zip artifact (the minimum single-download deliverable; installer.iss
#    builds a proper installer from the same dist if Inno Setup is present).
$zip = Join-Path $here "dist\FaceKeep-win64.zip"
if (Test-Path $zip) { Remove-Item $zip }
Compress-Archive -Path $dist -DestinationPath $zip
Write-Host "Built: $zip"
Write-Host "Run:   $exe"

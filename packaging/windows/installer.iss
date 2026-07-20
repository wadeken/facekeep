; Inno Setup script for FaceKeep (optional; build.ps1's zip is the minimum
; artifact). Build AFTER build.ps1 has produced dist\FaceKeep:
;
;   iscc installer.iss
;
; -> dist\FaceKeep-Setup.exe. Per-user install (no admin), Start-menu entry,
; optional desktop icon. The app itself offers "Start with Windows" from its
; tray menu (an HKCU Run value), so the installer registers nothing.

#define MyAppName "FaceKeep"
; Keep in sync with facekeep.__version__.
#define MyAppVersion "0.2.0"
#define MyAppExeName "FaceKeep.exe"

[Setup]
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher=FaceKeep
AppPublisherURL=https://github.com/wadeken/facekeep
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
OutputDir=dist
OutputBaseFilename=FaceKeep-Setup
Compression=lzma2
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64
; Per-user install: no UAC prompt, lands under %LocalAppData%\Programs.
PrivilegesRequired=lowest
UninstallDisplayIcon={app}\{#MyAppExeName}

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop icon"; Flags: unchecked

[Files]
Source: "dist\FaceKeep\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent

; Jarvis F1 Telemetry Suite - Inno Setup Installer Script
;
; Build with:
;   "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" scripts\jarvis_installer.iss
;
; Requires: PyInstaller build output in dist\Jarvis\

#define MyAppName "Jarvis F1 Telemetry"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "Jarvis F1"
#define MyAppExeName "Jarvis.exe"

[Setup]
AppId={{8F2B3A1C-4D5E-6F7A-8B9C-0D1E2F3A4B5C}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\Programs\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=..\dist
OutputBaseFilename=Jarvis-Setup
Compression=lzma2/max
SolidCompression=yes
PrivilegesRequired=lowest
SetupIconFile=..\ui\img\jarvis.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
WizardStyle=modern

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Files]
; Bundle everything from the PyInstaller output
Source: "..\dist\Jarvis\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Clean up runtime data created by the app
Type: filesandordirs; Name: "{app}\race_engineer_gguf"
Type: filesandordirs; Name: "{app}\postrace_gguf"
Type: filesandordirs; Name: "{app}\data"
Type: filesandordirs; Name: "{app}\config.json"

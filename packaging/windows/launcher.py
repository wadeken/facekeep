"""PyInstaller entry point for the packaged FaceKeep tray app (ROADMAP 11.3).

The frozen ``FaceKeep.exe`` runs the tray app (:func:`facekeep.app.main`);
``FaceKeep.exe --selftest`` is the headless packaging smoke test build.ps1
gates the build on.
"""

import multiprocessing

from facekeep.app import main

if __name__ == "__main__":
    # Frozen-Windows discipline: without freeze_support a spawned worker
    # process would re-run the app instead of the worker bootstrap. The tray
    # watch runs jobs=1 (no pool) today, but the guard is standard and cheap.
    multiprocessing.freeze_support()
    main()

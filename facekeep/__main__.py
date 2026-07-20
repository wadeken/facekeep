"""``python -m facekeep`` — the CLI entry point.

Exists so the tray app's start-with-Windows command (``pythonw -m facekeep
app``, see :func:`facekeep.app.startup_command`) works on a pip install with
no console script on PATH.
"""

from .cli import main

if __name__ == "__main__":
    main()

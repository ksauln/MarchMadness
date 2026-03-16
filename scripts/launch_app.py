from __future__ import annotations

import sys
from pathlib import Path

from streamlit.web.cli import main as streamlit_main

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "src" / "march_madness" / "ui" / "app.py"


if __name__ == "__main__":
    sys.argv = ["streamlit", "run", str(APP_PATH), *sys.argv[1:]]
    raise SystemExit(streamlit_main())

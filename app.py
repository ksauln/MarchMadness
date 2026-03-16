from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _running_under_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:
        return False
    return get_script_run_ctx() is not None


def _launch_streamlit() -> int:
    from streamlit.web.cli import main as streamlit_main

    sys.argv = ["streamlit", "run", str(ROOT / "app.py"), *sys.argv[1:]]
    return streamlit_main()


if __name__ == "__main__" and not _running_under_streamlit():
    raise SystemExit(_launch_streamlit())


from march_madness.ui import app as _app  # noqa: F401

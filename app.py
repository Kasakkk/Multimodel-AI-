import os
import subprocess
import sys


def main() -> int:
    """
    Compatibility launcher for users who still run `python app.py`.
    """
    app_file = os.path.join(os.path.dirname(__file__), "streamlit_app.py")
    if not os.path.exists(app_file):
        print("Error: streamlit_app.py not found.")
        return 1

    cmd = [sys.executable, "-m", "streamlit", "run", app_file]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())

"""Run helper scripts with project root on PYTHONPATH."""

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run.py <script> [args...]")
        print("Example: python run.py scripts/seed_db.py")
        raise SystemExit(1)
    target = ROOT / sys.argv[1]
    sys.argv = [str(target), *sys.argv[2:]]
    runpy.run_path(str(target), run_name="__main__")

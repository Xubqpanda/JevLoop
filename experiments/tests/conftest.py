"""让 pytest 无论从哪个目录启动都能 import `experiments`。"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]  # JevLoop/
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

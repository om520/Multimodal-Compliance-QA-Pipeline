"""
conftest.py (root) – Ensure Desktop/multimodel is on sys.path so that
complianceQApipeline.* imports resolve when running pytest from the
complianceQApipeline/ directory.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent  # Desktop/multimodel/
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

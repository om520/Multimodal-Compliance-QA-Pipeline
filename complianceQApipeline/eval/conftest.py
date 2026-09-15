"""
conftest.py – Ensure the repo root (Desktop/multimodel) is on sys.path so that
`complianceQApipeline.*` imports work when pytest is run from the
`complianceQApipeline/` subdirectory.
"""

import sys
from pathlib import Path

# When running: python -m pytest eval/...  from complianceQApipeline/
# we need  Desktop/multimodel/  on sys.path so "complianceQApipeline" is resolvable.
_REPO_ROOT = Path(__file__).resolve().parent.parent  # Desktop/multimodel/
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

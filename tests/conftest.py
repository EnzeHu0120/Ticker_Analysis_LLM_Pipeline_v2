from __future__ import annotations

import sys
from pathlib import Path


# Ensure project root (where fundamental_pipeline.py, technical_pipeline.py, llm_pipeline.py live)
# is on sys.path for all tests, so `import technical_pipeline` etc. work regardless of how tests
# are invoked (pytest or running a single test file directly).
ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)


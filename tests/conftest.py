"""
Shared pytest setup.

Two jobs:
1. Put src/ on sys.path so tests can `import config`, `import utils`, etc.
   the same flat way the pipeline modules import each other.
2. Stub out dependencies that are heavy, require a GPU, or hit the network
   to download model weights (torch, torchaudio, cv2, qwen_asr,
   sentence_transformers). Unit tests should never need a GPU or internet
   access to run - individual tests inject their own fakes/mocks with
   whatever behaviour they need to verify.

If the real packages happen to be installed (e.g. in an environment set
up from requirements.txt), the stubs still take priority for the test
run so behaviour is consistent and fast everywhere.
"""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))


def _stub_module(name):
    module = types.ModuleType(name)
    mock = MagicMock(name=f"stub:{name}")

    def __getattr__(attr, _mock=mock):
        return getattr(_mock, attr)

    module.__getattr__ = __getattr__
    sys.modules[name] = module
    return module


for _name in ("torch", "torchaudio", "cv2", "qwen_asr", "sentence_transformers"):
    _stub_module(_name)
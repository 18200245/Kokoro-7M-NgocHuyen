import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PATCHED = os.path.join(_HERE, "kokoro_patched")

for _p in [_HERE, _PATCHED]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

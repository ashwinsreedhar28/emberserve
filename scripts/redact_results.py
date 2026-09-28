"""Blank the `api_key` a sweep JSON recorded before the runner learned to redact it.

    python scripts/redact_results.py results/*.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    for name in sys.argv[1:]:
        p = Path(name)
        d = json.loads(p.read_text())
        a = d.get("args") or {}
        if isinstance(a, dict) and a.get("api_key") not in (None, "x", "<redacted>"):
            a["api_key"] = "<redacted>"
            p.write_text(json.dumps(d, indent=1))
            print(f"redacted {p}")


if __name__ == "__main__":
    main()

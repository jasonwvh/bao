from __future__ import annotations

import platform
from importlib.metadata import PackageNotFoundError, version
from typing import Dict, Iterable, List, Tuple


TRACKED_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scikit-learn": "scikit-learn",
    "torch": "torch",
}


def collect_library_versions() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for key, pkg_name in TRACKED_PACKAGES.items():
        try:
            out[key] = str(version(pkg_name))
        except PackageNotFoundError:
            out[key] = "not_installed"
    out["python"] = platform.python_version()
    return out


def compare_versions(
    expected: Dict[str, str] | None,
    actual: Dict[str, str],
    keys: Iterable[str] | None = None,
) -> Tuple[bool, List[str]]:
    if not isinstance(expected, dict):
        return True, []

    compare_keys = list(keys) if keys is not None else list(TRACKED_PACKAGES.keys())
    mismatches: List[str] = []
    for key in compare_keys:
        e = str(expected.get(key, "")).strip()
        a = str(actual.get(key, "")).strip()
        if e and a and e != a:
            mismatches.append(f"{key}: expected={e}, actual={a}")
    return len(mismatches) == 0, mismatches

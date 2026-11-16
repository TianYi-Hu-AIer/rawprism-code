"""Audit scene-level splits against per-scene result records.

Contract (used by tests/test_splits.py):
    audit(split_path, results_path) -> dict with:
        passed: bool          — no split errors, no duplicate scenes per method,
                                every method's scenes exist in exactly one split
        methods: {name: {unique_count: int, intersections: {train, val, test}}}
        split_errors: [...]   — malformed split entries
"""

import json
from pathlib import Path


def _split_ids(split):
    ids = {}
    errors = []
    seen_in = {}
    for key, entries in split.items():
        bucket = set()
        for entry in entries:
            scene = entry["scene"] if isinstance(entry, dict) else entry
            if not isinstance(scene, str) or not scene:
                errors.append("malformed entry in {}".format(key))
                continue
            bucket.add(scene)
            if scene in seen_in and seen_in[scene] != key:
                errors.append("scene {} appears in both {} and {}".format(
                    scene, seen_in[scene], key))
            seen_in[scene] = key
        ids[key] = bucket
    return ids, errors


def audit(split_path, results_path):
    split_path, results_path = Path(split_path), Path(results_path)
    split = json.loads(split_path.read_text())["splits"]
    results = json.loads(results_path.read_text())

    ids, split_errors = _split_ids(split)

    methods = {}
    for row in results:
        name = row["method"]
        methods.setdefault(name, []).append(row["scene"])

    passed = not split_errors
    method_report = {}
    for name, scenes in methods.items():
        seen = set()
        duplicates = False
        for scene in scenes:
            if scene in seen:
                duplicates = True
            seen.add(scene)
        intersections = {
            key: (seen & ids.get(key, set())) for key in ids
        }
        unknown = len(seen) - sum(len(v) for v in intersections.values())
        ok = (not duplicates) and unknown == 0
        passed = passed and ok
        method_report[name] = {
            "unique_count": len(seen),
            "intersections": intersections,
            "duplicates": duplicates,
            "unknown_scenes": unknown,
        }
    return {
        "passed": passed,
        "methods": method_report,
        "split_errors": split_errors,
    }


if __name__ == "__main__":
    import sys
    report = audit(sys.argv[1], sys.argv[2])
    print(json.dumps(report, indent=2)[:2000])

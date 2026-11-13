#!/usr/bin/env python3
"""Resolve SPEED-Bench "qualitative" with NVIDIA's specdec_bench, skipping
gated sources.

The Hugging Face copy masks part of the rows; specdec_bench fetches them
from their original datasets. One source, cais/hle, is gated on the Hub. To
keep the prompt set reproducible without gated access, rows from skipped
sources are dropped (not left masked) and listed in a sidecar JSON.

Run from anywhere, with specdec_bench's requirements installed:

  python resolve_speed.py --specdec-dir mo/examples/specdec_bench \
      --out data/speed-qualitative.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--specdec-dir", required=True, type=Path,
                    help="Model-Optimizer/examples/specdec_bench checkout")
    ap.add_argument("--skip-source", action="append", default=["cais/hle"],
                    help="source substring to drop instead of resolving (repeatable)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    sys.path.insert(0, str(args.specdec_dir.resolve()))
    from specdec_bench.datasets import speed as sp  # noqa: E402

    cls = sp.SPEEDBench
    orig = cls._fetch_all_turns_data

    def fetch(self, example, speed_config):
        if any(s in (example.get("source") or "") for s in args.skip_source):
            return example  # left masked; dropped below
        return orig(self, example, speed_config)

    cls._fetch_all_turns_data = fetch
    tmp = args.out.parent / "_specdec_prepare"
    path = cls.prepare_data(output_dir=tmp, config_name="qualitative")

    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = pq.read_table(path).to_pylist()
    kept, dropped = [], []
    for r in rows:
        masked = any(t.startswith(sp.TURNS_PLACEHOLDER) for t in r["turns"])
        skipped = any(s in (r.get("source") or "") for s in args.skip_source)
        if masked and skipped:
            dropped.append({"question_id": r["question_id"], "category": r["category"],
                            "source": r["source"]})
        elif masked:
            sys.exit(f"unresolved row {r['question_id']} from {r['source']}")
        else:
            kept.append(r)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(kept), args.out)
    side = {"rows": len(rows), "kept": len(kept), "dropped": dropped,
            "skip_source": args.skip_source}
    Path(str(args.out) + ".dropped.json").write_text(json.dumps(side, indent=1))
    print(f"kept {len(kept)} of {len(rows)} rows, dropped {len(dropped)} from {args.skip_source}")


if __name__ == "__main__":
    main()

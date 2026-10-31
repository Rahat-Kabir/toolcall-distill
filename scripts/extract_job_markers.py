#!/usr/bin/env python3
"""Extract BEGIN/END markers from hf jobs logs."""
from __future__ import annotations
import argparse, base64, gzip, json, re, subprocess
from pathlib import Path

def fetch_logs(job_id: str) -> str:
    if "/" not in job_id:
        raise ValueError("Use a fully qualified job reference: account/job_id")
    return subprocess.check_output(["hf", "jobs", "logs", job_id], text=True, stderr=subprocess.STDOUT)

def extract_blocks(text: str, marker: str):
    pattern = re.compile(rf"{re.escape(marker)}_BEGIN\n(.*?)\n{re.escape(marker)}_END", re.S)
    return pattern.findall(text)

def maybe_json(s: str):
    s = s.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job_id", help="Fully qualified account/job_id")
    ap.add_argument("--marker", action="append", default=[])
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--decode-b64gz", action="append", default=[], help="marker whose body is b64 gz text")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    text = fetch_logs(args.job_id)
    (args.out_dir / "raw_logs.txt").write_text(text)
    for marker in args.marker:
        blocks = extract_blocks(text, marker)
        for i, body in enumerate(blocks):
            suffix = "" if len(blocks) == 1 else f"_{i}"
            if marker in args.decode_b64gz:
                raw = gzip.decompress(base64.b64decode(body.strip()))
                path = args.out_dir / f"{marker.lower()}{suffix}.jsonl"
                path.write_bytes(raw)
            else:
                obj = maybe_json(body)
                path = args.out_dir / f"{marker.lower()}{suffix}.json"
                if isinstance(obj, (dict, list)):
                    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
                else:
                    path.write_text(str(obj) + "\n")
            print("wrote", path, "bytes", path.stat().st_size)
    if not args.marker:
        found = sorted(set(re.findall(r"([A-Z0-9_]+)_BEGIN", text)))
        print("markers:", found)

if __name__ == "__main__":
    main()

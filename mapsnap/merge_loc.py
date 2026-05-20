import json
import sys
from pathlib import Path

if __name__ == "__main__":
    in_glob, out = sys.argv[1:]
    results = []
    for p in Path().glob(in_glob):
        with open(p) as f:
            response = json.load(f)
        results += response["results"]
    with open(out, "w") as f:
        json.dump(results, f)
    print(f"Wrote {len(results)} Sanborn items to {out}")

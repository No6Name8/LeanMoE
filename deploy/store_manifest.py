"""Private transfer manifest, using SHA-256; never downloads or publishes weights.

create: hash existing store files and accompanying notices without modifying them.
verify: check the transferred directory against that independently transferred manifest.
"""
import argparse
import hashlib
import json
from pathlib import Path

FILES = ("store.json", "experts.bin", "resident.bin", "experts_index.json", "resident_index.json",
         "tokenizer.json", "tokenizer_config.json")
NOTICES = ("LICENSE", "NOTICE", "CHANGES", "provenance.json")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def create(root, notices):
    rows = []
    for base, names in ((root, FILES), (notices, NOTICES)):
        for name in names:
            path = base / name
            rows.append({"name": name, "bytes": path.stat().st_size, "sha256": digest(path)})
    return {"model": "Qwen/Qwen1.5-MoE-A2.7B-Chat",
            "revision": "ec052fda178e241c7c443468d2fa1db6618996be", "files": rows}


def verify(root, manifest):
    rows = manifest["files"]
    names = [r["name"] for r in rows]
    if len(names) != len(set(names)) or set(names) != set(FILES + NOTICES):
        raise ValueError("manifest must contain exactly the expected store and notice files")
    for row in rows:
        path = root / row["name"]
        if path.is_symlink() or path.resolve().parent != root.resolve():
            raise ValueError("store file escapes the transfer directory")
        if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            raise ValueError(f"integrity check failed: {row['name']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("create", "verify"))
    ap.add_argument("--store-root", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--notices", type=Path, default=Path(__file__).parent / "model-notices")
    a = ap.parse_args()
    if a.action == "create":
        if a.manifest.exists():
            ap.error("refusing to overwrite an existing transfer manifest")
        a.manifest.write_text(json.dumps(create(a.store_root, a.notices), indent=2) + "\n", encoding="utf-8")
    else:
        verify(a.store_root, json.loads(a.manifest.read_text(encoding="utf-8-sig")))
    print("PASS: transfer manifest " + a.action)


if __name__ == "__main__":
    main()

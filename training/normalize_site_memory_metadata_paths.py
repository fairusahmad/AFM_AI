import argparse
import json
from pathlib import Path
from pathlib import PurePosixPath


def normalize_relpath(value):
    if not value:
        return value
    return PurePosixPath(str(value).replace("\\", "/")).as_posix()


def normalize_metadata_tree(node):
    if isinstance(node, dict):
        normalized = {}
        for key, value in node.items():
            if key.endswith("_path"):
                normalized[key] = normalize_relpath(value)
            elif key in {"lowmag_landmarks", "highmag_landmarks"} and isinstance(value, list):
                items = []
                for item in value:
                    if isinstance(item, dict) and "patch_path" in item:
                        updated = dict(item)
                        updated["patch_path"] = normalize_relpath(updated["patch_path"])
                        items.append(updated)
                    else:
                        items.append(item)
                normalized[key] = items
            else:
                normalized[key] = normalize_metadata_tree(value)
        return normalized
    if isinstance(node, list):
        return [normalize_metadata_tree(item) for item in node]
    return node


def normalize_site_memory_root(site_memory_root):
    site_memory_root = Path(site_memory_root)
    updated = 0
    for metadata_path in sorted(site_memory_root.rglob("metadata.json")):
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
        normalized = normalize_metadata_tree(data)
        if normalized != data:
            metadata_path.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
            updated += 1
    return updated


def main():
    parser = argparse.ArgumentParser(description="Normalize stored metadata paths for cross-platform site-memory loading.")
    parser.add_argument(
        "--site-memory-root",
        default="collected_data/site_memories",
        help="Root containing metadata.json files to normalize.",
    )
    args = parser.parse_args()
    updated = normalize_site_memory_root(args.site_memory_root)
    print(f"Normalized metadata files: {updated}")


if __name__ == "__main__":
    main()

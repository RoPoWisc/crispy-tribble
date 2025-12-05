from __future__ import annotations

import json
import re
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "ad_copy_reviews"


def extract_prefix(name: str) -> str | None:
    """Return leading token like '_1' from a filename, ensuring whole-number match."""
    match = re.match(r"^(_\d+)\b", name)
    return match.group(1) if match else None


def org_files(prefix: str, base: Path) -> list[str]:
    """Return all sibling files that share the numeric prefix, excluding the .doc itself."""
    candidates: list[Path] = []
    for path in base.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() == ".doc":
            continue
        candidate_prefix = extract_prefix(path.name)
        if candidate_prefix == prefix:
            candidates.append(path)
    candidates.sort()
    return [p.relative_to(ROOT_DIR).as_posix() for p in candidates]


def build_index(base: Path) -> list[dict[str, str | None]]:
    items: list[dict[str, str | None]] = []
    for doc in sorted(base.glob("*.doc")):
        prefix = extract_prefix(doc.name)
        if not prefix:
            continue
        converted = doc.with_suffix(".docx").relative_to(ROOT_DIR).as_posix()
        items.append(
            {
                "name": doc.stem,
                "doc": doc.relative_to(ROOT_DIR).as_posix(),
                "org_files": org_files(prefix, base),
                "converted": converted,
            }
        )
    return items


def main() -> None:
    index = build_index(DATA_DIR)
    outfile = DATA_DIR / "index.json"
    outfile.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"Wrote {outfile}")


if __name__ == "__main__":
    main()

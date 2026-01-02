"""Count how many times each rule appears in a specDetect4ai_results.json file."""
import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable


def flatten_counts(data: Dict[str, Dict[str, Iterable[str]]]) -> Counter:
    """Return a Counter of rule -> total occurrences across all files."""
    counter: Counter[str] = Counter()
    for file_rules in data.values():
        for rule, findings in file_rules.items():
            counter[rule] += len(findings)
    return counter


def main() -> None:
    parser = argparse.ArgumentParser(description="Count rule occurrences in specDetect4ai results JSON.")
    parser.add_argument(
        "json_path",
        nargs="?",
        default="specDetect4ai_results.json",
        help="Path to specDetect4ai_results.json (default: specDetect4ai_results.json)",
    )
    args = parser.parse_args()

    path = Path(args.json_path)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    data = json.loads(path.read_text())
    counts = flatten_counts(data)

    for rule, count in counts.most_common():
        print(f"{rule}: {count}")

    total = sum(counts.values())
    print(f"Total: {total}")


if __name__ == "__main__":
    main()

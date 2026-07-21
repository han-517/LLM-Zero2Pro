from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

from student_pipeline import PipelineConfig, RawDocument, build_data_card, process_documents

PROJECT_ROOT = Path(__file__).resolve().parent


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the offline auditable data pipeline")
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data" / "raw_documents.jsonl")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts")
    return parser


def _load_records(path: Path) -> list[RawDocument]:
    records: list[RawDocument] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(RawDocument(**json.loads(line)))
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid JSONL record at line {line_number}") from exc
    return records


def _write_jsonl(path: Path, rows: list[object]) -> None:
    text = "\n".join(json.dumps(asdict(row), ensure_ascii=False) for row in rows)
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def main() -> int:
    args = _build_parser().parse_args()
    records = _load_records(args.input)
    processed, events = process_documents(records, PipelineConfig())
    data_card = build_data_card(records, processed, events)
    data_card["input_sha256"] = sha256(args.input.read_bytes()).hexdigest()
    data_card["pipeline_version"] = 1

    args.output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output / "processed.jsonl", processed)
    _write_jsonl(args.output / "audit.jsonl", events)
    (args.output / "data_card.json").write_text(
        json.dumps(data_card, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(data_card, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

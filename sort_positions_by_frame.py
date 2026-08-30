from __future__ import annotations

import argparse
import csv
import heapq
import tempfile
from pathlib import Path


def sort_key(row: dict[str, str]) -> tuple[int, str, str]:
    return (
        int(row["frame"]),
        row["entity_type"],
        row["person_id"],
    )


def write_sorted_chunk(rows: list[dict[str, str]], fieldnames: list[str], temp_dir: Path, index: int) -> Path:
    rows.sort(key=sort_key)
    chunk_path = temp_dir / f"chunk_{index:05d}.csv"
    with chunk_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return chunk_path


def chunked_sort(input_csv: Path, output_csv: Path, chunk_size: int) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="frame_sort_", dir=output_csv.parent) as temp_name:
        temp_dir = Path(temp_name)
        chunk_paths: list[Path] = []

        with input_csv.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            if not fieldnames:
                raise SystemExit(f"No CSV header found in {input_csv}")

            rows: list[dict[str, str]] = []
            for index, row in enumerate(reader, start=1):
                rows.append(row)
                if len(rows) >= chunk_size:
                    chunk_paths.append(write_sorted_chunk(rows, fieldnames, temp_dir, len(chunk_paths)))
                    rows = []

            if rows:
                chunk_paths.append(write_sorted_chunk(rows, fieldnames, temp_dir, len(chunk_paths)))

        readers = []
        heap: list[tuple[tuple[int, str, str], int, dict[str, str]]] = []

        try:
            for chunk_index, chunk_path in enumerate(chunk_paths):
                handle = chunk_path.open(newline="", encoding="utf-8")
                reader = csv.DictReader(handle)
                readers.append((handle, reader))
                row = next(reader, None)
                if row is not None:
                    heapq.heappush(heap, (sort_key(row), chunk_index, row))

            with output_csv.open("w", newline="", encoding="utf-8") as out_handle:
                writer = csv.DictWriter(out_handle, fieldnames=fieldnames)
                writer.writeheader()

                while heap:
                    _, chunk_index, row = heapq.heappop(heap)
                    writer.writerow(row)
                    next_row = next(readers[chunk_index][1], None)
                    if next_row is not None:
                        heapq.heappush(heap, (sort_key(next_row), chunk_index, next_row))
        finally:
            for handle, _ in readers:
                handle.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Externally sort a Dataset A positions CSV by frame.")
    parser.add_argument("input_csv", type=Path, help="Input CSV path")
    parser.add_argument("output_csv", type=Path, help="Output CSV path")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200_000,
        help="Rows per in-memory chunk before spilling to disk",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunked_sort(args.input_csv, args.output_csv, args.chunk_size)
    print(f"Wrote frame-sorted CSV to {args.output_csv}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Count YOLO label files and instances for few-shot dataset splits.

A YOLO label instance is counted as one non-empty line in a ``.txt`` file.
By default, this script checks the three project datasets for 1/5/10-shot and
seed0 through seed4. Results are printed as a Markdown table and can also be
saved as CSV and Markdown files.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, pstdev
from typing import Iterable, Sequence


DATASETS = (
    (
        "1000_original_6class_0improve",
        Path(
            "datasets/1000_original_6class_0improve/y/fewshot/"
            "fewshot_from_train_yolo"
        ),
    ),
    (
        "coco_PCB",
        Path("datasets/coco_PCB/fewshot/fewshot_from_train_yolo"),
    ),
    (
        "NEU-DET",
        Path(
            "datasets/NEU-DET/star-1/coco/fewshot/"
            "fewshot_from_train_yolo"
        ),
    ),
)
DEFAULT_SHOTS = ("1shot", "5shot", "10shot")
DEFAULT_SEEDS = ("seed0", "seed1", "seed2", "seed3", "seed4")
NUM_CLASSES = 6


@dataclass(frozen=True)
class Statistics:
    dataset: str
    shot: str
    seed: str
    txt_files: int
    label_instances: int
    class_instances: tuple[int, ...]
    empty_files: int
    malformed_lines: int


@dataclass(frozen=True)
class ClassSummary:
    dataset: str
    shot: str
    # One (mean, population standard deviation) pair for each class ID.
    class_mean_std: tuple[tuple[float, float], ...]


def count_labels(
    dataset: str, shot: str, seed: str, labels_dir: Path
) -> Statistics:
    """Return statistics for one dataset/shot/seed labels directory."""
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"标签目录不存在: {labels_dir}")

    txt_paths = sorted(path for path in labels_dir.iterdir() if path.is_file()
                       and path.suffix.lower() == ".txt")
    label_instances = 0
    class_instances = [0] * NUM_CLASSES
    empty_files = 0
    malformed_lines = 0

    for txt_path in txt_paths:
        file_instances = 0
        with txt_path.open("r", encoding="utf-8") as label_file:
            for line in label_file:
                stripped = line.strip()
                if not stripped:
                    continue
                file_instances += 1
                # A YOLO detection row has at least: class x_center y_center w h.
                fields = stripped.split()
                if len(fields) < 5:
                    malformed_lines += 1
                    continue
                try:
                    class_id = int(fields[0])
                except ValueError:
                    malformed_lines += 1
                    continue
                if not 0 <= class_id < NUM_CLASSES:
                    malformed_lines += 1
                    continue
                class_instances[class_id] += 1
        label_instances += file_instances
        if file_instances == 0:
            empty_files += 1

    return Statistics(
        dataset=dataset,
        shot=shot,
        seed=seed,
        txt_files=len(txt_paths),
        label_instances=label_instances,
        class_instances=tuple(class_instances),
        empty_files=empty_files,
        malformed_lines=malformed_lines,
    )


def summarize_classes(rows: Sequence[Statistics]) -> list[ClassSummary]:
    """Aggregate per-class counts as mean and population std across seeds."""
    grouped: dict[tuple[str, str], list[Statistics]] = {}
    for row in rows:
        grouped.setdefault((row.dataset, row.shot), []).append(row)

    summaries = []
    for (dataset, shot), seed_rows in grouped.items():
        mean_std = []
        for class_id in range(NUM_CLASSES):
            values = [row.class_instances[class_id] for row in seed_rows]
            mean_std.append((fmean(values), pstdev(values)))
        summaries.append(ClassSummary(dataset, shot, tuple(mean_std)))
    return summaries


def seed_table_markdown(rows: Sequence[Statistics]) -> str:
    """Format per-seed totals as a Markdown table."""
    lines = [
        "| 数据集 | Shot | Seed | TXT 文件数 | 标签实例数 |",
        "|---|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {row.dataset} | {row.shot} | {row.seed} | "
        f"{row.txt_files} | {row.label_instances} |"
        for row in rows
    )
    return "\n".join(lines)


def class_table_markdown(rows: Sequence[ClassSummary]) -> str:
    """Format per-class mean and std statistics as a Markdown table."""
    class_headers = " | ".join(
        f"Class {class_id + 1} instances" for class_id in range(NUM_CLASSES)
    )
    lines = [
        f"| Dataset | Shot | {class_headers} |",
        "|---|---:|" + "---:|" * NUM_CLASSES,
    ]
    for row in rows:
        cells = " | ".join(
            f"{mean_value:.2f}±{std_value:.2f}"
            for mean_value, std_value in row.class_mean_std
        )
        lines.append(f"| {row.dataset} | {row.shot} | {cells} |")
    return "\n".join(lines)


def write_csv(rows: Iterable[Statistics], output_path: Path) -> None:
    """Write detailed statistics to a UTF-8 CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "dataset",
                "shot",
                "seed",
                "txt_files",
                "label_instances",
                *(
                    f"class_{class_id + 1}_instances"
                    for class_id in range(NUM_CLASSES)
                ),
                "empty_files",
                "malformed_lines",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.dataset,
                    row.shot,
                    row.seed,
                    row.txt_files,
                    row.label_instances,
                    *row.class_instances,
                    row.empty_files,
                    row.malformed_lines,
                ]
            )


def write_class_summary_csv(
    rows: Iterable[ClassSummary], output_path: Path
) -> None:
    """Write per-class mean and population std across seeds to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    class_columns = []
    for class_id in range(NUM_CLASSES):
        class_columns.extend(
            [
                f"class_{class_id + 1}_mean",
                f"class_{class_id + 1}_std",
            ]
        )
    with output_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["dataset", "shot", *class_columns])
        for row in rows:
            values = [
                value
                for mean_std in row.class_mean_std
                for value in (f"{mean_std[0]:.6f}", f"{mean_std[1]:.6f}")
            ]
            writer.writerow([row.dataset, row.shot, *values])


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=project_root,
        help=f"项目根目录（默认: {project_root}）",
    )
    parser.add_argument(
        "--shots",
        nargs="+",
        default=DEFAULT_SHOTS,
        help="需要统计的 shot 目录名",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=DEFAULT_SEEDS,
        help="需要统计的 seed 目录名",
    )
    parser.add_argument("--csv", type=Path, help="可选的 CSV 输出路径")
    parser.add_argument(
        "--class-summary-csv",
        type=Path,
        help="可选的类别 mean±std 汇总 CSV 输出路径",
    )
    parser.add_argument("--markdown", type=Path, help="可选的 Markdown 输出路径")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows: list[Statistics] = []

    try:
        for dataset, relative_root in DATASETS:
            dataset_root = args.root / relative_root
            for shot in args.shots:
                for seed in args.seeds:
                    labels_dir = dataset_root / shot / seed / "labels"
                    rows.append(count_labels(dataset, shot, seed, labels_dir))
    except (OSError, UnicodeError) as error:
        print(f"错误: {error}", file=sys.stderr)
        return 1

    class_summaries = summarize_classes(rows)
    markdown = (
        "## 每个 seed 的总量\n\n"
        + seed_table_markdown(rows)
        + "\n\n## 每类实例数（五个 seed 的 mean±std）\n\n"
        + class_table_markdown(class_summaries)
    )
    print(markdown)

    if args.csv:
        write_csv(rows, args.csv)
        print(f"\nCSV 已保存: {args.csv}", file=sys.stderr)
    if args.class_summary_csv:
        write_class_summary_csv(class_summaries, args.class_summary_csv)
        print(f"类别汇总 CSV 已保存: {args.class_summary_csv}", file=sys.stderr)
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown + "\n", encoding="utf-8")
        print(f"Markdown 已保存: {args.markdown}", file=sys.stderr)

    empty_files = sum(row.empty_files for row in rows)
    malformed_lines = sum(row.malformed_lines for row in rows)
    print(
        f"检查结果: 空标签文件 {empty_files} 个，格式异常行 {malformed_lines} 行。",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

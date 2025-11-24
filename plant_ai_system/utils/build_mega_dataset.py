"""
Utility script to build a unified, deduplicated, and resized dataset
from the generated catalog.json produced by dataset_catalog_builder.py.

Features:
  * Consolidates images for every (plant, disease) class combination.
  * Normalizes class names (Plant_Disease) for folder structure.
  * Optionally applies a custom mapping file to override plant/disease names.
  * Deduplicates images using MD5 hashes to avoid overlaps across sources.
  * Resizes all images to the same resolution.
  * Splits each class into train/val/test subsets using 70/15/15 (configurable)
    while ensuring at least one sample per split if possible.
  * Exports dataset statistics and class index metadata.

Usage example:

    python build_mega_dataset.py \
        --catalog ../data/processed/mega_dataset/catalog.json \
        --output-dir ../../data/health_monitoring/mega_dataset \
        --image-size 256 \
        --workers 8

The script assumes you have already activated the Python virtual environment
for the project. Some operations (hashing and resizing large datasets) can take
time; adjust `--workers` to control concurrency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image


DEFAULT_IMAGE_SIZE = 256
DEFAULT_TRAIN_RATIO = 0.7
DEFAULT_VAL_RATIO = 0.15
DEFAULT_TEST_RATIO = 0.15


@dataclass
class CatalogEntry:
    dataset: str
    plant: str
    disease: str
    original_class: str
    file_path: Path

    @classmethod
    def from_dict(cls, data: Dict) -> "CatalogEntry":
        return cls(
            dataset=data["dataset"],
            plant=data["plant"],
            disease=data["disease"],
            original_class=data["original_class"],
            file_path=Path(data["file_path"]),
        )


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_catalog(path: Path) -> List[CatalogEntry]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    entries = [CatalogEntry.from_dict(item) for item in data]
    logging.info("Loaded %d catalog entries from %s", len(entries), path)
    return entries


def load_mapping(mapping_path: Optional[Path]) -> Dict[str, Dict[str, str]]:
    if not mapping_path:
        return {}
    with mapping_path.open("r", encoding="utf-8") as fh:
        raw_mapping = json.load(fh)

    # Normalize keys for case-insensitive lookup.
    mapping: Dict[str, Dict[str, str]] = {}

    def normalize_key(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")

    for key, info in raw_mapping.items():
        mapping[normalize_key(key)] = info
    logging.info("Loaded %d label mapping overrides from %s", len(mapping), mapping_path)
    return mapping


def canonicalize_text(text: str) -> str:
    """Return a title-case canonical representation for display."""
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    return " ".join(word.capitalize() for word in text.strip().split())


def slugify(text: str) -> str:
    """Convert text into safe slug format for folder/file names."""
    text = canonicalize_text(text)
    slug = re.sub(r"[^0-9A-Za-z]+", "_", text)
    slug = re.sub(r"_+", "_", slug)
    return slug.strip("_")


def apply_mapping(
    plant: str,
    disease: str,
    original_class: str,
    mapping: Dict[str, Dict[str, str]],
) -> Tuple[str, str]:
    """
    Apply optional mapping overrides. Keys can be stored using any of:
      - original_class exact match
      - dataset:original_class
      - plant:disease
    The first matching key will be used.
    """

    def normalize(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")

    candidate_keys = [
        normalize(original_class),
        normalize(f"{plant}:{disease}"),
        normalize(f"{plant}_{disease}"),
    ]

    for key in candidate_keys:
        if key in mapping:
            override = mapping[key]
            return (
                override.get("plant", plant),
                override.get("disease", disease),
            )

    return plant, disease


def collect_class_entries(
    entries: Sequence[CatalogEntry],
    mapping: Dict[str, Dict[str, str]],
) -> Dict[Tuple[str, str], List[Path]]:
    """Group file paths by canonical (plant, disease) keys."""
    grouped: Dict[Tuple[str, str], List[Path]] = defaultdict(list)
    missing_files = 0

    for entry in entries:
        plant_raw = canonicalize_text(entry.plant)
        disease_raw = canonicalize_text(entry.disease)

        plant_canon, disease_canon = apply_mapping(
            plant_raw, disease_raw, entry.original_class, mapping
        )
        plant_canon = canonicalize_text(plant_canon)
        disease_canon = canonicalize_text(disease_canon)

        key = (plant_canon, disease_canon)

        if not entry.file_path.exists():
            logging.warning("Missing file skipped: %s", entry.file_path)
            missing_files += 1
            continue

        grouped[key].append(entry.file_path)

    logging.info(
        "Collected %d unique plant/disease classes (skipped %d missing files)",
        len(grouped),
        missing_files,
    )
    return grouped


def compute_md5(path: Path) -> str:
    """Compute a file's MD5 hash."""
    hash_md5 = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def deduplicate_paths(paths: Sequence[Path]) -> List[Path]:
    """Remove duplicate images based on MD5 hash."""
    unique_paths: List[Path] = []
    seen_hashes: set[str] = set()
    for src in paths:
        file_hash = compute_md5(src)
        if file_hash in seen_hashes:
            continue
        seen_hashes.add(file_hash)
        unique_paths.append(src)
    return unique_paths


def split_dataset(
    items: List[Path],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[List[Path], List[Path], List[Path]]:
    """Split items into train/val/test ensuring full coverage."""
    if not items:
        return [], [], []

    total = len(items)
    train_count = max(1, int(round(total * train_ratio)))
    val_count = max(1, int(round(total * val_ratio)))

    if train_count + val_count > total:
        val_count = max(0, total - train_count)

    test_count = total - train_count - val_count
    if test_count == 0 and total > 1:
        if val_count > 1:
            val_count -= 1
            test_count = 1
        elif train_count > 1:
            train_count -= 1
            test_count = 1

    train = items[:train_count]
    val = items[train_count : train_count + val_count]
    test = items[train_count + val_count :]
    return train, val, test


def resize_and_save(
    src: Path,
    dest: Path,
    image_size: int,
) -> None:
    """Resize image and save to destination path."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Image.open(src) as img:
            img = img.convert("RGB")
            if image_size > 0:
                img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
            img.save(dest, format="JPEG", quality=95)
    except Exception as exc:
        logging.warning("Failed to process image %s: %s", src, exc)


def process_class(
    class_key: Tuple[str, str],
    paths: List[Path],
    output_dir: Path,
    image_size: int,
    ratios: Tuple[float, float, float],
    seed: int,
    max_workers: int,
) -> Dict[str, int]:
    """Deduplicate, split, and copy images for a single class."""
    plant, disease = class_key
    class_slug = f"{slugify(plant)}_{slugify(disease)}"

    # Deduplicate by hash.
    unique_paths = deduplicate_paths(paths)
    random.Random(seed).shuffle(unique_paths)

    train_ratio, val_ratio, test_ratio = ratios
    train_paths, val_paths, test_paths = split_dataset(
        unique_paths, train_ratio, val_ratio, test_ratio
    )

    split_mapping = {
        "train": train_paths,
        "val": val_paths,
        "test": test_paths,
    }

    stats = {split: len(items) for split, items in split_mapping.items()}

    # Prepare thread pool for copying.
    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for split, items in split_mapping.items():
            for idx, src in enumerate(items):
                filename = f"{class_slug}_{idx:06d}.jpg"
                dest = output_dir / split / class_slug / filename
                futures.append(executor.submit(resize_and_save, src, dest, image_size))

        for future in as_completed(futures):
            future.result()

    logging.info(
        "Processed class %-40s | train=%4d val=%4d test=%4d",
        class_slug,
        stats["train"],
        stats["val"],
        stats["test"],
    )

    return {
        "plant": plant,
        "disease": disease,
        "slug": class_slug,
        "train": stats["train"],
        "val": stats["val"],
        "test": stats["test"],
        "total": sum(stats.values()),
    }


def build_dataset(
    entries: Sequence[CatalogEntry],
    output_dir: Path,
    image_size: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    mapping: Dict[str, Dict[str, str]],
    max_workers: int,
) -> Dict:
    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean previous content (optional but recommended to avoid leftovers).
    splits = ["train", "val", "test"]
    for split in splits:
        split_path = output_dir / split
        if split_path.exists():
            shutil.rmtree(split_path)

    grouped = collect_class_entries(entries, mapping)

    class_stats: List[Dict] = []
    for class_key, paths in grouped.items():
        stats = process_class(
            class_key=class_key,
            paths=paths,
            output_dir=output_dir,
            image_size=image_size,
            ratios=(train_ratio, val_ratio, test_ratio),
            seed=seed,
            max_workers=max_workers,
        )
        class_stats.append(stats)

    summary = {
        "image_size": image_size,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "seed": seed,
        "total_classes": len(class_stats),
        "total_images": int(sum(stat["total"] for stat in class_stats)),
        "classes": class_stats,
    }

    return summary


def write_metadata(output_dir: Path, summary: Dict) -> None:
    classes_info = [
        {"slug": cls["slug"], "plant": cls["plant"], "disease": cls["disease"]}
        for cls in summary["classes"]
    ]

    classes_path = output_dir / "classes.json"
    stats_path = output_dir / "stats.json"

    with classes_path.open("w", encoding="utf-8") as fh:
        json.dump(classes_info, fh, indent=2, ensure_ascii=False)

    with stats_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    logging.info("Classes metadata saved to %s", classes_path)
    logging.info("Dataset statistics saved to %s", stats_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build unified plant dataset.")
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "data"
        / "processed"
        / "mega_dataset"
        / "catalog.json",
        help="Path to catalog.json generated by dataset_catalog_builder.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "data"
        / "health_monitoring"
        / "mega_dataset",
        help="Destination directory for the unified dataset",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=None,
        help="Optional JSON file with label mapping overrides",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help="Target square image size (pixels)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=DEFAULT_TRAIN_RATIO,
        help="Training split ratio",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=DEFAULT_VAL_RATIO,
        help="Validation split ratio",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=DEFAULT_TEST_RATIO,
        help="Test split ratio",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic splits",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker threads for resizing/copying",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def validate_ratios(train: float, val: float, test: float) -> None:
    total = train + val + test
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Split ratios must sum to 1.0 (received train={train}, "
            f"val={val}, test={test}, total={total})"
        )


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    catalog_entries = load_catalog(args.catalog)
    mapping = load_mapping(args.mapping)

    summary = build_dataset(
        entries=catalog_entries,
        output_dir=args.output_dir,
        image_size=args.image_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        mapping=mapping,
        max_workers=max(1, args.workers),
    )

    write_metadata(args.output_dir, summary)
    logging.info(
        "Unified dataset built successfully at %s | total images: %d | classes: %d",
        args.output_dir,
        summary["total_images"],
        summary["total_classes"],
    )


if __name__ == "__main__":
    main()



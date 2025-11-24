"""
Utility script to consolidate plant disease image datasets into a unified catalog.

The script scans predefined dataset folders under the project workspace, extracts
metadata (plant type, disease, split, original path) for each image, and stores
the consolidated information in JSON files under `plant_ai_system/data/processed`.

This provides a single source of truth for training or analysis pipelines without
physically copying the original images.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class DatasetEntry:
    dataset: str
    plant: str
    split: str
    disease: str
    original_class: str
    file_path: Path
    disease_db_key: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "dataset": self.dataset,
            "plant": self.plant,
            "split": self.split,
            "disease": self.disease,
            "original_class": self.original_class,
            "file_path": str(self.file_path),
            "disease_db_key": self.disease_db_key,
        }


def normalize_text(value: str) -> str:
    """Normalize a text string for fuzzy comparisons."""
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def load_disease_database_index(database_path: Path) -> Dict[str, str]:
    """Create a lookup index for disease database entries."""
    index: Dict[str, str] = {}
    if not database_path.exists():
        return index

    with database_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    for key, disease in data.get("diseases", {}).items():
        plant_type = disease.get("plant_type", "")
        disease_name = disease.get("name", key)
        variants = {
            normalize_text(key),
            normalize_text(disease_name),
            normalize_text(f"{plant_type} {disease_name}"),
            normalize_text(f"{plant_type} {key}"),
        }
        for variant in variants:
            if variant:
                index.setdefault(variant, key)

    return index


def normalize_disease_name(name: str) -> str:
    """Remove numeric suffixes and extra separators from disease folder names."""
    name = re.sub(r"\d+$", "", name)
    name = name.replace("_", " ").strip()
    name = re.sub(r"\s+", " ", name)
    return name


def iter_image_files(folder: Path) -> Iterable[Path]:
    for path in folder.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def parse_ccmt_augmented(root: Path, dataset_name: str) -> Iterable[DatasetEntry]:
    """Parse the CCMT augmented dataset which follows plant/train_set/disease_nameXXX."""
    for plant_dir in root.iterdir():
        if not plant_dir.is_dir():
            continue

        plant_name = plant_dir.name.strip()

        for split_dir in plant_dir.iterdir():
            if not split_dir.is_dir():
                continue

            split_raw = split_dir.name.lower()
            if "train" in split_raw:
                split = "train"
            elif "test" in split_raw or "valid" in split_raw:
                split = "test"
            elif "val" in split_raw:
                split = "val"
            else:
                split = split_dir.name

            for disease_dir in split_dir.iterdir():
                if not disease_dir.is_dir():
                    continue

                disease_name = normalize_disease_name(disease_dir.name)
                for image_path in iter_image_files(disease_dir):
                    yield DatasetEntry(
                        dataset=dataset_name,
                        plant=plant_name,
                        split=split,
                        disease=disease_name,
                        original_class=disease_dir.name,
                        file_path=image_path.resolve(),
                    )


def parse_ccmt_raw(root: Path, dataset_name: str) -> Iterable[DatasetEntry]:
    """Parse the CCMT raw dataset without explicit splits."""
    for plant_dir in root.iterdir():
        if not plant_dir.is_dir():
            continue

        plant_name = plant_dir.name.strip()

        for disease_dir in plant_dir.iterdir():
            if not disease_dir.is_dir():
                continue

            disease_name = normalize_disease_name(disease_dir.name)
            for image_path in iter_image_files(disease_dir):
                yield DatasetEntry(
                    dataset=dataset_name,
                    plant=plant_name,
                    split="unspecified",
                    disease=disease_name,
                    original_class=disease_dir.name,
                    file_path=image_path.resolve(),
                )


def parse_durian_dataset(root: Path, dataset_name: str) -> Iterable[DatasetEntry]:
    """Parse the durian dataset with train/val/test splits and disease folders."""
    plant_name = "Durian"

    for split_dir in root.iterdir():
        if not split_dir.is_dir():
            continue

        split = split_dir.name.lower()
        for disease_dir in split_dir.iterdir():
            if not disease_dir.is_dir():
                continue

            disease_name = normalize_disease_name(disease_dir.name.replace("Leaf_", ""))
            for image_path in iter_image_files(disease_dir):
                yield DatasetEntry(
                    dataset=dataset_name,
                    plant=plant_name,
                    split=split,
                    disease=disease_name,
                    original_class=disease_dir.name,
                    file_path=image_path.resolve(),
                )


def parse_simple_class_dir(root: Path, dataset_name: str) -> Iterable[DatasetEntry]:
    """Parse datasets where each subfolder encodes plant and disease in the name."""
    for class_dir in root.iterdir():
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name.strip()
        if "___" in class_name:
            plant_part, disease_part = class_name.split("___", 1)
        elif "__" in class_name:
            plant_part, disease_part = class_name.split("__", 1)
        else:
            parts = class_name.split("_", 1)
            plant_part = parts[0]
            disease_part = parts[1] if len(parts) > 1 else class_name

        plant_name = plant_part.replace("_", " ").strip()
        disease_name = disease_part.replace("_", " ").strip()

        for image_path in iter_image_files(class_dir):
            yield DatasetEntry(
                dataset=dataset_name,
                plant=plant_name,
                split="unspecified",
                disease=disease_name,
                original_class=class_dir.name,
                file_path=image_path.resolve(),
            )


DATASET_PARSERS = {
    "ccmt_augmented": parse_ccmt_augmented,
    "ccmt_raw": parse_ccmt_raw,
    "durian": parse_durian_dataset,
    "class_dir": parse_simple_class_dir,
}


def build_catalog(workspace_root: Path) -> Tuple[List[DatasetEntry], Dict]:
    """Collect dataset entries across sources and construct summary statistics."""
    sources = [
        {
            "name": "CCMT_Augmented",
            "path": workspace_root
            / "data"
            / "Crop_Pest_Disease_Detection"
            / "Dataset for Crop Pest and Disease Detection"
            / "CCMT Dataset-Augmented",
            "parser": "ccmt_augmented",
        },
        {
            "name": "CCMT_Raw",
            "path": workspace_root
            / "data"
            / "Crop_Pest_Disease_Detection"
            / "Dataset for Crop Pest and Disease Detection"
            / "Raw Data"
            / "CCMT Dataset",
            "parser": "ccmt_raw",
        },
        {
            "name": "PlantLeafDiseases",
            "path": workspace_root
            / "data"
            / "Data for Identification of Plant Leaf Diseases Using a 9-layer Deep Convolutional Neural Network"
            / "extracted_without_augmentation"
            / "Plant_leave_diseases_dataset_without_augmentation",
            "parser": "class_dir",
        },
        {
            "name": "DurianLeafDiseases",
            "path": workspace_root
            / "data"
            / "A Durian Leaf Image Dataset"
            / "A Durian Leaf Image Dataset of Common Diseases in Vietnam for Agricultural Diagnosis"
            / "Durian_Leaf_Diseases",
            "parser": "durian",
        },
    ]

    entries: List[DatasetEntry] = []
    missing_sources: List[str] = []

    for source in sources:
        dataset_path: Path = source["path"]
        parser_key = source["parser"]
        parser = DATASET_PARSERS.get(parser_key)

        if parser is None:
            print(f"[WARN] No parser registered for key '{parser_key}'. Skipping.")
            continue

        if not dataset_path.exists():
            missing_sources.append(source["name"])
            print(f"[WARN] Dataset path not found: {dataset_path}")
            continue

        print(f"[INFO] Parsing dataset '{source['name']}' from {dataset_path}")
        entries.extend(list(parser(dataset_path, source["name"])))

    summary: Dict = {
        "total_images": len(entries),
        "datasets": defaultdict(lambda: {"images": 0, "plants": defaultdict(lambda: {"images": 0, "diseases": defaultdict(int)})}),
        "missing_sources": missing_sources,
    }

    for entry in entries:
        dataset_summary = summary["datasets"][entry.dataset]
        dataset_summary["images"] += 1
        plant_summary = dataset_summary["plants"][entry.plant]
        plant_summary["images"] += 1
        plant_summary["diseases"][entry.disease] += 1

    # Convert default dicts to plain dicts for JSON serialization
    def serialize(obj):
        if isinstance(obj, defaultdict):
            return {k: serialize(v) for k, v in obj.items()}
        if isinstance(obj, dict):
            return {k: serialize(v) for k, v in obj.items()}
        return obj

    summary["datasets"] = serialize(summary["datasets"])

    return entries, summary


def enrich_with_database_keys(entries: List[DatasetEntry], database_index: Dict[str, str]) -> None:
    """Attach disease database keys where possible."""
    for entry in entries:
        candidate_keys = {
            normalize_text(entry.original_class),
            normalize_text(f"{entry.plant} {entry.disease}"),
            normalize_text(f"{entry.plant}_{entry.disease}"),
        }
        for candidate in candidate_keys:
            if candidate in database_index:
                entry.disease_db_key = database_index[candidate]
                break


def write_outputs(entries: List[DatasetEntry], summary: Dict, output_dir: Path) -> None:
    """Persist catalog and summary information to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    catalog_path = output_dir / "catalog.json"
    summary_path = output_dir / "summary.json"
    classes_path = output_dir / "class_index.json"

    with catalog_path.open("w", encoding="utf-8") as fh:
        json.dump([entry.to_dict() for entry in entries], fh, indent=2, ensure_ascii=False)

    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    # Build a mapping of unique plant/disease combinations encountered
    class_index: Dict[str, Dict[str, str]] = {}
    for entry in entries:
        key = f"{entry.plant}::{entry.disease}".lower()
        if key not in class_index:
            class_index[key] = {
                "plant": entry.plant,
                "disease": entry.disease,
                "disease_db_key": entry.disease_db_key,
            }

    with classes_path.open("w", encoding="utf-8") as fh:
        json.dump(class_index, fh, indent=2, ensure_ascii=False)

    print(f"[OK] Catalog written to {catalog_path}")
    print(f"[OK] Summary written to {summary_path}")
    print(f"[OK] Class index written to {classes_path}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build unified dataset catalog.")
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Path to the project workspace root (default: two levels up from this file).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "processed" / "mega_dataset",
        help="Directory to store the generated catalog files.",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "disease_database.json",
        help="Path to the disease database JSON file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    entries, summary = build_catalog(args.workspace_root)

    if not entries:
        print("[WARN] No dataset entries were collected. Nothing to write.")
        return

    database_index = load_disease_database_index(args.database)
    if database_index:
        enrich_with_database_keys(entries, database_index)
    else:
        print("[WARN] Disease database could not be loaded; skipping key enrichment.")

    write_outputs(entries, summary, args.output_dir)


if __name__ == "__main__":
    main()









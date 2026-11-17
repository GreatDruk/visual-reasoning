"""Build source-only ZIPs consumed by the pixel reasoning notebooks."""

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "scripts/pixel_reasoning.py",
    "src/__init__.py",
    "src/models/__init__.py",
    "src/models/count_distribution.py",
    "src/models/pair_choice.py",
    "src/models/slot_program.py",
    "src/models/slots.py",
    "src/pixel_regions.py",
    "src/primitive_regions.py",
    "src/region_shape_classifier.py",
    "src/scene_posterior.py",
    "src/text_aug.py",
    "src/twins.py",
    "src/utils.py",
)
ARCHIVES = ("visual-reasoning-pixels.zip", "visual-reasoning-silhouettes.zip", "silhouette-counts.zip")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artefacts")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ARCHIVES:
        path = args.output_dir / name
        with ZipFile(path, "w", ZIP_DEFLATED) as archive:
            for relative in SOURCES:
                archive.write(ROOT / relative, relative)
        print(path)


if __name__ == "__main__":
    main()

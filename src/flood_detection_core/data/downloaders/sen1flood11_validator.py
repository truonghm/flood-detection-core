from pathlib import Path

from rich import print

from flood_detection_core.data.constants import (
    CatalogSubdirs,
    FloodEventSubdirs,
    HandLabeledCatalogSubdir,
    HandLabeledCatalogSubdirs,
    HandLabeledFloodEventSubdir,
    HandLabeledFloodEventSubdirs,
    WeaklyLabeledCatalogSubdir,
    WeaklyLabeledCatalogSubdirs,
    WeaklyLabeledFloodEventSubdir,
    WeaklyLabeledFloodEventSubdirs,
)


def validate_sen1flood11(data_path: Path) -> dict[str, list[str]]:
    """
    Validate that the Sen1Flood11 data directory has the correct structure.

    Path provided must be to the version level (e.g. data/sen1flood11/v1.1)

    This directory is expected to contain these subdirectories and files:

    v1.1/
            catalog/
            data/
            Sen1Floods11_Metadata.geojson
    """

    error_report = {
        "general": [],
        "catalog": [],
        "flood_events": [],
        "catalog vs flood_events": [],
    }

    if not data_path.is_dir():
        error_report["general"].append(f"Path {data_path} is not a directory")
        return error_report

    expected_subdirs = ["catalog", "data"]
    expected_files = ["Sen1Floods11_Metadata.geojson"]

    for subdir in expected_subdirs:
        if not (data_path / subdir).is_dir():
            error_report["general"].append(
                f"Path {data_path} does not contain a {subdir} directory"
            )

    for file in expected_files:
        if not (data_path / file).is_file():
            error_report["general"].append(
                f"Path {data_path} does not contain a {file} file"
            )

    # check catalog subdir
    catalog_subdir = data_path / "catalog"
    expected_catalog_subdirs = CatalogSubdirs

    catalog_subdir_counts: dict[str, int] = {
        subdir: 0 for subdir in expected_catalog_subdirs
    }
    catalog_subdir_files: dict[str, set[str]] = {
        subdir: set() for subdir in expected_catalog_subdirs
    }

    for subdir in expected_catalog_subdirs:
        if not (catalog_subdir / subdir).is_dir():
            error_report["catalog"].append(
                f"Path {data_path} does not contain a {subdir} directory in the catalog subdirectory"
            )

        for elem in (catalog_subdir / subdir).iterdir():
            if elem.is_dir() and len(list(elem.iterdir())) == 1:
                catalog_file = list(elem.iterdir())[0]
                file_name = catalog_file.name
                if file_name.endswith(".json") and catalog_file.stem == elem.name:
                    catalog_subdir_files[subdir].add(elem.name)
                    catalog_subdir_counts[subdir] += 1

    if (
        catalog_subdir_counts[HandLabeledCatalogSubdir.LABEL]
        != catalog_subdir_counts[HandLabeledCatalogSubdir.SOURCE]
    ):
        error_report["catalog"].append(
            "The number of hand labeled labels in the catalog subdirectory does not match the number of hand labeled sources"
        )

    if (
        catalog_subdir_counts[WeaklyLabeledCatalogSubdir.LABEL]
        != catalog_subdir_counts[WeaklyLabeledCatalogSubdir.SOURCE]
    ):
        error_report["catalog"].append(
            "The number of weakly labeled labels in the catalog subdirectory does not match the number of weakly labeled sources"
        )

    catalog_site_tile_pairs: dict[str, set[tuple[str, str]]] = {
        subdir: set() for subdir in expected_catalog_subdirs
    }
    for dir, files in catalog_subdir_files.items():
        site_tile_pairs = set()

        for file in files:
            site_tile = "_".join(file.split("_")[:2])
            site_tile_pairs.add(site_tile)

        catalog_site_tile_pairs[dir] = site_tile_pairs

    def check_sets_equal(set_collection: list[set[tuple[str, str]]]) -> bool:
        if not set_collection:
            return True

        first_set = set_collection[0]
        return all(first_set == s for s in set_collection[1:])

    hand_labeled_catalog_site_tile_pairs = [
        pairs
        for dir, pairs in catalog_site_tile_pairs.items()
        if dir in HandLabeledCatalogSubdirs
    ]
    weakly_labeled_catalog_site_tile_pairs = [
        pairs
        for dir, pairs in catalog_site_tile_pairs.items()
        if dir in WeaklyLabeledCatalogSubdirs
    ]

    if not check_sets_equal(hand_labeled_catalog_site_tile_pairs):
        error_report["catalog vs flood_events"].append(
            "The site-tile pairs in the hand labeled catalog subdirectories do not match"
        )

    if not check_sets_equal(weakly_labeled_catalog_site_tile_pairs):
        error_report["catalog vs flood_events"].append(
            "The site-tile pairs in the weakly labeled catalog subdirectories do not match"
        )

    # check data/flood_events subdir
    flood_events_subdir = data_path / "data" / "flood_events"
    expected_flood_events_subdirs = FloodEventSubdirs
    flood_events_subdir_counts: dict[str, int] = {
        subdir: 0 for subdir in expected_flood_events_subdirs
    }
    flood_events_subdir_files: dict[str, set[str]] = {
        subdir: set() for subdir in expected_flood_events_subdirs
    }

    for subdir in expected_flood_events_subdirs:
        if not (flood_events_subdir / subdir).is_dir():
            error_report["flood_events"].append(
                f"Path {data_path} does not contain a {subdir} directory in the flood_events subdirectory"
            )

        for elem in (flood_events_subdir / subdir).iterdir():
            if elem.is_file() and elem.name.endswith(".tif"):
                flood_events_subdir_files[subdir].add(elem.name)
                flood_events_subdir_counts[subdir] += 1

    hand_labeled_counts = {
        count
        for subdir, count in flood_events_subdir_counts.items()
        if subdir in HandLabeledFloodEventSubdirs
    }
    weakly_labeled_counts = {
        count
        for subdir, count in flood_events_subdir_counts.items()
        if subdir in WeaklyLabeledFloodEventSubdirs
    }

    if len(hand_labeled_counts) != 1:
        error_report["flood_events"].append(
            "The number of hand labeled flood event images are not the same"
            + f"({hand_labeled_counts})"
        )

    if len(weakly_labeled_counts) != 1:
        error_report["flood_events"].append(
            "The number of weakly labeled flood event images are not the same"
            + f"({weakly_labeled_counts})"
        )

    flood_events_site_tile_pairs: dict[str, set[tuple[str, str]]] = {
        subdir: set() for subdir in expected_flood_events_subdirs
    }
    for dir, files in flood_events_subdir_files.items():
        site_tile_pairs = set()
        for file in files:
            site_tile = "_".join(file.split("_")[:2])
            site_tile_pairs.add(site_tile)
        flood_events_site_tile_pairs[dir] = site_tile_pairs

    hand_labeled_site_tile_pairs = [
        pairs
        for dir, pairs in flood_events_site_tile_pairs.items()
        if dir in HandLabeledFloodEventSubdirs
    ]
    weakly_labeled_site_tile_pairs = [
        pairs
        for dir, pairs in flood_events_site_tile_pairs.items()
        if dir in WeaklyLabeledFloodEventSubdirs
    ]

    if not check_sets_equal(hand_labeled_site_tile_pairs):
        error_report["catalog vs flood_events"].append(
            "The site-tile pairs in the hand labeled flood event subdirectories do not match"
        )

    if not check_sets_equal(weakly_labeled_site_tile_pairs):
        error_report["catalog vs flood_events"].append(
            "The site-tile pairs in the weakly labeled flood event subdirectories do not match"
        )

    # start comparing catalog and flood events
    # using s1 hand as representative
    if (
        flood_events_subdir_counts[HandLabeledFloodEventSubdir.S1_HAND]
        != catalog_subdir_counts[HandLabeledCatalogSubdir.LABEL]
    ):
        error_report["catalog vs flood_events"].append(
            "The number of hand labeled flood event images are not the same as the number of hand labeled labels in the catalog subdirectory"
        )

    if (
        flood_events_subdir_counts[WeaklyLabeledFloodEventSubdir.S1_WEAK]
        != catalog_subdir_counts[WeaklyLabeledCatalogSubdir.LABEL]
    ):
        error_report["catalog vs flood_events"].append(
            "The number of weakly labeled flood event images are not the same as the number of weakly labeled labels in the catalog subdirectory"
        )

    if hand_labeled_site_tile_pairs[0] != hand_labeled_catalog_site_tile_pairs[0]:
        error_report["catalog vs flood_events"].append(
            "The site-tile pairs in the hand labeled flood event subdirectories do not match the site-tile pairs in the hand labeled label catalog subdirectory"
        )

    if weakly_labeled_site_tile_pairs[0] != weakly_labeled_catalog_site_tile_pairs[0]:
        error_report["catalog vs flood_events"].append(
            "The site-tile pairs in the weakly labeled flood event subdirectories do not match the site-tile pairs in the weakly labeled label catalog subdirectory"
        )

    return error_report


if __name__ == "__main__":
    print(validate_sen1flood11(Path("./data/sen1flood11/v1.1")))

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from skimage.filters import threshold_otsu
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score

from flood_detection_core.config import DataConfig
from flood_detection_core.data.processing.split import get_flood_event_tile_pairs


def load_ground_truths(site: str, data_config: DataConfig, filter_threshold: float = 0.35) -> dict[str, np.ndarray]:
    tile_pairs = get_flood_event_tile_pairs(
        dataset_type="test",
        pre_flood_split_csv_path=data_config.splits.pre_flood_split,
        post_flood_split_csv_path=data_config.splits.post_flood_split,
    )

    tile_pairs_by_site = [pair for pair in tile_pairs if pair["site"] == site]

    ground_truths = {}
    for tile_pair in tile_pairs_by_site:
        tile_id = tile_pair["tile_id"]
        gt_path = tile_pair["ground_truth_path"]
        with rasterio.open(gt_path) as src:
            gt = src.read(1)
            # filter out images with >= filter_threshold of -1 -> too much missing data
            if np.sum(gt == -1) / gt.size >= filter_threshold:
                continue
            # gt = (gt > 0).astype(np.uint8)
            ground_truths[tile_id] = gt

    return ground_truths


def compute_change_maps_otsu(distance_maps: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    change_maps_otsu = {}
    tile_thresholds = {}
    for tile_id, d in distance_maps.items():
        v = d[np.isfinite(d)]
        if v.size == 0:
            th = 0.5
        else:
            try:
                th = float(threshold_otsu(v))
            except Exception:
                th = float(np.mean(v))
        tile_thresholds[tile_id] = th
        change_maps_otsu[tile_id] = (d > th).astype(np.uint8)

    return change_maps_otsu, tile_thresholds


def compute_change_maps(threshold: float, distance_maps: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    change_maps = {}
    for tile_id, d in distance_maps.items():
        change_maps[tile_id] = (d > threshold).astype(np.uint8)
    return change_maps


def flatten_binary(arr: np.ndarray) -> np.ndarray:
    """Convert to flat binary array."""
    return (arr.ravel() > 0).astype(int)


def compute_per_tile_metrics_otsu(
    change_maps_otsu: dict[str, np.ndarray],
    ground_truths: dict[str, np.ndarray],
    tile_thresholds: dict[str, float],
) -> dict[str, dict[str, float]]:
    per_tile_metrics = {}
    for tile_id, pred_map in change_maps_otsu.items():
        gt_bin = flatten_binary(ground_truths[tile_id])
        pred_bin = flatten_binary(pred_map)

        per_tile_metrics[tile_id] = {
            "iou": jaccard_score(gt_bin, pred_bin, zero_division=0),
            "f1": f1_score(gt_bin, pred_bin, zero_division=0),
            "precision": precision_score(gt_bin, pred_bin, zero_division=0),
            "recall": recall_score(gt_bin, pred_bin, zero_division=0),
            "threshold": tile_thresholds[tile_id],
        }

    return per_tile_metrics


def compute_per_tile_metrics(
    change_maps: dict[str, np.ndarray],
    ground_truths: dict[str, np.ndarray],
    threshold: float,
) -> dict[str, dict[str, float]]:
    per_tile_metrics = {}
    for tile_id, pred_map in change_maps.items():
        gt_bin = flatten_binary(ground_truths[tile_id])
        pred_bin = flatten_binary(pred_map)

        per_tile_metrics[tile_id] = {
            "iou": jaccard_score(gt_bin, pred_bin, zero_division=0),
            "f1": f1_score(gt_bin, pred_bin, zero_division=0),
            "precision": precision_score(gt_bin, pred_bin, zero_division=0),
            "recall": recall_score(gt_bin, pred_bin, zero_division=0),
            "threshold": threshold,
        }

    return per_tile_metrics


def compute_macro_avg_metrics(per_tile_metrics: dict[str, dict[str, float]]) -> dict[str, float]:
    macro_avg = (
        {
            metric: np.mean([tile_metrics[metric] for tile_metrics in per_tile_metrics.values()])
            for metric in ["iou", "f1", "precision", "recall"]
        }
        if per_tile_metrics
        else {"iou": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    )
    return macro_avg


def compute_micro_avg_metrics(
    change_maps: dict[str, np.ndarray],
    ground_truths: dict[str, np.ndarray],
) -> dict[str, float]:
    all_gt = np.concatenate([flatten_binary(ground_truths[tile_id]) for tile_id in change_maps.keys()])
    all_pred = np.concatenate([flatten_binary(change_maps[tile_id]) for tile_id in change_maps.keys()])

    micro_avg = {
        "iou": jaccard_score(all_gt, all_pred, zero_division=0),
        "f1": f1_score(all_gt, all_pred, zero_division=0),
        "precision": precision_score(all_gt, all_pred, zero_division=0),
        "recall": recall_score(all_gt, all_pred, zero_division=0),
    }

    return micro_avg


def plot_distance_maps_distribution(distance_maps: dict[str, np.ndarray]) -> None:
    n_tiles = len(distance_maps)
    n_cols = math.ceil(n_tiles / 3)

    fig, axes = plt.subplots(3, n_cols, figsize=(n_cols * 8, 12))

    # Handle case where there's only one column
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    # Flatten axes for easier iteration, but keep track of 2D structure
    axes_flat = axes.flatten()

    for i, (tile_id, distance_map) in enumerate(distance_maps.items()):
        finite_values = distance_map[np.isfinite(distance_map)]
        if finite_values.size > 0:
            axes_flat[i].hist(finite_values, bins=50, alpha=0.7, edgecolor="black")
            axes_flat[i].axvline(0.5, color="red", linestyle="--", label="Current threshold (0.5)")
            axes_flat[i].axvline(
                finite_values.mean(),
                color="green",
                linestyle="--",
                label=f"Mean ({finite_values.mean():.3f})",
            )
            try:
                otsu_thresh = threshold_otsu(finite_values)
                axes_flat[i].axvline(otsu_thresh, color="orange", linestyle="--", label=f"Otsu ({otsu_thresh:.3f})")
            except Exception:
                pass
            axes_flat[i].set_title(f"{tile_id}\nDistance Distribution")
            axes_flat[i].set_xlabel("Cosine Distance")
            axes_flat[i].set_ylabel("Frequency")
            axes_flat[i].legend()
        else:
            axes_flat[i].text(0.5, 0.5, "No finite values", ha="center", va="center", transform=axes_flat[i].transAxes)
            axes_flat[i].set_title(f"{tile_id}\nNo data")

    # Hide any unused subplots
    for j in range(n_tiles, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    plt.show()

    return fig, axes


def summarise_distance_maps(distance_maps: dict[str, np.ndarray]) -> pd.DataFrame:
    stats_data = []

    for tile_id, distance_map in distance_maps.items():
        finite_values = distance_map[np.isfinite(distance_map)]
        total_pixels = distance_map.size
        finite_pixels = finite_values.size

        if finite_values.size > 0:
            # Calculate Otsu threshold
            try:
                otsu_thresh = threshold_otsu(finite_values)
            except Exception:
                otsu_thresh = np.nan

            # Calculate percentiles
            percentiles = np.percentile(finite_values, [25, 50, 75, 90, 95, 99])

            # Values above threshold
            values_above_05 = np.sum(finite_values > 0.5)

            stats_data.append(
                {
                    "Tile ID": tile_id,
                    "Total Pixels": total_pixels,
                    "Finite Pixels": finite_pixels,
                    "Finite %": finite_pixels / total_pixels * 100,
                    "Mean": finite_values.mean(),
                    "Median": np.median(finite_values),
                    "Std Dev": finite_values.std(),
                    "Min": finite_values.min(),
                    "Max": finite_values.max(),
                    "Range": finite_values.max() - finite_values.min(),
                    "25th %ile": percentiles[0],
                    "50th %ile": percentiles[1],
                    "75th %ile": percentiles[2],
                    "90th %ile": percentiles[3],
                    "95th %ile": percentiles[4],
                    "99th %ile": percentiles[5],
                    "Otsu Threshold": otsu_thresh,
                    "Above 0.5": values_above_05,
                    "Above 0.5 %": values_above_05 / finite_pixels * 100,
                }
            )
        else:
            stats_data.append(
                {
                    "Tile ID": tile_id,
                    # "Total Pixels": total_pixels,
                    "Finite Pixels": 0,
                    "Finite %": 0,
                    "Mean": np.nan,
                    "Median": np.nan,
                    "Std Dev": np.nan,
                    "Min": np.nan,
                    "Max": np.nan,
                    "Range": np.nan,
                    "25th %ile": np.nan,
                    "50th %ile": np.nan,
                    "75th %ile": np.nan,
                    "90th %ile": np.nan,
                    "95th %ile": np.nan,
                    "99th %ile": np.nan,
                    "Otsu Threshold": np.nan,
                    "Above 0.5": 0,
                    "Above 0.5 %": 0,
                }
            )

    # Create DataFrame
    stats_df = pd.DataFrame(stats_data)

    # Format numeric columns for better display
    numeric_cols = [
        "Finite %",
        "Mean",
        "Median",
        "Std Dev",
        "Min",
        "Max",
        "Range",
        "25th %ile",
        "50th %ile",
        "75th %ile",
        "90th %ile",
        # "95th %ile",
        "99th %ile",
        "Otsu Threshold",
        "Above 0.5 %",
    ]

    for col in numeric_cols:
        if col in ["Finite %", "Above 0.5 %"]:
            stats_df[col] = stats_df[col].round(1)
        else:
            stats_df[col] = stats_df[col].round(4)

    return stats_df


def test_thresholds(
    thresholds: list[float],
    distance_maps: dict[str, np.ndarray],
    ground_truths: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    micro_avg_results = {}
    macro_avg_results = {}
    for threshold in thresholds:
        change_maps = compute_change_maps(threshold, distance_maps)
        per_tile_metrics = compute_per_tile_metrics(change_maps, ground_truths, threshold)

        # micro average
        all_pred = np.concatenate([flatten_binary(change_maps[tile_id]) for tile_id in change_maps.keys()])

        micro_avg_metrics = compute_micro_avg_metrics(change_maps, ground_truths)
        micro_avg_results[threshold] = {
            "micro-iou": micro_avg_metrics["iou"],
            "micro-f1": micro_avg_metrics["f1"],
            "micro-precision": micro_avg_metrics["precision"],
            "micro-recall": micro_avg_metrics["recall"],
        }

        macro_avg_metrics = compute_macro_avg_metrics(per_tile_metrics)
        macro_avg_results[threshold] = {
            "macro-iou": macro_avg_metrics["iou"],
            "macro-f1": macro_avg_metrics["f1"],
            "macro-precision": macro_avg_metrics["precision"],
            "macro-recall": macro_avg_metrics["recall"],
            "positive_pixels": np.sum(all_pred),
            "total_pixels": len(all_pred),
        }

    micro_metrics_df = pd.DataFrame(micro_avg_results).T
    macro_metrics_df = pd.DataFrame(macro_avg_results).T

    # merge micro and macro metrics on index
    merged_metrics_df = pd.merge(micro_metrics_df, macro_metrics_df, left_index=True, right_index=True).reset_index()
    merged_metrics_df.rename(columns={"index": "threshold"}, inplace=True)
    merged_metrics_df["pos_pct"] = 100 * merged_metrics_df["positive_pixels"] / merged_metrics_df["total_pixels"]
    merged_metrics_df.drop(columns=["positive_pixels", "total_pixels"], inplace=True)

    return merged_metrics_df

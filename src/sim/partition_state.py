from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from ..auto_explore import compute_density_map, connected_components, frontier_mask


@dataclass
class PartitionState:
    labels: np.ndarray
    density_map: np.ndarray
    centroids_xy: np.ndarray
    generators_xy: np.ndarray
    generator_colors: list
    partition_rgba: np.ndarray
    density_rgba: np.ndarray
    frontier_components: list


def _compute_partition_labels(shape, grid_resolution, generator_xy, blocked_mask=None):
    ny, nx = shape
    labels = -np.ones((ny, nx), dtype=int)
    if len(generator_xy) == 0:
        return labels

    generators = np.asarray(generator_xy, dtype=float)
    xs = (np.arange(nx, dtype=float) + 0.5) * float(grid_resolution)
    ys = (np.arange(ny, dtype=float) + 0.5) * float(grid_resolution)
    grid_x, grid_y = np.meshgrid(xs, ys)
    dist2 = (
        (grid_x[None, :, :] - generators[:, 0][:, None, None]) ** 2
        + (grid_y[None, :, :] - generators[:, 1][:, None, None]) ** 2
    )
    labels[:, :] = np.argmin(dist2, axis=0)
    if blocked_mask is not None:
        labels[np.asarray(blocked_mask, dtype=bool)] = -1
    return labels


def _compute_weighted_centroids(labels, density_map, grid_resolution):
    ny, nx = labels.shape
    if labels.size == 0 or np.max(labels) < 0:
        return np.zeros((0, 2), dtype=float)

    xs = (np.arange(nx, dtype=float) + 0.5) * float(grid_resolution)
    ys = (np.arange(ny, dtype=float) + 0.5) * float(grid_resolution)
    grid_x, grid_y = np.meshgrid(xs, ys)

    n_labels = int(np.max(labels)) + 1
    centroids = np.full((n_labels, 2), np.nan, dtype=float)
    for idx in range(n_labels):
        mask = labels == idx
        if not np.any(mask):
            continue
        weights = np.asarray(density_map[mask], dtype=float)
        if float(np.sum(weights)) <= 1e-9:
            weights = np.ones_like(weights, dtype=float)
        total = float(np.sum(weights))
        centroids[idx, 0] = float(np.sum(grid_x[mask] * weights) / total)
        centroids[idx, 1] = float(np.sum(grid_y[mask] * weights) / total)
    return centroids


def compute_partition_state(
    shared_known_grid,
    generators_xy,
    generator_colors,
    grid_resolution,
    unknown_value,
    free_value,
    occupied_value,
    frontier_weight,
    unknown_weight,
    free_weight,
    smoothing_passes,
    min_frontier_cells,
):
    blocked = np.asarray(shared_known_grid == occupied_value, dtype=bool)
    labels = _compute_partition_labels(shared_known_grid.shape, grid_resolution, generators_xy, blocked_mask=blocked)
    density_map = compute_density_map(
        shared_known_grid,
        unknown_value,
        free_value,
        occupied_value,
        frontier_weight=frontier_weight,
        unknown_weight=unknown_weight,
        free_weight=free_weight,
        smoothing_passes=smoothing_passes,
    )
    centroids_xy = _compute_weighted_centroids(labels, density_map, grid_resolution)

    rgba = np.zeros(shared_known_grid.shape + (4,), dtype=float)
    for idx, color in enumerate(generator_colors):
        mask = labels == idx
        if np.any(mask):
            rgba[mask, :3] = np.asarray(color[:3], dtype=float)
            rgba[mask, 3] = 0.12
    rgba[blocked, 3] = 0.0

    dens = np.clip(np.asarray(density_map, dtype=float), 0.0, None)
    dmax = float(np.max(dens)) if dens.size else 0.0
    if dmax > 1e-9:
        norm = dens / dmax
        density_rgba = plt.cm.YlOrRd(norm)
        density_rgba[..., 3] = 0.38 * np.clip(norm, 0.0, 1.0)
        density_rgba[blocked, 3] = 0.0
    else:
        density_rgba = np.zeros(shared_known_grid.shape + (4,), dtype=float)

    frontier_components = connected_components(
        frontier_mask(shared_known_grid, unknown_value, free_value),
        min_cells=min_frontier_cells,
    )
    return PartitionState(
        labels=labels,
        density_map=density_map,
        centroids_xy=centroids_xy,
        generators_xy=np.asarray(generators_xy, dtype=float),
        generator_colors=list(generator_colors),
        partition_rgba=rgba,
        density_rgba=density_rgba,
        frontier_components=frontier_components,
    )

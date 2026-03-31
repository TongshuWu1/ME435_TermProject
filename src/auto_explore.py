import math
from collections import deque

import numpy as np


def frontier_mask(known_grid, unknown_value, free_value):
    ny, nx = known_grid.shape
    mask = np.zeros((ny, nx), dtype=bool)
    for gy in range(1, ny - 1):
        for gx in range(1, nx - 1):
            if known_grid[gy, gx] != free_value:
                continue
            if (
                known_grid[gy - 1, gx] == unknown_value
                or known_grid[gy + 1, gx] == unknown_value
                or known_grid[gy, gx - 1] == unknown_value
                or known_grid[gy, gx + 1] == unknown_value
            ):
                mask[gy, gx] = True
    return mask


def connected_components(mask, min_cells=1):
    ny, nx = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components = []
    neigh = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
    for gy in range(ny):
        for gx in range(nx):
            if not mask[gy, gx] or visited[gy, gx]:
                continue
            q = deque([(gx, gy)])
            visited[gy, gx] = True
            cells = []
            while q:
                cx, cy = q.popleft()
                cells.append((cx, cy))
                for dx, dy in neigh:
                    nx2 = cx + dx
                    ny2 = cy + dy
                    if 0 <= nx2 < nx and 0 <= ny2 < ny and mask[ny2, nx2] and not visited[ny2, nx2]:
                        visited[ny2, nx2] = True
                        q.append((nx2, ny2))
            if len(cells) >= int(min_cells):
                components.append(cells)
    return components


def partition_generators_from_positions(positions_xy, start_pose, epsilon_m=0.35):
    positions = np.asarray(positions_xy, dtype=float)
    if len(positions) == 0:
        return positions
    gens = positions.copy()
    base_x, base_y, base_angle_deg = start_pose
    base_angle = math.radians(base_angle_deg)
    spread = np.max(np.linalg.norm(gens - np.mean(gens, axis=0), axis=1)) if len(gens) > 1 else 0.0
    if spread < 0.35 * max(float(epsilon_m), 1e-6):
        n = len(gens)
        center_angle = base_angle
        offsets = np.linspace(-0.65, 0.65, n) if n > 1 else [0.0]
        for i, off in enumerate(offsets):
            ang = center_angle + off
            gens[i, 0] = base_x + float(epsilon_m) * math.cos(ang)
            gens[i, 1] = base_y + float(epsilon_m) * math.sin(ang)
    return gens


def compute_partition_labels(shape, grid_to_world_fn, generator_xy, blocked_mask=None):
    ny, nx = shape
    labels = -np.ones((ny, nx), dtype=int)
    if len(generator_xy) == 0:
        return labels
    gens = np.asarray(generator_xy, dtype=float)
    for gy in range(ny):
        for gx in range(nx):
            if blocked_mask is not None and blocked_mask[gy, gx]:
                continue
            xw, yw = grid_to_world_fn((gx, gy))
            d2 = np.sum((gens - np.array([xw, yw])) ** 2, axis=1)
            labels[gy, gx] = int(np.argmin(d2))
    return labels


def compute_partition_centroids(labels, grid_to_world_fn):
    centroids = []
    for idx in range(int(labels.max()) + 1 if labels.size else 0):
        ys, xs = np.where(labels == idx)
        if len(xs) == 0:
            centroids.append((np.nan, np.nan))
            continue
        xsum = 0.0
        ysum = 0.0
        for gx, gy in zip(xs, ys):
            xw, yw = grid_to_world_fn((int(gx), int(gy)))
            xsum += xw
            ysum += yw
        centroids.append((xsum / len(xs), ysum / len(xs)))
    return np.asarray(centroids, dtype=float) if centroids else np.zeros((0, 2), dtype=float)


def choose_frontier_goal_for_robot(
    robot_index,
    frontier_components,
    partition_labels,
    grid_to_world_fn,
    robot_xy,
    fallback_global=True,
):
    best = None
    best_score = float('inf')
    best_meta = None
    for cells in frontier_components:
        assigned = 0
        xsum = 0.0
        ysum = 0.0
        for gx, gy in cells:
            xw, yw = grid_to_world_fn((gx, gy))
            xsum += xw
            ysum += yw
            if partition_labels[gy, gx] == robot_index:
                assigned += 1
        if assigned == 0 and not fallback_global:
            continue
        if assigned == 0:
            continue
        frac = assigned / max(len(cells), 1)
        centroid = (xsum / len(cells), ysum / len(cells))
        dist = math.hypot(centroid[0] - robot_xy[0], centroid[1] - robot_xy[1])
        score = dist - 1.25 * frac + 0.02 * len(cells)
        if score < best_score:
            best_score = score
            best = centroid
            best_meta = {'cells': cells, 'fraction': frac, 'size': len(cells)}
    if best is not None:
        return best, best_meta
    if not fallback_global:
        return None, None

    for cells in frontier_components:
        xsum = 0.0
        ysum = 0.0
        for gx, gy in cells:
            xw, yw = grid_to_world_fn((gx, gy))
            xsum += xw
            ysum += yw
        centroid = (xsum / len(cells), ysum / len(cells))
        dist = math.hypot(centroid[0] - robot_xy[0], centroid[1] - robot_xy[1])
        if dist < best_score:
            best_score = dist
            best = centroid
            best_meta = {'cells': cells, 'fraction': 0.0, 'size': len(cells)}
    return best, best_meta

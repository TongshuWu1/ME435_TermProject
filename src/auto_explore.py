import math
from collections import deque

import numpy as np

from config import GOAL_OBSTACLE_CLEARANCE_GAIN, GOAL_OBSTACLE_CLEARANCE_METERS, A_STAR_GOAL_CLEARANCE_CELLS


def frontier_mask(known_grid, unknown_value, free_value):
    mask = np.zeros_like(known_grid, dtype=bool)
    interior = known_grid[1:-1, 1:-1] == free_value
    adjacent_unknown = (
        (known_grid[:-2, 1:-1] == unknown_value)
        | (known_grid[2:, 1:-1] == unknown_value)
        | (known_grid[1:-1, :-2] == unknown_value)
        | (known_grid[1:-1, 2:] == unknown_value)
    )
    mask[1:-1, 1:-1] = interior & adjacent_unknown
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


def compute_density_map(
    known_grid,
    unknown_value,
    free_value,
    occupied_value,
    frontier_weight=2.0,
    unknown_weight=1.0,
    free_weight=0.05,
    smoothing_passes=0,
):
    frontier = frontier_mask(known_grid, unknown_value, free_value)
    density = np.zeros_like(known_grid, dtype=float)
    density[known_grid == free_value] = float(free_weight)
    density[known_grid == unknown_value] = float(unknown_weight)
    density[frontier] += float(frontier_weight)
    density[known_grid == occupied_value] = 0.0

    passes = max(0, int(smoothing_passes))
    if passes <= 0:
        return density

    traversable = known_grid != occupied_value
    for _ in range(passes):
        padded = np.pad(density, 1, mode='edge')
        padded_mask = np.pad(traversable.astype(float), 1, mode='constant', constant_values=0.0)
        accum = np.zeros_like(density, dtype=float)
        counts = np.zeros_like(density, dtype=float)
        for dy in range(3):
            for dx in range(3):
                accum += padded[dy:dy + density.shape[0], dx:dx + density.shape[1]]
                counts += padded_mask[dy:dy + density.shape[0], dx:dx + density.shape[1]]
        density = np.where(counts > 0.0, accum / np.maximum(counts, 1e-9), 0.0)
        density[~traversable] = 0.0
    return density


def compute_weighted_partition_centroids(labels, grid_to_world_fn, density_map):
    centroids = []
    for idx in range(int(labels.max()) + 1 if labels.size else 0):
        ys, xs = np.where(labels == idx)
        if len(xs) == 0:
            centroids.append((np.nan, np.nan))
            continue
        weights = density_map[ys, xs].astype(float)
        total_w = float(np.sum(weights))
        if total_w <= 1e-9:
            xsum = 0.0
            ysum = 0.0
            for gx, gy in zip(xs, ys):
                xw, yw = grid_to_world_fn((int(gx), int(gy)))
                xsum += xw
                ysum += yw
            centroids.append((xsum / len(xs), ysum / len(xs)))
            continue
        xsum = 0.0
        ysum = 0.0
        for gx, gy, w in zip(xs, ys, weights):
            xw, yw = grid_to_world_fn((int(gx), int(gy)))
            xsum += float(w) * xw
            ysum += float(w) * yw
        centroids.append((xsum / total_w, ysum / total_w))
    return np.asarray(centroids, dtype=float) if centroids else np.zeros((0, 2), dtype=float)


def _unknown_contacts(known_grid, gx, gy, unknown_value):
    ny, nx = known_grid.shape
    count = 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx2 = gx + dx
            ny2 = gy + dy
            if 0 <= nx2 < nx and 0 <= ny2 < ny and known_grid[ny2, nx2] == unknown_value:
                count += 1
    return count


def _path_length(path):
    if not path or len(path) < 2:
        return 0.0
    total = 0.0
    for p0, p1 in zip(path[:-1], path[1:]):
        total += math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    return total


def _weighted_xy(source):
    if isinstance(source, dict):
        xy = source.get('xy', None)
        if xy is None or len(xy) < 2:
            return None
        return float(xy[0]), float(xy[1]), float(source.get('weight', 1.0))
    if isinstance(source, (list, tuple)):
        if len(source) >= 3:
            return float(source[0]), float(source[1]), float(source[2])
        if len(source) >= 2:
            return float(source[0]), float(source[1]), 1.0
    return None


def _teammate_penalty(xy, teammate_positions, teammate_goal_positions, radius, gain):
    if radius <= 1e-9 or gain <= 0.0:
        return 0.0
    penalty = 0.0
    inv_r2 = 1.0 / (radius * radius)
    for source in list(teammate_positions) + list(teammate_goal_positions):
        parsed = _weighted_xy(source)
        if parsed is None:
            continue
        sx, sy, weight = parsed
        dx = xy[0] - sx
        dy = xy[1] - sy
        d2 = dx * dx + dy * dy
        if d2 < radius * radius:
            penalty += float(weight) * gain * math.exp(-1.6 * d2 * inv_r2)
    return penalty


def _path_history_penalty(xy, own_recent_path=None, teammate_path_histories=None, own_radius=0.0, own_gain=0.0, teammate_radius=0.0, teammate_gain=0.0):
    penalty = 0.0
    if own_recent_path and own_radius > 1e-9 and own_gain > 0.0:
        inv_r2 = 1.0 / (own_radius * own_radius)
        for px, py in own_recent_path:
            dx = xy[0] - float(px)
            dy = xy[1] - float(py)
            d2 = dx * dx + dy * dy
            if d2 < own_radius * own_radius:
                penalty += own_gain * math.exp(-1.4 * d2 * inv_r2)
    if teammate_path_histories and teammate_radius > 1e-9 and teammate_gain > 0.0:
        inv_r2 = 1.0 / (teammate_radius * teammate_radius)
        for history in teammate_path_histories:
            if isinstance(history, dict):
                points = history.get('points', [])
                weight = float(history.get('weight', 1.0))
            else:
                points = history
                weight = 1.0
            for px, py in points:
                dx = xy[0] - float(px)
                dy = xy[1] - float(py)
                d2 = dx * dx + dy * dy
                if d2 < teammate_radius * teammate_radius:
                    penalty += weight * teammate_gain * math.exp(-1.2 * d2 * inv_r2)
    return penalty


def _obstacle_clearance_penalty(xy, known_grid, occupied_value, world_to_grid_fn, grid_resolution, clearance_m, gain):
    if clearance_m <= 1e-9 or gain <= 0.0:
        return 0.0
    gx, gy = world_to_grid_fn(float(xy[0]), float(xy[1]))
    radius_cells = max(1, int(math.ceil(clearance_m / max(float(grid_resolution), 1e-9))))
    best = None
    for ny in range(max(0, gy - radius_cells), min(known_grid.shape[0], gy + radius_cells + 1)):
        for nx in range(max(0, gx - radius_cells), min(known_grid.shape[1], gx + radius_cells + 1)):
            if known_grid[ny, nx] != occupied_value:
                continue
            d = math.hypot((nx - gx) * grid_resolution, (ny - gy) * grid_resolution)
            if best is None or d < best:
                best = d
    if best is None or best >= clearance_m:
        return 0.0
    ratio = max(0.0, 1.0 - best / clearance_m)
    return gain * (ratio ** 2)
def choose_frontier_goal_for_robot(
    robot_index,
    frontier_components,
    partition_labels,
    grid_to_world_fn,
    robot_xy,
    known_grid,
    planner,
    occupied_value,
    unknown_value,
    fallback_global=True,
    teammate_positions=None,
    teammate_goal_positions=None,
    top_k_candidates=8,
    info_gain=1.15,
    partition_penalty=7.5,
    teammate_radius=4.0,
    teammate_penalty=2.75,
    progress_weight=0.55,
    min_goal_distance=0.0,
    density_map=None,
    centroid_xy=None,
    centroid_weight=0.0,
    density_value_weight=0.0,
    own_recent_path=None,
    teammate_path_histories=None,
    partition_penalty_scale=1.0,
    own_path_radius=0.0,
    own_path_gain=0.0,
    teammate_path_radius=0.0,
    teammate_path_gain=0.0,
):
    teammate_positions = [] if teammate_positions is None else list(teammate_positions)
    teammate_goal_positions = [] if teammate_goal_positions is None else list(teammate_goal_positions)

    quick_candidates = []
    fallback_candidates = []

    for comp_idx, cells in enumerate(frontier_components):
        comp_size = len(cells)
        comp_bonus = 0.2 * math.sqrt(max(comp_size, 1))
        for gx, gy in cells:
            xy = grid_to_world_fn((gx, gy))
            contacts = _unknown_contacts(known_grid, gx, gy, unknown_value)
            if contacts <= 0:
                continue
            in_partition = partition_labels[gy, gx] == robot_index
            info_score = float(contacts) + comp_bonus
            teammate_cost = _teammate_penalty(xy, teammate_positions, teammate_goal_positions, teammate_radius, teammate_penalty)
            dist = math.hypot(xy[0] - robot_xy[0], xy[1] - robot_xy[1])
            if dist < float(min_goal_distance):
                continue
            density_value = 0.0 if density_map is None else float(density_map[gy, gx])
            centroid_cost = 0.0
            if centroid_xy is not None and np.all(np.isfinite(np.asarray(centroid_xy, dtype=float))):
                centroid_cost = float(centroid_weight) * math.hypot(xy[0] - float(centroid_xy[0]), xy[1] - float(centroid_xy[1]))
            path_history_cost = _path_history_penalty(
                xy,
                own_recent_path=own_recent_path,
                teammate_path_histories=teammate_path_histories,
                own_radius=own_path_radius,
                own_gain=own_path_gain,
                teammate_radius=teammate_path_radius,
                teammate_gain=teammate_path_gain,
            )
            obstacle_clearance_cost = _obstacle_clearance_penalty(
                xy, known_grid, occupied_value, grid_to_world_fn.__self__.world_to_grid if hasattr(grid_to_world_fn, '__self__') and hasattr(grid_to_world_fn.__self__, 'world_to_grid') else planner.world_to_grid,
                planner.grid_resolution, GOAL_OBSTACLE_CLEARANCE_METERS, GOAL_OBSTACLE_CLEARANCE_GAIN,
            )
            quick_score = dist - info_gain * info_score - density_value_weight * density_value + teammate_cost + centroid_cost + path_history_cost + obstacle_clearance_cost
            meta = {
                'xy': xy,
                'cell': (gx, gy),
                'component_index': comp_idx,
                'component_size': comp_size,
                'unknown_contacts': contacts,
                'info_score': info_score,
                'in_partition': in_partition,
                'teammate_cost': teammate_cost,
                'euclid_dist': dist,
                'density_value': density_value,
                'centroid_cost': centroid_cost,
                'path_history_cost': path_history_cost,
                'obstacle_clearance_cost': obstacle_clearance_cost,
            }
            if in_partition:
                quick_candidates.append((quick_score, meta))
            elif fallback_global:
                fallback_candidates.append((quick_score + partition_penalty * float(partition_penalty_scale), meta))

    candidate_pool = quick_candidates
    used_fallback = False
    if not candidate_pool and fallback_global:
        candidate_pool = fallback_candidates
        used_fallback = True
    if not candidate_pool and float(min_goal_distance) > 1e-9:
        return choose_frontier_goal_for_robot(
            robot_index,
            frontier_components,
            partition_labels,
            grid_to_world_fn,
            robot_xy,
            known_grid,
            planner,
            occupied_value,
            unknown_value,
            fallback_global=fallback_global,
            teammate_positions=teammate_positions,
            teammate_goal_positions=teammate_goal_positions,
            top_k_candidates=top_k_candidates,
            info_gain=info_gain,
            partition_penalty=partition_penalty,
            teammate_radius=teammate_radius,
            teammate_penalty=teammate_penalty,
            progress_weight=progress_weight,
            min_goal_distance=0.0,
        )
    if not candidate_pool:
        return None, None

    candidate_pool.sort(key=lambda item: item[0])
    limit = max(1, int(top_k_candidates))

    best_goal = None
    best_meta = None
    best_score = float('inf')

    for _, meta in candidate_pool[:limit]:
        path = planner.astar(robot_xy, meta['xy'], known_grid, occupied_value, goal_clearance_cells=A_STAR_GOAL_CLEARANCE_CELLS)
        if path is None:
            continue
        path_len = _path_length(path)
        progress_cost = progress_weight * max(0.0, path_len - meta['euclid_dist'])
        final_score = path_len + progress_cost - info_gain * meta['info_score'] - density_value_weight * meta.get('density_value', 0.0) + meta['teammate_cost'] + meta.get('centroid_cost', 0.0) + meta.get('path_history_cost', 0.0) + meta.get('obstacle_clearance_cost', 0.0)
        if used_fallback and not meta['in_partition']:
            final_score += partition_penalty * float(partition_penalty_scale)
        if final_score < best_score:
            best_score = final_score
            best_goal = meta['xy']
            best_meta = dict(meta)
            best_meta['path_length'] = path_len
            best_meta['progress_cost'] = progress_cost
            best_meta['used_fallback'] = used_fallback and not meta['in_partition']

    return best_goal, best_meta

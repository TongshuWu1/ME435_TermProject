import math
from dataclasses import dataclass

import numpy as np


def _path_length(path):
    if not path or len(path) < 2:
        return 0.0
    total = 0.0
    for p0, p1 in zip(path[:-1], path[1:]):
        total += math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    return total


def _teammate_penalty(xy, teammate_positions, teammate_goal_positions, radius, gain):
    if radius <= 1e-9 or gain <= 0.0:
        return 0.0
    penalty = 0.0
    inv_r2 = 1.0 / (radius * radius)
    for source in list(teammate_positions) + list(teammate_goal_positions):
        dx = xy[0] - source[0]
        dy = xy[1] - source[1]
        d2 = dx * dx + dy * dy
        if d2 < radius * radius:
            penalty += gain * math.exp(-1.4 * d2 * inv_r2)
    return penalty


@dataclass
class WeightedCoverageController:
    fallback_global: bool
    top_k_candidates: int
    density_gain: float
    centroid_pull: float
    robot_distance_weight: float
    teammate_radius: float
    teammate_penalty: float
    progress_weight: float
    frontier_bonus: float
    min_goal_distance: float
    free_density_baseline: float

    def _candidate_cells(self, partition_labels, known_grid, free_value, robot_index, use_global=False):
        free_mask = np.asarray(known_grid == free_value, dtype=bool)
        if use_global:
            return free_mask
        return free_mask & (np.asarray(partition_labels) == int(robot_index))

    def _active_density_mask(self, density_map):
        return np.asarray(density_map, dtype=float) > (float(self.free_density_baseline) + 1e-6)

    def _nearest_centroid_cell(self, candidate_mask, centroid_xy, world_to_grid_fn):
        if centroid_xy is None:
            return None
        centroid = np.asarray(centroid_xy, dtype=float)
        if centroid.size != 2 or not np.all(np.isfinite(centroid)):
            return None
        cx, cy = world_to_grid_fn(float(centroid[0]), float(centroid[1]))
        ys, xs = np.where(candidate_mask)
        if len(xs) == 0:
            return None
        dist2 = (xs - cx) ** 2 + (ys - cy) ** 2
        idx = int(np.argmin(dist2))
        return int(xs[idx]), int(ys[idx])

    def _build_candidate_pool(
        self,
        *,
        robot_index,
        partition_labels,
        density_map,
        known_grid,
        free_value,
        centroid_xy,
        frontier_mask,
        grid_to_world_fn,
        world_to_grid_fn,
        robot_xy,
        teammate_positions,
        teammate_goal_positions,
    ):
        active_density = self._active_density_mask(density_map)
        local_mask = self._candidate_cells(partition_labels, known_grid, free_value, robot_index, use_global=False)
        local_active = local_mask & active_density
        if not np.any(local_active) and self.fallback_global:
            local_active = self._candidate_cells(partition_labels, known_grid, free_value, robot_index, use_global=True) & active_density
        if not np.any(local_active):
            return []

        ys, xs = np.where(local_active)
        candidates = []
        centroid = None if centroid_xy is None else np.asarray(centroid_xy, dtype=float)
        has_centroid = centroid is not None and centroid.size == 2 and np.all(np.isfinite(centroid))
        frontier_mask = np.asarray(frontier_mask, dtype=bool) if frontier_mask is not None else np.zeros_like(local_active, dtype=bool)

        for gx, gy in zip(xs, ys):
            xy = grid_to_world_fn((int(gx), int(gy)))
            dist_robot = math.hypot(xy[0] - robot_xy[0], xy[1] - robot_xy[1])
            if dist_robot < float(self.min_goal_distance):
                continue
            density_value = float(density_map[gy, gx] - self.free_density_baseline)
            centroid_dist = 0.0
            if has_centroid:
                centroid_dist = math.hypot(xy[0] - float(centroid[0]), xy[1] - float(centroid[1]))
            teammate_cost = _teammate_penalty(xy, teammate_positions, teammate_goal_positions, self.teammate_radius, self.teammate_penalty)
            frontier_gain = float(self.frontier_bonus) if frontier_mask[gy, gx] else 0.0
            quick_score = (
                self.density_gain * density_value
                + frontier_gain
                - self.centroid_pull * centroid_dist
                - self.robot_distance_weight * dist_robot
                - teammate_cost
            )
            candidates.append((quick_score, {
                'cell': (int(gx), int(gy)),
                'xy': xy,
                'dist_robot': dist_robot,
                'density_value': density_value,
                'centroid_dist': centroid_dist,
                'teammate_cost': teammate_cost,
                'frontier_gain': frontier_gain,
            }))

        anchor = self._nearest_centroid_cell(local_active, centroid_xy, world_to_grid_fn)
        if anchor is not None:
            gx, gy = anchor
            xy = grid_to_world_fn((gx, gy))
            dist_robot = math.hypot(xy[0] - robot_xy[0], xy[1] - robot_xy[1])
            if dist_robot >= float(self.min_goal_distance):
                centroid_dist = 0.0 if centroid is None else math.hypot(xy[0] - float(centroid[0]), xy[1] - float(centroid[1]))
                teammate_cost = _teammate_penalty(xy, teammate_positions, teammate_goal_positions, self.teammate_radius, self.teammate_penalty)
                density_value = float(density_map[gy, gx] - self.free_density_baseline)
                frontier_gain = float(self.frontier_bonus) if frontier_mask[gy, gx] else 0.0
                candidates.append((1e9, {
                    'cell': (int(gx), int(gy)),
                    'xy': xy,
                    'dist_robot': dist_robot,
                    'density_value': density_value,
                    'centroid_dist': centroid_dist,
                    'teammate_cost': teammate_cost,
                    'frontier_gain': frontier_gain,
                }))

        candidates.sort(key=lambda item: item[0], reverse=True)
        unique = []
        seen = set()
        for _, meta in candidates:
            cell = meta['cell']
            if cell in seen:
                continue
            seen.add(cell)
            unique.append(meta)
            if len(unique) >= max(3, int(self.top_k_candidates)):
                break
        return unique

    def choose_goal(
        self,
        *,
        robot_index,
        partition_labels,
        density_map,
        known_grid,
        free_value,
        occupied_value,
        planner,
        grid_to_world_fn,
        world_to_grid_fn,
        robot_xy,
        centroid_xy,
        teammate_positions,
        teammate_goal_positions,
        frontier_components,
    ):
        if not frontier_components:
            return None, None

        frontier_mask = np.zeros_like(partition_labels, dtype=bool)
        for comp in frontier_components:
            for gx, gy in comp:
                frontier_mask[gy, gx] = True

        candidates = self._build_candidate_pool(
            robot_index=robot_index,
            partition_labels=partition_labels,
            density_map=density_map,
            known_grid=known_grid,
            free_value=free_value,
            centroid_xy=centroid_xy,
            frontier_mask=frontier_mask,
            grid_to_world_fn=grid_to_world_fn,
            world_to_grid_fn=world_to_grid_fn,
            robot_xy=robot_xy,
            teammate_positions=teammate_positions,
            teammate_goal_positions=teammate_goal_positions,
        )
        if not candidates:
            return None, None

        best_goal = None
        best_meta = None
        best_score = float('inf')
        for meta in candidates:
            path = planner.astar(robot_xy, meta['xy'], known_grid, occupied_value)
            if path is None:
                continue
            path_len = _path_length(path)
            progress_cost = self.progress_weight * max(0.0, path_len - meta['dist_robot'])
            score = (
                path_len
                + progress_cost
                + self.centroid_pull * meta['centroid_dist']
                - self.density_gain * meta['density_value']
                - meta['frontier_gain']
                + meta['teammate_cost']
            )
            if score < best_score:
                best_score = score
                best_goal = meta['xy']
                best_meta = dict(meta)
                best_meta['path_length'] = path_len
                best_meta['progress_cost'] = progress_cost
                best_meta['score'] = score
        return best_goal, best_meta

import math
from collections import deque
from dataclasses import dataclass

import numpy as np


def _path_length(path):
    if not path or len(path) < 2:
        return 0.0
    total = 0.0
    for p0, p1 in zip(path[:-1], path[1:]):
        total += math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    return total


def _teammate_penalty(xy, teammate_positions, teammate_goal_positions, radius, gain, goal_gain_scale=1.35):
    if radius <= 1e-9 or gain <= 0.0:
        return 0.0
    penalty = 0.0
    inv_r2 = 1.0 / (radius * radius)
    for source in teammate_positions:
        dx = xy[0] - source[0]
        dy = xy[1] - source[1]
        d2 = dx * dx + dy * dy
        if d2 < radius * radius:
            penalty += gain * math.exp(-1.4 * d2 * inv_r2)
    for source in teammate_goal_positions:
        dx = xy[0] - source[0]
        dy = xy[1] - source[1]
        d2 = dx * dx + dy * dy
        if d2 < radius * radius:
            penalty += (goal_gain_scale * gain) * math.exp(-1.2 * d2 * inv_r2)
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
    frontier_contact_gain: float
    frontier_cluster_gain: float
    local_mass_gain: float
    frontier_proximity_gain: float
    nonmax_radius_cells: int
    global_fallback_penalty: float
    unknown_window_gain: float = 0.55
    frontier_window_gain: float = 0.45
    revisit_penalty_gain: float = 0.80
    revisit_window_radius: int = 2
    unknown_window_radius: int = 3

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

    def _window_sum(self, array, gx, gy, radius):
        y0 = max(0, gy - radius)
        y1 = min(array.shape[0], gy + radius + 1)
        x0 = max(0, gx - radius)
        x1 = min(array.shape[1], gx + radius + 1)
        return float(np.sum(array[y0:y1, x0:x1]))

    def _window_max(self, array, gx, gy, radius):
        y0 = max(0, gy - radius)
        y1 = min(array.shape[0], gy + radius + 1)
        x0 = max(0, gx - radius)
        x1 = min(array.shape[1], gx + radius + 1)
        return float(np.max(array[y0:y1, x0:x1]))

    def _unknown_contacts(self, unknown_mask, gx, gy):
        y0 = max(0, gy - 1)
        y1 = min(unknown_mask.shape[0], gy + 2)
        x0 = max(0, gx - 1)
        x1 = min(unknown_mask.shape[1], gx + 2)
        patch = unknown_mask[y0:y1, x0:x1]
        count = int(np.sum(patch))
        if 0 <= gy - y0 < patch.shape[0] and 0 <= gx - x0 < patch.shape[1] and patch[gy - y0, gx - x0]:
            count -= 1
        return count

    def _frontier_steps(self, traversable_mask, frontier_mask):
        inf = np.iinfo(np.int32).max
        steps = np.full(traversable_mask.shape, inf, dtype=np.int32)
        q = deque()
        ys, xs = np.where(frontier_mask & traversable_mask)
        for gx, gy in zip(xs, ys):
            steps[gy, gx] = 0
            q.append((int(gx), int(gy)))
        if not q:
            return steps
        neigh = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]
        ny, nx = traversable_mask.shape
        while q:
            cx, cy = q.popleft()
            base = int(steps[cy, cx]) + 1
            for dx, dy in neigh:
                nx2 = cx + dx
                ny2 = cy + dy
                if 0 <= nx2 < nx and 0 <= ny2 < ny and traversable_mask[ny2, nx2] and steps[ny2, nx2] > base:
                    steps[ny2, nx2] = base
                    q.append((nx2, ny2))
        return steps

    def _frontier_cluster_sizes(self, frontier_components, shape):
        cluster_sizes = np.zeros(shape, dtype=float)
        for comp in frontier_components:
            comp_size = float(len(comp))
            for gx, gy in comp:
                cluster_sizes[gy, gx] = comp_size
        return cluster_sizes

    def _visited_mask(self, known_grid_shape, visited_cells):
        mask = np.zeros(known_grid_shape, dtype=float)
        if not visited_cells:
            return mask
        for cell in visited_cells:
            try:
                gx, gy = int(cell[0]), int(cell[1])
            except Exception:
                continue
            if 0 <= gy < known_grid_shape[0] and 0 <= gx < known_grid_shape[1]:
                mask[gy, gx] = 1.0
        return mask

    def _build_candidate_pool(
        self,
        *,
        robot_index,
        partition_labels,
        density_map,
        known_grid,
        free_value,
        occupied_value,
        centroid_xy,
        frontier_mask,
        frontier_components,
        grid_to_world_fn,
        world_to_grid_fn,
        robot_xy,
        teammate_positions,
        teammate_goal_positions,
        visited_cells=None,
    ):
        known_grid = np.asarray(known_grid)
        density_map = np.asarray(density_map, dtype=float)
        active_density = self._active_density_mask(density_map)
        local_mask = self._candidate_cells(partition_labels, known_grid, free_value, robot_index, use_global=False)
        using_global = False
        candidate_mask = local_mask & (active_density | frontier_mask)
        if not np.any(candidate_mask):
            candidate_mask = local_mask
        if not np.any(candidate_mask) and self.fallback_global:
            using_global = True
            global_mask = self._candidate_cells(partition_labels, known_grid, free_value, robot_index, use_global=True)
            candidate_mask = global_mask & (active_density | frontier_mask)
            if not np.any(candidate_mask):
                candidate_mask = global_mask
        if not np.any(candidate_mask):
            return []

        unknown_mask = (known_grid != free_value) & (known_grid != occupied_value)
        free_mask = known_grid == free_value
        frontier_steps = self._frontier_steps(free_mask, frontier_mask)
        cluster_sizes = self._frontier_cluster_sizes(frontier_components, known_grid.shape)
        positive_density = np.maximum(density_map - self.free_density_baseline, 0.0)
        visited_mask = self._visited_mask(known_grid.shape, visited_cells)

        ys, xs = np.where(candidate_mask)
        candidates = []
        centroid = None if centroid_xy is None else np.asarray(centroid_xy, dtype=float)
        has_centroid = centroid is not None and centroid.size == 2 and np.all(np.isfinite(centroid))

        for gx, gy in zip(xs, ys):
            xy = grid_to_world_fn((int(gx), int(gy)))
            dist_robot = math.hypot(xy[0] - robot_xy[0], xy[1] - robot_xy[1])
            if dist_robot < float(self.min_goal_distance):
                continue

            density_value = max(0.0, float(density_map[gy, gx] - self.free_density_baseline))
            local_mass = max(0.0, self._window_sum(positive_density, int(gx), int(gy), radius=2))
            frontier_here = bool(frontier_mask[gy, gx])
            frontier_gain = float(self.frontier_bonus) if frontier_here else 0.0
            unknown_contacts = self._unknown_contacts(unknown_mask, int(gx), int(gy))
            unknown_window = self._window_sum(unknown_mask.astype(float), int(gx), int(gy), radius=max(1, int(self.unknown_window_radius)))
            frontier_window = self._window_sum(frontier_mask.astype(float), int(gx), int(gy), radius=max(1, int(self.unknown_window_radius)))
            revisit_mass = self._window_sum(visited_mask, int(gx), int(gy), radius=max(0, int(self.revisit_window_radius)))

            step_val = frontier_steps[gy, gx]
            if step_val >= np.iinfo(np.int32).max:
                frontier_proximity = 0.0
            else:
                frontier_proximity = 1.0 / (1.0 + float(step_val))

            nearby_cluster_size = self._window_max(cluster_sizes, int(gx), int(gy), radius=2)
            cluster_gain = math.log1p(max(0.0, nearby_cluster_size))

            centroid_dist = 0.0
            if has_centroid:
                centroid_dist = math.hypot(xy[0] - float(centroid[0]), xy[1] - float(centroid[1]))
            teammate_cost = _teammate_penalty(
                xy,
                teammate_positions,
                teammate_goal_positions,
                self.teammate_radius,
                self.teammate_penalty,
            )

            quick_score = (
                self.density_gain * density_value
                + self.local_mass_gain * local_mass
                + frontier_gain
                + self.frontier_contact_gain * float(unknown_contacts)
                + self.unknown_window_gain * unknown_window
                + self.frontier_window_gain * frontier_window
                + self.frontier_cluster_gain * cluster_gain
                + self.frontier_proximity_gain * frontier_proximity
                - 0.45 * self.centroid_pull * centroid_dist
                - self.robot_distance_weight * dist_robot
                - self.revisit_penalty_gain * revisit_mass
                - teammate_cost
            )
            if using_global and partition_labels[gy, gx] != int(robot_index):
                quick_score -= float(self.global_fallback_penalty)

            candidates.append((quick_score, {
                'cell': (int(gx), int(gy)),
                'xy': xy,
                'dist_robot': dist_robot,
                'density_value': density_value,
                'local_mass': local_mass,
                'centroid_dist': centroid_dist,
                'teammate_cost': teammate_cost,
                'frontier_gain': frontier_gain,
                'unknown_contacts': int(unknown_contacts),
                'unknown_window': float(unknown_window),
                'frontier_window': float(frontier_window),
                'revisit_mass': float(revisit_mass),
                'frontier_proximity': frontier_proximity,
                'cluster_gain': cluster_gain,
                'in_partition': bool(partition_labels[gy, gx] == int(robot_index)),
            }))

        if not candidates:
            return []

        # Add a centroid anchor only as a backup, never as a forced winner.
        anchor = self._nearest_centroid_cell(candidate_mask, centroid_xy, world_to_grid_fn)
        if anchor is not None:
            gx, gy = anchor
            if not any(meta['cell'] == (gx, gy) for _, meta in candidates):
                xy = grid_to_world_fn((gx, gy))
                dist_robot = math.hypot(xy[0] - robot_xy[0], xy[1] - robot_xy[1])
                if dist_robot >= float(self.min_goal_distance):
                    density_value = max(0.0, float(density_map[gy, gx] - self.free_density_baseline))
                    local_mass = max(0.0, self._window_sum(positive_density, int(gx), int(gy), radius=2))
                    unknown_contacts = self._unknown_contacts(unknown_mask, int(gx), int(gy))
                    unknown_window = self._window_sum(unknown_mask.astype(float), int(gx), int(gy), radius=max(1, int(self.unknown_window_radius)))
                    frontier_window = self._window_sum(frontier_mask.astype(float), int(gx), int(gy), radius=max(1, int(self.unknown_window_radius)))
                    revisit_mass = self._window_sum(visited_mask, int(gx), int(gy), radius=max(0, int(self.revisit_window_radius)))
                    step_val = frontier_steps[gy, gx]
                    frontier_proximity = 0.0 if step_val >= np.iinfo(np.int32).max else 1.0 / (1.0 + float(step_val))
                    cluster_gain = math.log1p(max(0.0, self._window_max(cluster_sizes, int(gx), int(gy), radius=2)))
                    centroid_dist = 0.0 if not has_centroid else math.hypot(xy[0] - float(centroid[0]), xy[1] - float(centroid[1]))
                    teammate_cost = _teammate_penalty(
                        xy,
                        teammate_positions,
                        teammate_goal_positions,
                        self.teammate_radius,
                        self.teammate_penalty,
                    )
                    anchor_boost = 0.25 * self.density_gain * density_value + 0.10 * self.local_mass_gain * local_mass
                    quick_score = (
                        anchor_boost
                        + 0.25 * self.frontier_contact_gain * float(unknown_contacts)
                        + 0.15 * self.unknown_window_gain * unknown_window
                        + 0.15 * self.frontier_window_gain * frontier_window
                        + 0.15 * self.frontier_proximity_gain * frontier_proximity
                        - 0.18 * self.centroid_pull * centroid_dist
                        - 0.35 * self.robot_distance_weight * dist_robot
                        - 0.50 * self.revisit_penalty_gain * revisit_mass
                        - 0.55 * teammate_cost
                    )
                    candidates.append((quick_score, {
                        'cell': (int(gx), int(gy)),
                        'xy': xy,
                        'dist_robot': dist_robot,
                        'density_value': density_value,
                        'local_mass': local_mass,
                        'centroid_dist': centroid_dist,
                        'teammate_cost': teammate_cost,
                        'frontier_gain': float(self.frontier_bonus) if frontier_mask[gy, gx] else 0.0,
                        'unknown_contacts': int(unknown_contacts),
                        'unknown_window': float(unknown_window),
                        'frontier_window': float(frontier_window),
                        'revisit_mass': float(revisit_mass),
                        'frontier_proximity': frontier_proximity,
                        'cluster_gain': cluster_gain,
                        'in_partition': bool(partition_labels[gy, gx] == int(robot_index)),
                        'anchor_candidate': True,
                    }))

        candidates.sort(key=lambda item: item[0], reverse=True)
        unique = []
        radius2 = max(0, int(self.nonmax_radius_cells)) ** 2
        for _, meta in candidates:
            cell = meta['cell']
            too_close = False
            if radius2 > 0:
                for chosen in unique:
                    dx = chosen['cell'][0] - cell[0]
                    dy = chosen['cell'][1] - cell[1]
                    if dx * dx + dy * dy <= radius2:
                        too_close = True
                        break
            if too_close:
                continue
            unique.append(meta)
            if len(unique) >= max(5, int(self.top_k_candidates)):
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
        unknown_value,
        planner,
        grid_to_world_fn,
        world_to_grid_fn,
        robot_xy,
        centroid_xy,
        teammate_positions,
        teammate_goal_positions,
        frontier_components,
        visited_cells=None,
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
            occupied_value=occupied_value,
            centroid_xy=centroid_xy,
            frontier_mask=frontier_mask,
            frontier_components=frontier_components,
            grid_to_world_fn=grid_to_world_fn,
            world_to_grid_fn=world_to_grid_fn,
            robot_xy=robot_xy,
            teammate_positions=teammate_positions,
            teammate_goal_positions=teammate_goal_positions,
            visited_cells=visited_cells,
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
                + 0.22 * self.centroid_pull * meta['centroid_dist']
                - self.density_gain * meta['density_value']
                - self.local_mass_gain * meta['local_mass']
                - meta['frontier_gain']
                - self.frontier_contact_gain * float(meta['unknown_contacts'])
                - self.unknown_window_gain * float(meta.get('unknown_window', 0.0))
                - self.frontier_window_gain * float(meta.get('frontier_window', 0.0))
                - self.frontier_cluster_gain * meta['cluster_gain']
                - self.frontier_proximity_gain * meta['frontier_proximity']
                + self.revisit_penalty_gain * float(meta.get('revisit_mass', 0.0))
                + meta['teammate_cost']
            )
            if not meta.get('in_partition', True):
                score += float(self.global_fallback_penalty)
            if score < best_score:
                best_score = score
                best_goal = meta['xy']
                best_meta = dict(meta)
                best_meta['path_length'] = path_len
                best_meta['progress_cost'] = progress_cost
                best_meta['score'] = score
                if meta['frontier_gain'] > 0.0 or meta['unknown_contacts'] >= 2:
                    best_meta['goal_flavor'] = 'coverage-frontier'
                elif meta.get('unknown_window', 0.0) >= 8.0 and meta.get('revisit_mass', 0.0) <= 2.0:
                    best_meta['goal_flavor'] = 'coverage-pocket'
                elif meta.get('anchor_candidate', False):
                    best_meta['goal_flavor'] = 'coverage-anchor'
                else:
                    best_meta['goal_flavor'] = 'coverage-fill'
        return best_goal, best_meta

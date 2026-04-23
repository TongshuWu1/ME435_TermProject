import math
from types import SimpleNamespace
from collections import deque

import numpy as np

from config import (
    ENCLOSED_FILL_MAX_CELLS,
    ENCLOSED_FILL_MIN_CELLS,
    ENCLOSED_FILL_SEAL_ONE_CELL_GAPS,
    FILL_ENCLOSED_OBSTACLE_INTERIORS,
    MAP_FREE_LOGODDS_DELTA,
    MAP_MIN_UPDATE_WEIGHT,
    MAP_OCCUPIED_LOGODDS_DELTA,
    MAP_POSE_TRACE_SCALE,
    MAP_RANGE_WEIGHT_GAIN,
    MAP_SOLIDIFY_CONFIDENCE,
    MAP_SOLIDIFY_LOGODDS,
    MAP_WEIGHT_HEADING_GAIN,
    MAP_POSE_UNCERTAINTY_HEADING_GAIN,
    MAP_POSE_UNCERTAINTY_RANGE_GAIN,
    MAP_POSE_UNCERTAINTY_FREE_DECAY,
)


def reveal_start_area(grid, x, y, radius_m, world_to_grid, offsets_for_margin, nx, ny, unknown_value, free_value):
    gx, gy = world_to_grid(x, y)
    for dx, dy in offsets_for_margin(radius_m):
        nx_cell = gx + dx
        ny_cell = gy + dy
        if 0 <= nx_cell < nx and 0 <= ny_cell < ny and grid[ny_cell, nx_cell] == unknown_value:
            grid[ny_cell, nx_cell] = free_value


def initialize_belief_grids(grid, occupied_value, *, occupied_logodds=6.0, occupied_confidence=6.0):
    """Create log-odds and confidence layers matching an occupancy grid."""
    logodds = np.zeros_like(grid, dtype=np.float32)
    confidence = np.zeros_like(grid, dtype=np.float32)
    occupied_mask = grid == occupied_value
    logodds[occupied_mask] = float(occupied_logodds)
    confidence[occupied_mask] = float(occupied_confidence)
    return logodds, confidence


def initialize_pose_uncertainty_grid(grid, occupied_value, *, occupied_uncertainty=0.05):
    pose_unc = np.full_like(grid, np.inf, dtype=np.float32)
    pose_unc[grid == occupied_value] = float(occupied_uncertainty)
    return pose_unc


def reveal_start_area_with_belief(
    grid,
    logodds,
    confidence,
    pose_uncertainty,
    x,
    y,
    radius_m,
    world_to_grid,
    offsets_for_margin,
    nx,
    ny,
    unknown_value,
    free_value,
):
    reveal_start_area(grid, x, y, radius_m, world_to_grid, offsets_for_margin, nx, ny, unknown_value, free_value)
    gx, gy = world_to_grid(x, y)
    for dx, dy in offsets_for_margin(radius_m):
        nx_cell = gx + dx
        ny_cell = gy + dy
        if 0 <= nx_cell < nx and 0 <= ny_cell < ny:
            logodds[ny_cell, nx_cell] = min(float(logodds[ny_cell, nx_cell]), MAP_FREE_LOGODDS_DELTA)
            confidence[ny_cell, nx_cell] = max(float(confidence[ny_cell, nx_cell]), 0.5)
            pose_uncertainty[ny_cell, nx_cell] = min(float(pose_uncertainty[ny_cell, nx_cell]), 0.10)


def scan_buffer_entry_from_scan(robot, scan, pose_estimate, pose_cov, timestamp):
    """Store a compact scan record that can later be replayed with a corrected pose."""
    entry = {
        'pose': tuple(float(v) for v in pose_estimate),
        'pose_trace': float(np.trace(pose_cov[:2, :2])) if pose_cov is not None else 0.0,
        'heading_var_deg2': float(pose_cov[2, 2]) if pose_cov is not None else 0.0,
        'timestamp': float(timestamp),
        'rays': [],
    }
    robot_angle = float(robot.angle)
    for ray in scan:
        rel = ((float(ray['angle_deg']) - robot_angle + 180.0) % 360.0) - 180.0
        dist = float(ray['distance'])
        entry['rays'].append({
            'rel_angle_deg': rel,
            'distance': dist,
            'is_obstacle_hit': bool(ray.get('is_obstacle_hit', dist < 0.0)),
        })
    return entry


def scan_from_buffer_entry(entry):
    x, y, theta = (float(v) for v in entry['pose'])
    rays = []
    for ray in entry.get('rays', []):
        rel = float(ray.get('rel_angle_deg', 0.0))
        ang = (theta + rel) % 360.0
        dist = float(ray['distance'])
        ang_rad = math.radians(ang)
        rays.append({
            'angle_deg': ang,
            'rel_angle_deg': rel,
            'distance': dist,
            'hit_x': x + dist * math.cos(ang_rad),
            'hit_y': y + dist * math.sin(ang_rad),
            'is_obstacle_hit': bool(ray.get('is_obstacle_hit', dist < 0.0)),
        })
    return rays


def shifted_scan_buffer_entry(entry, delta_pose):
    dx, dy, dtheta = (float(v) for v in delta_pose)
    px, py, ptheta = (float(v) for v in entry['pose'])
    shifted = dict(entry)
    shifted['pose'] = (px + dx, py + dy, (ptheta + dtheta) % 360.0)
    shifted['rays'] = [dict(ray) for ray in entry.get('rays', [])]
    return shifted


def _projected_hit_uncertainty_m(pose_cov, distance):
    if pose_cov is None:
        return max(0.02, MAP_POSE_UNCERTAINTY_RANGE_GAIN * float(distance))
    cov = np.asarray(pose_cov, dtype=float)
    sigma_xy = math.sqrt(max(0.0, 0.5 * (float(cov[0, 0]) + float(cov[1, 1]))))
    sigma_theta_deg = math.sqrt(max(0.0, float(cov[2, 2])))
    sigma_theta = math.radians(sigma_theta_deg)
    heading_sigma = float(distance) * sigma_theta * MAP_POSE_UNCERTAINTY_HEADING_GAIN
    sensor_sigma = MAP_POSE_UNCERTAINTY_RANGE_GAIN * float(distance)
    return max(0.02, sigma_xy + heading_sigma + sensor_sigma)


def _observation_weight(pose_cov, distance, view_distance):
    pose_trace = float(np.trace(np.asarray(pose_cov, dtype=float)[:2, :2])) if pose_cov is not None else 0.0
    heading_var = float(np.asarray(pose_cov, dtype=float)[2, 2]) if pose_cov is not None else 0.0
    pose_weight = 1.0 / (1.0 + MAP_POSE_TRACE_SCALE * max(0.0, pose_trace))
    range_ratio = 0.0 if view_distance <= 1e-9 else min(1.0, max(0.0, float(distance) / float(view_distance)))
    range_weight = 1.0 / (1.0 + MAP_RANGE_WEIGHT_GAIN * range_ratio)
    heading_term = math.radians(max(0.0, heading_var) ** 0.5) * float(distance)
    heading_weight = 1.0 / (1.0 + MAP_WEIGHT_HEADING_GAIN * heading_term)
    return max(float(MAP_MIN_UPDATE_WEIGHT), float(pose_weight * range_weight * heading_weight))


def _discrete_state_from_belief(logodds, confidence, unknown_value, free_value, occupied_value):
    if confidence < 1e-6 and abs(logodds) < 1e-6:
        return unknown_value
    if logodds > 0.08:
        return occupied_value
    if logodds < -0.08:
        return free_value
    return free_value if confidence > 0.2 else unknown_value


def _weighted_cell_update(
    grid,
    logodds_grid,
    confidence_grid,
    pose_uncertainty_grid,
    gx,
    gy,
    delta,
    weight,
    obs_pose_uncertainty,
    unknown_value,
    free_value,
    occupied_value,
):
    prev = int(grid[gy, gx])
    logodds_grid[gy, gx] = float(logodds_grid[gy, gx]) + float(delta) * float(weight)
    confidence_grid[gy, gx] = float(confidence_grid[gy, gx]) + float(weight)
    if obs_pose_uncertainty is not None:
        obs_pose_uncertainty = float(obs_pose_uncertainty)
        current = float(pose_uncertainty_grid[gy, gx])
        if not np.isfinite(current):
            pose_uncertainty_grid[gy, gx] = obs_pose_uncertainty
        else:
            pose_uncertainty_grid[gy, gx] = min(current, obs_pose_uncertainty)
    grid[gy, gx] = _discrete_state_from_belief(float(logodds_grid[gy, gx]), float(confidence_grid[gy, gx]), unknown_value, free_value, occupied_value)
    return int(grid[gy, gx]) != prev


def _sync_solidified_cells(grid, logodds_grid, confidence_grid, pose_uncertainty_grid, occupied_value):
    solid_mask = (grid == occupied_value) & (confidence_grid < MAP_SOLIDIFY_CONFIDENCE)
    if not np.any(solid_mask):
        return 0
    logodds_grid[solid_mask] = np.maximum(logodds_grid[solid_mask], MAP_SOLIDIFY_LOGODDS)
    confidence_grid[solid_mask] = np.maximum(confidence_grid[solid_mask], MAP_SOLIDIFY_CONFIDENCE)
    pose_uncertainty_grid[solid_mask] = np.minimum(pose_uncertainty_grid[solid_mask], 0.12)
    return int(np.count_nonzero(solid_mask))


def _seal_small_gaps(occupied_mask, radius_cells=1):
    if radius_cells <= 0:
        return occupied_mask.copy()

    sealed = occupied_mask.copy()
    ny, nx = occupied_mask.shape
    for gy in range(radius_cells, ny - radius_cells):
        for gx in range(radius_cells, nx - radius_cells):
            if occupied_mask[gy, gx]:
                continue
            for r in range(1, radius_cells + 1):
                if occupied_mask[gy, gx - r] and occupied_mask[gy, gx + r]:
                    sealed[gy, gx] = True
                    break
                if occupied_mask[gy - r, gx] and occupied_mask[gy + r, gx]:
                    sealed[gy, gx] = True
                    break
                if occupied_mask[gy - r, gx - r] and occupied_mask[gy + r, gx + r]:
                    sealed[gy, gx] = True
                    break
                if occupied_mask[gy - r, gx + r] and occupied_mask[gy + r, gx - r]:
                    sealed[gy, gx] = True
                    break
    return sealed


def _component_labels(mask, connectivity=8):
    ny, nx = mask.shape
    labels = -np.ones((ny, nx), dtype=np.int32)
    components = []
    nbrs = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)) if connectivity == 8 else ((1, 0), (-1, 0), (0, 1), (0, -1))
    label = 0
    for gy in range(ny):
        for gx in range(nx):
            if not mask[gy, gx] or labels[gy, gx] != -1:
                continue
            q = deque([(gx, gy)])
            labels[gy, gx] = label
            pts = []
            min_x = max_x = gx
            min_y = max_y = gy
            while q:
                cx, cy = q.popleft()
                pts.append((cx, cy))
                min_x = min(min_x, cx)
                max_x = max(max_x, cx)
                min_y = min(min_y, cy)
                max_y = max(max_y, cy)
                for dx, dy in nbrs:
                    nx2 = cx + dx
                    ny2 = cy + dy
                    if 0 <= nx2 < nx and 0 <= ny2 < ny and mask[ny2, nx2] and labels[ny2, nx2] == -1:
                        labels[ny2, nx2] = label
                        q.append((nx2, ny2))
            components.append({'label': label, 'points': pts, 'min_x': min_x, 'max_x': max_x, 'min_y': min_y, 'max_y': max_y, 'size': len(pts)})
            label += 1
    return labels, components


def _solidify_rectangular_components(grid, occupied_mask, unknown_value, occupied_value, max_cells):
    labels, components = _component_labels(occupied_mask, connectivity=8)
    ny, nx = grid.shape
    filled = 0
    for comp in components:
        min_x, max_x = comp['min_x'], comp['max_x']
        min_y, max_y = comp['min_y'], comp['max_y']
        width = max_x - min_x + 1
        height = max_y - min_y + 1
        bbox_area = width * height
        if bbox_area <= 0 or bbox_area > max_cells or width < 3 or height < 3:
            continue
        if min_x <= 1 or min_y <= 1 or max_x >= nx - 2 or max_y >= ny - 2:
            continue
        fill_ratio = comp['size'] / float(bbox_area)
        if fill_ratio < 0.22:
            continue
        comp_mask = labels[min_y:max_y + 1, min_x:max_x + 1] == comp['label']
        row_counts = comp_mask.sum(axis=1)
        col_counts = comp_mask.sum(axis=0)
        row_support = np.count_nonzero(row_counts >= max(1, int(math.ceil(0.45 * width))))
        col_support = np.count_nonzero(col_counts >= max(1, int(math.ceil(0.45 * height))))
        side_hits = sum((np.any(comp_mask[0, :]), np.any(comp_mask[-1, :]), np.any(comp_mask[:, 0]), np.any(comp_mask[:, -1])))
        strong_box = fill_ratio >= 0.50 and side_hits >= 3
        moderate_box = fill_ratio >= 0.32 and row_support >= 2 and col_support >= 2 and side_hits >= 2
        chunky_box = fill_ratio >= 0.62
        if not (strong_box or moderate_box or chunky_box):
            continue
        subgrid = grid[min_y:max_y + 1, min_x:max_x + 1]
        fill_mask = subgrid != occupied_value
        if not np.any(fill_mask):
            continue
        free_cells = np.count_nonzero(subgrid == 1)
        if free_cells > 0.45 * bbox_area and fill_ratio < 0.55:
            continue
        subgrid[fill_mask] = occupied_value
        filled += int(np.count_nonzero(fill_mask))
    return filled


def _fill_enclosed_unknown_pockets(grid, unknown_value, occupied_value, max_cells, min_cells, seal_one_cell_gaps):
    occupied_mask = grid == occupied_value
    barrier = _seal_small_gaps(occupied_mask, radius_cells=1 if seal_one_cell_gaps else 0)
    traversable = ~barrier
    ny, nx = grid.shape
    outside = np.zeros((ny, nx), dtype=bool)
    q = deque()

    def enqueue(x, y):
        if traversable[y, x] and not outside[y, x]:
            outside[y, x] = True
            q.append((x, y))

    for gx in range(nx):
        enqueue(gx, 0)
        enqueue(gx, ny - 1)
    for gy in range(ny):
        enqueue(0, gy)
        enqueue(nx - 1, gy)

    nbrs = ((1, 0), (-1, 0), (0, 1), (0, -1))
    while q:
        gx, gy = q.popleft()
        for dx, dy in nbrs:
            nx2 = gx + dx
            ny2 = gy + dy
            if 0 <= nx2 < nx and 0 <= ny2 < ny and traversable[ny2, nx2] and not outside[ny2, nx2]:
                outside[ny2, nx2] = True
                q.append((nx2, ny2))

    enclosed = traversable & (~outside)
    if not np.any(enclosed):
        return 0

    visited = np.zeros((ny, nx), dtype=bool)
    filled = 0
    for gy in range(1, ny - 1):
        for gx in range(1, nx - 1):
            if not enclosed[gy, gx] or visited[gy, gx]:
                continue
            comp = []
            q = deque([(gx, gy)])
            visited[gy, gx] = True
            unknown_count = 0
            touches_boundary = False
            while q:
                cx, cy = q.popleft()
                comp.append((cx, cy))
                if grid[cy, cx] == unknown_value:
                    unknown_count += 1
                if cx <= 1 or cy <= 1 or cx >= nx - 2 or cy >= ny - 2:
                    touches_boundary = True
                for dx, dy in nbrs:
                    nx2 = cx + dx
                    ny2 = cy + dy
                    if 0 <= nx2 < nx and 0 <= ny2 < ny and enclosed[ny2, nx2] and not visited[ny2, nx2]:
                        visited[ny2, nx2] = True
                        q.append((nx2, ny2))
            comp_size = len(comp)
            if touches_boundary or comp_size < min_cells or comp_size > max_cells or unknown_count == 0:
                continue
            for cx, cy in comp:
                if grid[cy, cx] != occupied_value:
                    grid[cy, cx] = occupied_value
                    filled += 1
    return filled


def solidify_obstacle_interiors(grid, unknown_value, occupied_value, max_cells=ENCLOSED_FILL_MAX_CELLS, min_cells=ENCLOSED_FILL_MIN_CELLS, seal_one_cell_gaps=ENCLOSED_FILL_SEAL_ONE_CELL_GAPS):
    if not FILL_ENCLOSED_OBSTACLE_INTERIORS:
        return 0
    filled = 0
    occupied_mask = grid == occupied_value
    if seal_one_cell_gaps:
        occupied_mask = _seal_small_gaps(occupied_mask, radius_cells=1)
    filled += _solidify_rectangular_components(grid, occupied_mask, unknown_value, occupied_value, max_cells)
    filled += _fill_enclosed_unknown_pockets(grid, unknown_value, occupied_value, max_cells, min_cells, seal_one_cell_gaps)
    return filled


def apply_scan_to_grid(
    grid,
    logodds_grid,
    confidence_grid,
    pose_uncertainty_grid,
    robot,
    scan,
    world_to_grid,
    stamp_obstacle_hit,
    grid_resolution,
    view_distance,
    unknown_value,
    free_value,
    occupied_value,
    pose_cov=None,
):
    changed = False
    saw_obstacle_hit = False
    x0, y0 = float(robot.x), float(robot.y)
    theta = float(robot.angle)
    for ray in scan:
        rel_angle = float(ray.get('rel_angle_deg', ((float(ray['angle_deg']) - theta + 180.0) % 360.0) - 180.0))
        ang = (theta + rel_angle) % 360.0
        ang_rad = math.radians(ang)
        dist = float(ray['distance'])
        x1 = x0 + dist * math.cos(ang_rad)
        y1 = y0 + dist * math.sin(ang_rad)
        base_weight = _observation_weight(pose_cov, dist, view_distance)
        obs_pose_uncertainty = _projected_hit_uncertainty_m(pose_cov, dist)
        steps = max(2, int(math.ceil(dist / max(0.18, 0.45 * grid_resolution))))
        for i in range(steps):
            t = i / steps
            x = x0 + t * (x1 - x0)
            y = y0 + t * (y1 - y0)
            gx, gy = world_to_grid(x, y)
            free_weight = base_weight * (0.65 + 0.25 * (1.0 - t))
            ray_unc = max(0.02, obs_pose_uncertainty * (MAP_POSE_UNCERTAINTY_FREE_DECAY + (1.0 - MAP_POSE_UNCERTAINTY_FREE_DECAY) * t))
            changed = bool(_weighted_cell_update(grid, logodds_grid, confidence_grid, pose_uncertainty_grid, gx, gy, MAP_FREE_LOGODDS_DELTA, free_weight, ray_unc, unknown_value, free_value, occupied_value) or changed)
        gx_hit, gy_hit = world_to_grid(x1, y1)
        is_obstacle_hit = bool(ray.get('is_obstacle_hit', dist < view_distance - 1e-6))
        if is_obstacle_hit:
            saw_obstacle_hit = True
            changed = bool(_weighted_cell_update(grid, logodds_grid, confidence_grid, pose_uncertainty_grid, gx_hit, gy_hit, MAP_OCCUPIED_LOGODDS_DELTA, base_weight, obs_pose_uncertainty, unknown_value, free_value, occupied_value) or changed)
            changed = bool(stamp_obstacle_hit(grid, gx_hit, gy_hit) or changed)
        else:
            changed = bool(_weighted_cell_update(grid, logodds_grid, confidence_grid, pose_uncertainty_grid, gx_hit, gy_hit, MAP_FREE_LOGODDS_DELTA, base_weight * 0.60, obs_pose_uncertainty, unknown_value, free_value, occupied_value) or changed)
    return changed, saw_obstacle_hit


def update_known_map_from_scan(drone, shared_known_grid, shared_logodds_grid, shared_confidence_grid, shared_pose_uncertainty_grid, scan, apply_to_grid, occupied_value):
    before_shared = int(np.count_nonzero(shared_known_grid == occupied_value))
    before_local = int(np.count_nonzero(drone['local_known_grid'] == occupied_value))

    x_est, y_est, theta_est = map(float, drone['odometry'].mu)
    pose_proxy = SimpleNamespace(x=x_est, y=y_est, angle=theta_est)
    pose_cov = np.array(drone['odometry'].cov, dtype=float)

    local_changed, local_saw_obstacle = apply_to_grid(
        drone['local_known_grid'],
        drone['local_logodds_grid'],
        drone['local_confidence_grid'],
        drone['local_pose_uncertainty_grid'],
        pose_proxy,
        scan,
        pose_cov,
    )
    local_occ_after_apply = int(np.count_nonzero(drone['local_known_grid'] == occupied_value))
    if local_saw_obstacle and local_occ_after_apply > before_local:
        solidify_obstacle_interiors(drone['local_known_grid'], 0, occupied_value)
        _sync_solidified_cells(drone['local_known_grid'], drone['local_logodds_grid'], drone['local_confidence_grid'], drone['local_pose_uncertainty_grid'], occupied_value)

    shared_changed, shared_saw_obstacle = apply_to_grid(
        shared_known_grid,
        shared_logodds_grid,
        shared_confidence_grid,
        shared_pose_uncertainty_grid,
        pose_proxy,
        scan,
        pose_cov,
    )
    shared_occ_after_apply = int(np.count_nonzero(shared_known_grid == occupied_value))
    if shared_saw_obstacle and shared_occ_after_apply > before_shared:
        solidify_obstacle_interiors(shared_known_grid, 0, occupied_value)
        _sync_solidified_cells(shared_known_grid, shared_logodds_grid, shared_confidence_grid, shared_pose_uncertainty_grid, occupied_value)

    after_shared = int(np.count_nonzero(shared_known_grid == occupied_value))
    after_local = int(np.count_nonzero(drone['local_known_grid'] == occupied_value))
    drone['known_occ_count'] = after_shared
    drone['local_known_occ_count'] = after_local
    drone['just_discovered_obstacle'] = after_shared > before_shared
    return bool(local_changed or shared_changed or after_shared != before_shared or after_local != before_local)


def clearance_groups(scan):
    distances = np.array([r['distance'] for r in scan], dtype=float)
    n = len(distances)
    third = max(1, n // 3)
    left = distances[:third]
    center = distances[third:n - third] if n - third > third else distances[third:]
    right = distances[n - third:]
    return left, center, right

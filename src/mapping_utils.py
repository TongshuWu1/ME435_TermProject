import math

import numpy as np


def reveal_start_area(grid, x, y, radius_m, world_to_grid, offsets_for_margin, nx, ny, unknown_value, free_value):
    gx, gy = world_to_grid(x, y)
    for dx, dy in offsets_for_margin(radius_m):
        nx_cell = gx + dx
        ny_cell = gy + dy
        if 0 <= nx_cell < nx and 0 <= ny_cell < ny and grid[ny_cell, nx_cell] == unknown_value:
            grid[ny_cell, nx_cell] = free_value


def _pose_components(pose):
    if hasattr(pose, 'x') and hasattr(pose, 'y') and hasattr(pose, 'angle'):
        return float(pose.x), float(pose.y), float(pose.angle)
    if isinstance(pose, np.ndarray):
        pose = pose.tolist()
    if isinstance(pose, (tuple, list)) and len(pose) >= 3:
        return float(pose[0]), float(pose[1]), float(pose[2])
    raise TypeError('pose must be a robot-like object or a length-3 iterable [x, y, heading_deg]')


def apply_scan_to_grid(grid, pose, scan, world_to_grid, stamp_obstacle_hit, grid_resolution, view_distance, unknown_value, free_value):
    x0, y0, theta_deg = _pose_components(pose)

    for ray in scan:
        dist = float(ray['distance'])
        rel_angle_deg = float(ray.get('relative_angle_deg', ray.get('angle_deg', theta_deg) - theta_deg))
        ray_angle_rad = math.radians(theta_deg + rel_angle_deg)
        x1 = x0 + dist * math.cos(ray_angle_rad)
        y1 = y0 + dist * math.sin(ray_angle_rad)

        steps = max(2, int(math.ceil(dist / (0.45 * grid_resolution))))
        for i in range(steps):
            t = i / steps
            x = x0 + t * (x1 - x0)
            y = y0 + t * (y1 - y0)
            gx, gy = world_to_grid(x, y)
            if grid[gy, gx] == unknown_value:
                grid[gy, gx] = free_value

        gx_hit, gy_hit = world_to_grid(x1, y1)
        if dist < view_distance - 1e-6:
            stamp_obstacle_hit(grid, gx_hit, gy_hit)
        elif grid[gy_hit, gx_hit] == unknown_value:
            grid[gy_hit, gx_hit] = free_value


def update_known_map_from_scan(drone, shared_known_grid, scan, apply_to_grid, occupied_value):
    before_shared = int(np.count_nonzero(shared_known_grid == occupied_value))
    before_local = int(np.count_nonzero(drone['local_known_grid'] == occupied_value))

    est_pose = np.array(drone['odometry'].mu, dtype=float)
    apply_to_grid(drone['local_known_grid'], est_pose, scan)
    apply_to_grid(shared_known_grid, est_pose, scan)

    after_shared = int(np.count_nonzero(shared_known_grid == occupied_value))
    after_local = int(np.count_nonzero(drone['local_known_grid'] == occupied_value))
    drone['known_occ_count'] = after_shared
    drone['local_known_occ_count'] = after_local
    drone['just_discovered_obstacle'] = after_shared > before_shared


def clearance_groups(scan):
    distances = np.array([r['distance'] for r in scan], dtype=float)
    n = len(distances)
    third = max(1, n // 3)
    left = distances[:third]
    center = distances[third:n - third] if n - third > third else distances[third:]
    right = distances[n - third:]
    return left, center, right

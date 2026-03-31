import math

import numpy as np


def reveal_start_area(grid, x, y, radius_m, world_to_grid, offsets_for_margin, nx, ny, unknown_value, free_value):
    gx, gy = world_to_grid(x, y)
    for dx, dy in offsets_for_margin(radius_m):
        nx_cell = gx + dx
        ny_cell = gy + dy
        if 0 <= nx_cell < nx and 0 <= ny_cell < ny and grid[ny_cell, nx_cell] == unknown_value:
            grid[ny_cell, nx_cell] = free_value


def apply_scan_to_grid(grid, robot, scan, world_to_grid, stamp_obstacle_hit, grid_resolution, view_distance, unknown_value, free_value):
    for ray in scan:
        x0, y0 = robot.x, robot.y
        x1, y1 = ray['hit_x'], ray['hit_y']
        dist = ray['distance']
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

    apply_to_grid(drone['local_known_grid'], drone['robot'], scan)
    apply_to_grid(shared_known_grid, drone['robot'], scan)

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

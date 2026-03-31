import math
from collections import deque

import numpy as np

from config import (
    ENCLOSED_FILL_MAX_CELLS,
    ENCLOSED_FILL_MIN_CELLS,
    ENCLOSED_FILL_SEAL_ONE_CELL_GAPS,
    FILL_ENCLOSED_OBSTACLE_INTERIORS,
)


def reveal_start_area(grid, x, y, radius_m, world_to_grid, offsets_for_margin, nx, ny, unknown_value, free_value):
    gx, gy = world_to_grid(x, y)
    for dx, dy in offsets_for_margin(radius_m):
        nx_cell = gx + dx
        ny_cell = gy + dy
        if 0 <= nx_cell < nx and 0 <= ny_cell < ny and grid[ny_cell, nx_cell] == unknown_value:
            grid[ny_cell, nx_cell] = free_value


def _seal_small_gaps(occupied_mask, radius_cells=1):
    """Close tiny one/two-cell slits without performing a broad dilation.

    A cell is sealed if short occupied runs exist on opposite sides horizontally,
    vertically, or diagonally. This keeps tiny sensor gaps from preventing a
    solid obstacle fill while staying conservative around real passages.
    """
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
    if connectivity == 8:
        nbrs = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
    else:
        nbrs = ((1, 0), (-1, 0), (0, 1), (0, -1))

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
            components.append({
                'label': label,
                'points': pts,
                'min_x': min_x,
                'max_x': max_x,
                'min_y': min_y,
                'max_y': max_y,
                'size': len(pts),
            })
            label += 1
    return labels, components


def _solidify_rectangular_components(grid, occupied_mask, unknown_value, occupied_value, max_cells):
    """Fill interiors of obstacle blobs that look like solid axis-aligned blocks.

    The simulated environment uses square/rectangular obstacles, so once a blob
    has support along the sides of a bounding box and enough occupied density,
    we treat the obstacle as solid even if the sensed outline still has tiny gaps.
    """
    labels, components = _component_labels(occupied_mask, connectivity=8)
    ny, nx = grid.shape
    filled = 0

    for comp in components:
        min_x, max_x = comp['min_x'], comp['max_x']
        min_y, max_y = comp['min_y'], comp['max_y']
        width = max_x - min_x + 1
        height = max_y - min_y + 1
        bbox_area = width * height
        if bbox_area <= 0 or bbox_area > max_cells:
            continue
        if width < 3 or height < 3:
            continue
        if min_x <= 1 or min_y <= 1 or max_x >= nx - 2 or max_y >= ny - 2:
            continue

        comp_size = comp['size']
        fill_ratio = comp_size / float(bbox_area)
        if fill_ratio < 0.22:
            continue

        comp_mask = labels[min_y:max_y + 1, min_x:max_x + 1] == comp['label']
        row_counts = comp_mask.sum(axis=1)
        col_counts = comp_mask.sum(axis=0)

        row_support = np.count_nonzero(row_counts >= max(1, int(math.ceil(0.45 * width))))
        col_support = np.count_nonzero(col_counts >= max(1, int(math.ceil(0.45 * height))))
        has_top = np.any(comp_mask[0, :])
        has_bottom = np.any(comp_mask[-1, :])
        has_left = np.any(comp_mask[:, 0])
        has_right = np.any(comp_mask[:, -1])
        side_hits = sum((has_top, has_bottom, has_left, has_right))

        # Strong case: almost box-like already.
        strong_box = fill_ratio >= 0.50 and side_hits >= 3
        # Moderate case: enough support in both dimensions plus multiple sides.
        moderate_box = fill_ratio >= 0.32 and row_support >= 2 and col_support >= 2 and side_hits >= 2
        # Fallback compact case for chunky blobs whose stamp nearly fills the box.
        chunky_box = fill_ratio >= 0.62

        if not (strong_box or moderate_box or chunky_box):
            continue

        subgrid = grid[min_y:max_y + 1, min_x:max_x + 1]
        fill_mask = subgrid != occupied_value
        if not np.any(fill_mask):
            continue
        # Avoid converting large known-free corridors into obstacle just because
        # a sparse blob produced a big box. Require most of the box to be unknown
        # or already occupied, not confidently free.
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

    # Seed from all image edges, but only non-occupied cells. This preserves the
    # occupied boundary while correctly marking truly outside reachable space.
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
            if touches_boundary:
                continue
            if comp_size < min_cells or comp_size > max_cells:
                continue
            if unknown_count == 0:
                continue
            for cx, cy in comp:
                if grid[cy, cx] != occupied_value:
                    grid[cy, cx] = occupied_value
                    filled += 1
    return filled


def solidify_obstacle_interiors(
    grid,
    unknown_value,
    occupied_value,
    max_cells=ENCLOSED_FILL_MAX_CELLS,
    min_cells=ENCLOSED_FILL_MIN_CELLS,
    seal_one_cell_gaps=ENCLOSED_FILL_SEAL_ONE_CELL_GAPS,
):
    if not FILL_ENCLOSED_OBSTACLE_INTERIORS:
        return 0

    filled = 0
    occupied_mask = grid == occupied_value
    if seal_one_cell_gaps:
        occupied_mask = _seal_small_gaps(occupied_mask, radius_cells=1)

    filled += _solidify_rectangular_components(grid, occupied_mask, unknown_value, occupied_value, max_cells)
    filled += _fill_enclosed_unknown_pockets(grid, unknown_value, occupied_value, max_cells, min_cells, seal_one_cell_gaps)
    return filled


def apply_scan_to_grid(grid, robot, scan, world_to_grid, stamp_obstacle_hit, grid_resolution, view_distance, unknown_value, free_value):
    changed = False
    saw_obstacle_hit = False
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
                changed = True
        gx_hit, gy_hit = world_to_grid(x1, y1)
        if dist < view_distance - 1e-6:
            saw_obstacle_hit = True
            changed = bool(stamp_obstacle_hit(grid, gx_hit, gy_hit) or changed)
        elif grid[gy_hit, gx_hit] == unknown_value:
            grid[gy_hit, gx_hit] = free_value
            changed = True
    return changed, saw_obstacle_hit


def update_known_map_from_scan(drone, shared_known_grid, scan, apply_to_grid, occupied_value):
    before_shared = int(np.count_nonzero(shared_known_grid == occupied_value))
    before_local = int(np.count_nonzero(drone['local_known_grid'] == occupied_value))

    local_changed, local_saw_obstacle = apply_to_grid(drone['local_known_grid'], drone['robot'], scan)
    if local_saw_obstacle:
        solidify_obstacle_interiors(drone['local_known_grid'], 0, occupied_value)

    shared_changed, shared_saw_obstacle = apply_to_grid(shared_known_grid, drone['robot'], scan)
    if shared_saw_obstacle:
        solidify_obstacle_interiors(shared_known_grid, 0, occupied_value)

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

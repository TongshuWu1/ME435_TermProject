import heapq
import math
from collections import OrderedDict

import numpy as np

from config import (
    A_STAR_MIN_SEGMENT_CLEARANCE_METERS,
    A_STAR_PATH_CLEARANCE_WEIGHT,
    WORLD_HEIGHT_METERS,
    WORLD_WIDTH_METERS,
)


class GridPlanner:
    def __init__(self, grid_resolution, inflation_margin):
        self.grid_resolution = float(grid_resolution)
        self.inflation_margin = float(inflation_margin)
        self.nx = int(round(WORLD_WIDTH_METERS / self.grid_resolution))
        self.ny = int(round(WORLD_HEIGHT_METERS / self.grid_resolution))
        self.planning_inflation_offsets = self.offsets_for_margin(self.inflation_margin)
        self.hit_stamp_offsets = [(0, 0)]
        self._grid_x, self._grid_y = np.meshgrid(
            np.arange(self.nx, dtype=np.float32),
            np.arange(self.ny, dtype=np.float32),
        )
        self._planning_occ_cache = OrderedDict()
        self._clearance_field_cache = OrderedDict()
        self._cache_limit = 12

    def offsets_for_margin(self, margin_m):
        radius_cells = max(0, int(math.ceil(float(margin_m) / self.grid_resolution)))
        offsets = []
        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                if math.hypot(dx, dy) * self.grid_resolution <= float(margin_m) + 1e-9:
                    offsets.append((dx, dy))
        return offsets


    def _cache_key(self, array, extra=None):
        arr = np.ascontiguousarray(array)
        return (arr.shape, arr.dtype.str, extra, arr.tobytes())

    def _get_cached(self, cache, key):
        value = cache.get(key)
        if value is None:
            return None
        cache.move_to_end(key)
        return value

    def _set_cached(self, cache, key, value):
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > self._cache_limit:
            cache.popitem(last=False)
        return value

    def world_to_grid(self, x, y):
        gx = int(round((float(x) - 0.5 * self.grid_resolution) / self.grid_resolution))
        gy = int(round((float(y) - 0.5 * self.grid_resolution) / self.grid_resolution))
        gx = 0 if gx < 0 else self.nx - 1 if gx >= self.nx else gx
        gy = 0 if gy < 0 else self.ny - 1 if gy >= self.ny else gy
        return gx, gy

    def grid_to_world(self, cell):
        gx, gy = cell
        return (gx + 0.5) * self.grid_resolution, (gy + 0.5) * self.grid_resolution

    def init_known_grid(self, occupied_value):
        known = np.zeros((self.ny, self.nx), dtype=np.uint8)
        known[0, :] = occupied_value
        known[-1, :] = occupied_value
        known[:, 0] = occupied_value
        known[:, -1] = occupied_value
        return known

    def build_truth_occupancy_grid(self, obstacles):
        occ = np.zeros((self.ny, self.nx), dtype=bool)
        # Keep the truth grid exact: obstacles occupy only their snapped grid footprint.
        # Robot clearance belongs in planning inflation, not in the obstacle truth map.
        for gy in range(self.ny):
            for gx in range(self.nx):
                x, y = self.grid_to_world((gx, gy))
                if gx == 0 or gy == 0 or gx == self.nx - 1 or gy == self.ny - 1:
                    occ[gy, gx] = True
                    continue
                for obs in obstacles:
                    half = obs['size'] / 2.0
                    if (obs['x'] - half) <= x <= (obs['x'] + half) and (obs['y'] - half) <= y <= (obs['y'] + half):
                        occ[gy, gx] = True
                        break
        return occ

    def inflate_occ_mask(self, occ_mask, offsets=None):
        offsets = self.planning_inflation_offsets if offsets is None else offsets
        inflated = occ_mask.copy()
        for gy, gx in np.argwhere(occ_mask):
            for dx, dy in offsets:
                nx = gx + dx
                ny = gy + dy
                if 0 <= nx < self.nx and 0 <= ny < self.ny:
                    inflated[ny, nx] = True
        return inflated

    def stamp_obstacle_hit(self, known_grid, gx, gy, occupied_value):
        changed = False
        for dx, dy in self.hit_stamp_offsets:
            nx = gx + dx
            ny = gy + dy
            if 0 <= nx < self.nx and 0 <= ny < self.ny and known_grid[ny, nx] != occupied_value:
                known_grid[ny, nx] = occupied_value
                changed = True
        return changed

    def planning_occupancy(self, known_grid, occupied_value):
        key = self._cache_key(known_grid, extra=int(occupied_value))
        cached = self._get_cached(self._planning_occ_cache, key)
        if cached is not None:
            return cached
        occ_mask = np.asarray(known_grid == occupied_value, dtype=bool)
        if self.planning_inflation_offsets:
            occ_mask = self.inflate_occ_mask(occ_mask)
        return self._set_cached(self._planning_occ_cache, key, occ_mask)

    def clearance_field(self, planning_occ):
        key = self._cache_key(np.asarray(planning_occ, dtype=np.uint8), extra='clearance')
        cached = self._get_cached(self._clearance_field_cache, key)
        if cached is not None:
            return cached

        occ = np.asarray(planning_occ, dtype=bool)
        if not np.any(occ):
            field = np.full((self.ny, self.nx), float(max(self.nx, self.ny)), dtype=np.float32)
            return self._set_cached(self._clearance_field_cache, key, field)

        inf = np.float32(1e9)
        step = np.float32(1.0)
        diag = np.float32(math.sqrt(2.0))
        field = np.full((self.ny, self.nx), inf, dtype=np.float32)
        field[occ] = 0.0

        for y in range(self.ny):
            for x in range(self.nx):
                best = field[y, x]
                if y > 0:
                    best = min(best, field[y - 1, x] + step)
                    if x > 0:
                        best = min(best, field[y - 1, x - 1] + diag)
                    if x + 1 < self.nx:
                        best = min(best, field[y - 1, x + 1] + diag)
                if x > 0:
                    best = min(best, field[y, x - 1] + step)
                field[y, x] = best

        for y in range(self.ny - 1, -1, -1):
            for x in range(self.nx - 1, -1, -1):
                best = field[y, x]
                if y + 1 < self.ny:
                    best = min(best, field[y + 1, x] + step)
                    if x > 0:
                        best = min(best, field[y + 1, x - 1] + diag)
                    if x + 1 < self.nx:
                        best = min(best, field[y + 1, x + 1] + diag)
                if x + 1 < self.nx:
                    best = min(best, field[y, x + 1] + step)
                field[y, x] = best

        return self._set_cached(self._clearance_field_cache, key, field)

    def _cell_has_clearance(self, gx, gy, planning_occ, clearance_cells=0, clearance_field=None):
        if planning_occ[gy, gx]:
            return False
        clearance_cells = max(0.0, float(clearance_cells))
        if clearance_cells <= 0.0:
            return True
        field = self.clearance_field(planning_occ) if clearance_field is None else clearance_field
        return bool(field[gy, gx] >= clearance_cells)

    def clearance_distance_cells(self, gx, gy, planning_occ, clearance_field=None):
        if not (0 <= gx < self.nx and 0 <= gy < self.ny):
            return 0.0
        field = self.clearance_field(planning_occ) if clearance_field is None else clearance_field
        return float(field[gy, gx])

    def clearance_distance_world(self, xy, planning_occ, clearance_field=None):
        gx, gy = self.world_to_grid(*xy)
        return self.clearance_distance_cells(gx, gy, planning_occ, clearance_field=clearance_field) * self.grid_resolution

    def nearest_free_cell(self, cell, planning_occ, clearance_cells=0, clearance_field=None):
        gx, gy = cell
        target_clearance = max(0.0, float(clearance_cells))
        field = self.clearance_field(planning_occ) if clearance_field is None else clearance_field
        if self._cell_has_clearance(gx, gy, planning_occ, clearance_cells=target_clearance, clearance_field=field):
            return gx, gy
        best_fallback = None
        best_fallback_score = -1e9
        for radius in range(1, max(self.nx, self.ny)):
            shell = []
            for ny in range(max(0, gy - radius), min(self.ny, gy + radius + 1)):
                for nx in range(max(0, gx - radius), min(self.nx, gx + radius + 1)):
                    if abs(nx - gx) != radius and abs(ny - gy) != radius:
                        continue
                    if planning_occ[ny, nx]:
                        continue
                    clear_score = float(field[ny, nx])
                    score = clear_score - 0.08 * math.hypot(nx - gx, ny - gy)
                    shell.append((score, clear_score, nx, ny))
            if not shell:
                continue
            feasible = [item for item in shell if item[1] >= target_clearance]
            if feasible:
                feasible.sort(key=lambda item: (item[0], item[1]), reverse=True)
                _, _, nx, ny = feasible[0]
                return nx, ny
            shell.sort(key=lambda item: (item[0], item[1]), reverse=True)
            if shell[0][0] > best_fallback_score:
                best_fallback_score = shell[0][0]
                best_fallback = (shell[0][2], shell[0][3])
        return best_fallback if best_fallback is not None else (gx, gy)

    @staticmethod
    def heuristic(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def line_crosses_blocked(self, p0, p1, planning_occ, sample_step=None):
        x0, y0 = p0
        x1, y1 = p1
        if sample_step is None:
            sample_step = max(0.15, 0.22 * self.grid_resolution)
        dist = math.hypot(x1 - x0, y1 - y0)
        if dist < 1e-9:
            return False
        num = max(2, int(math.ceil(dist / sample_step)))
        for i in range(num + 1):
            t = i / num
            x = x0 + t * (x1 - x0)
            y = y0 + t * (y1 - y0)
            gx, gy = self.world_to_grid(x, y)
            if planning_occ[gy, gx]:
                return True
        return False

    def line_has_clearance(self, p0, p1, planning_occ, min_clearance_m=A_STAR_MIN_SEGMENT_CLEARANCE_METERS, sample_step=None, clearance_field=None):
        x0, y0 = p0
        x1, y1 = p1
        if sample_step is None:
            sample_step = max(0.15, 0.22 * self.grid_resolution)
        dist = math.hypot(x1 - x0, y1 - y0)
        field = self.clearance_field(planning_occ) if clearance_field is None else clearance_field
        if dist < 1e-9:
            return self.clearance_distance_world((x0, y0), planning_occ, clearance_field=field) >= float(min_clearance_m)
        num = max(2, int(math.ceil(dist / sample_step)))
        for i in range(num + 1):
            t = i / num
            x = x0 + t * (x1 - x0)
            y = y0 + t * (y1 - y0)
            if self.clearance_distance_world((x, y), planning_occ, clearance_field=field) < float(min_clearance_m):
                return False
        return True

    def compress_path(self, path, planning_occ, clearance_field=None):
        if len(path) <= 2:
            return path
        compressed = [path[0]]
        anchor = path[0]
        idx = 1
        while idx < len(path):
            furthest = idx
            while (
                furthest + 1 < len(path)
                and not self.line_crosses_blocked(anchor, path[furthest + 1], planning_occ)
                and self.line_has_clearance(anchor, path[furthest + 1], planning_occ, clearance_field=clearance_field)
            ):
                furthest += 1
            compressed.append(path[furthest])
            anchor = path[furthest]
            idx = furthest + 1
        return compressed

    def astar_on_occupancy(self, start_xy, goal_xy, planning_occ, goal_clearance_cells=0):
        clearance_field = self.clearance_field(planning_occ)
        start = self.nearest_free_cell(self.world_to_grid(*start_xy), planning_occ, clearance_cells=0, clearance_field=clearance_field)
        goal = self.nearest_free_cell(self.world_to_grid(*goal_xy), planning_occ, clearance_cells=goal_clearance_cells, clearance_field=clearance_field)
        safe_goal_xy = self.grid_to_world(goal)
        if start == goal:
            return [start_xy, safe_goal_xy]

        open_heap = [(0.0, start)]
        came_from = {}
        g_score = {start: 0.0}
        closed = set()
        neighbors = [
            (-1, -1), (0, -1), (1, -1),
            (-1, 0),            (1, 0),
            (-1, 1),  (0, 1),   (1, 1),
        ]

        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            if current == goal:
                break
            closed.add(current)

            for dx, dy in neighbors:
                nx, ny = current[0] + dx, current[1] + dy
                if not (0 <= nx < self.nx and 0 <= ny < self.ny):
                    continue
                if planning_occ[ny, nx]:
                    continue
                if dx != 0 and dy != 0 and (planning_occ[current[1], nx] or planning_occ[ny, current[0]]):
                    continue
                step_cost = math.sqrt(2.0) if dx != 0 and dy != 0 else 1.0
                clearance_cells = self.clearance_distance_cells(nx, ny, planning_occ, clearance_field=clearance_field)
                clearance_m = max(0.05, clearance_cells * self.grid_resolution)
                clearance_penalty = float(A_STAR_PATH_CLEARANCE_WEIGHT) / clearance_m
                tentative_g = g_score[current] + step_cost + clearance_penalty
                neighbor = (nx, ny)
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_heap, (f, neighbor))

        if goal not in came_from and goal != start:
            return None

        cells = [goal]
        cur = goal
        while cur != start:
            cur = came_from[cur]
            cells.append(cur)
        cells.reverse()

        path = [start_xy]
        path.extend(self.grid_to_world(cell) for cell in cells[1:-1])
        path.append(safe_goal_xy)
        return self.compress_path(path, planning_occ, clearance_field=clearance_field)

    def astar(self, start_xy, goal_xy, known_grid, occupied_value, goal_clearance_cells=0):
        planning_occ = self.planning_occupancy(known_grid, occupied_value)
        return self.astar_on_occupancy(start_xy, goal_xy, planning_occ, goal_clearance_cells=goal_clearance_cells)

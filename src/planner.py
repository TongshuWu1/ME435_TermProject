import heapq
import math

import numpy as np

from config import WORLD_HEIGHT_METERS, WORLD_WIDTH_METERS


class GridPlanner:
    def __init__(self, grid_resolution, inflation_margin):
        self.grid_resolution = float(grid_resolution)
        self.inflation_margin = float(inflation_margin)
        self.nx = int(round(WORLD_WIDTH_METERS / self.grid_resolution))
        self.ny = int(round(WORLD_HEIGHT_METERS / self.grid_resolution))
        self.planning_inflation_offsets = self.offsets_for_margin(self.inflation_margin)
        self.hit_stamp_offsets = self.offsets_for_margin(max(0.55, 0.7 * self.inflation_margin))

    def offsets_for_margin(self, margin_m):
        radius_cells = max(0, int(math.ceil(float(margin_m) / self.grid_resolution)))
        offsets = []
        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                if math.hypot(dx, dy) * self.grid_resolution <= float(margin_m) + 1e-9:
                    offsets.append((dx, dy))
        return offsets

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
        margin = self.inflation_margin + 0.25
        for gy in range(self.ny):
            for gx in range(self.nx):
                x, y = self.grid_to_world((gx, gy))
                if x < margin or x > WORLD_WIDTH_METERS - margin or y < margin or y > WORLD_HEIGHT_METERS - margin:
                    occ[gy, gx] = True
                    continue
                for obs in obstacles:
                    half = obs['size'] / 2.0 + margin
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
        occ_mask = known_grid == occupied_value
        if self.planning_inflation_offsets:
            occ_mask = self.inflate_occ_mask(occ_mask)
        return occ_mask

    def nearest_free_cell(self, cell, planning_occ):
        gx, gy = cell
        if not planning_occ[gy, gx]:
            return gx, gy
        for radius in range(1, max(self.nx, self.ny)):
            for ny in range(max(0, gy - radius), min(self.ny, gy + radius + 1)):
                for nx in range(max(0, gx - radius), min(self.nx, gx + radius + 1)):
                    if abs(nx - gx) != radius and abs(ny - gy) != radius:
                        continue
                    if not planning_occ[ny, nx]:
                        return nx, ny
        return gx, gy

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

    def compress_path(self, path, planning_occ):
        if len(path) <= 2:
            return path
        compressed = [path[0]]
        anchor = path[0]
        idx = 1
        while idx < len(path):
            furthest = idx
            while furthest + 1 < len(path) and not self.line_crosses_blocked(anchor, path[furthest + 1], planning_occ):
                furthest += 1
            compressed.append(path[furthest])
            anchor = path[furthest]
            idx = furthest + 1
        return compressed

    def astar(self, start_xy, goal_xy, known_grid, occupied_value):
        planning_occ = self.planning_occupancy(known_grid, occupied_value)
        start = self.nearest_free_cell(self.world_to_grid(*start_xy), planning_occ)
        goal = self.nearest_free_cell(self.world_to_grid(*goal_xy), planning_occ)
        if start == goal:
            return [start_xy, goal_xy]

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
                tentative_g = g_score[current] + step_cost
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
        path.append(goal_xy)
        return self.compress_path(path, planning_occ)

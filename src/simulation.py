import math
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.animation import FuncAnimation

from .config import (
    A_STAR_GOAL_TOLERANCE,
    A_STAR_GRID_LINE_ALPHA,
    A_STAR_GRID_MAX_DRONES,
    A_STAR_GRID_OCC_ALPHA,
    A_STAR_GRID_PATH_ALPHA,
    A_STAR_GRID_RESOLUTION,
    A_STAR_GRID_UPDATE_FRAMES,
    A_STAR_GRID_WINDOW_CELLS,
    A_STAR_INFLATION_MARGIN,
    A_STAR_LOOKAHEAD_STEPS,
    A_STAR_REPLAN_SECONDS,
    COLOR_BOUNDARY,
    COLOR_OBSTACLE,
    DRONE_NAMES,
    DRONE_START_POSE,
    DRONE_START_SPACING,
    FOV_ANGLE,
    KNOWN_MAP_REPLAN_ON_NEW_OBS,
    MEASUREMENT_ALPHA,
    MEASUREMENT_NOISE,
    OBSTACLE_AVOID_DISTANCE,
    OBSTACLE_TURN_GAIN,
    PATH_LINE_WIDTH,
    PLANNING_MAP_MODE,
    PREDICTION_NOISE,
    RANDOM_SEED,
    RAY_ALPHA,
    RAY_LINE_WIDTH,
    SHOW_ASTAR_LOCAL_GRID,
    SHOW_VISION_RAYS,
    START_SIMULATION_RUNNING,
    STUCK_PROGRESS_EPS,
    STUCK_RECOVERY_SECONDS,
    STUCK_REVERSE_SPEED,
    STUCK_TURN_SPEED,
    STUCK_WINDOW_SECONDS,
    SUBGOAL_MARKER_SIZE,
    TARGET_MARKER_SIZE,
    TIME_STEP,
    VIEW_DISTANCE,
    VISION_RAY_COUNT,
    WORLD_HEIGHT_METERS,
    WORLD_WIDTH_METERS,
)
from .environment import empty_target_sequences, generate_environment, square_contains_point
from .landmark import Landmark
from .localization import OdometryEstimator
from .mapping_utils import apply_scan_to_grid, clearance_groups, reveal_start_area, update_known_map_from_scan
from .output_utils import save_map_outputs, save_trajectory_png
from .planner import GridPlanner
from .robot import Robot
from .sim_ui import SimulatorUI

RENDER_FPS = 30
UNKNOWN = 0
FREE = 1
OCCUPIED = 2

class Simulator:
    def __init__(self):
        self.current_seed = RANDOM_SEED
        self.landmark_patches = []
        self.obstacle_patches = []
        self.landmarks = []
        self.obstacles = []
        self.fov_angle = FOV_ANGLE
        self.view_distance = VIEW_DISTANCE
        self.auto_mode = START_SIMULATION_RUNNING
        self.trace_interval = 5
        self.trace_counter = 0
        self.time_elapsed = 0.0

        self.grid_resolution = A_STAR_GRID_RESOLUTION
        self.planner = GridPlanner(A_STAR_GRID_RESOLUTION, A_STAR_INFLATION_MARGIN)
        self.nx = self.planner.nx
        self.ny = self.planner.ny
        self.truth_occupancy = self._build_truth_occupancy_grid()
        self.shared_known_grid = self._init_known_grid()

        self.selected_drone_index = 0
        self.edit_mode = 'add_waypoint'
        self.last_saved_plot_path = ''
        self.last_saved_map_path = ''

        self.ui = SimulatorUI(self)
        self.fig, self.ax = self.ui.build()

        self.ax.add_patch(
            patches.Rectangle(
                (0, 0),
                WORLD_WIDTH_METERS,
                WORLD_HEIGHT_METERS,
                linewidth=2.0,
                edgecolor=COLOR_BOUNDARY,
                facecolor='none',
            )
        )

        self.colors = plt.cm.tab10(np.linspace(0, 1, len(DRONE_NAMES)))
        self._load_environment(self.current_seed)

        self.drones = []

        for idx, path in enumerate(self.generated_target_sequences):
            drone_name = DRONE_NAMES[idx] if idx < len(DRONE_NAMES) else f'Drone {idx + 1}'
            color = self.colors[idx]
            drone = self._create_drone(drone_name, color, path, drone_index=idx, drone_count=len(self.generated_target_sequences))
            self._reveal_start_area(drone['local_known_grid'], drone['robot'].x, drone['robot'].y)
            self._reveal_start_area(self.shared_known_grid, drone['robot'].x, drone['robot'].y)
            self.drones.append(drone)

        legend_handles = [
            patches.Patch(facecolor=self.colors[i], edgecolor='black', label=self.drones[i]['name'])
            for i in range(len(self.drones))
        ]
        self.ax.legend(handles=legend_handles, loc='upper left')
        self.ui.refresh_all()

    def _refresh_toggle_button(self):
        self.ui.refresh_toggle_button()

    def _refresh_control_text(self):
        self.ui.refresh_status_text()

    def on_select_robot(self, label):
        try:
            self.selected_drone_index = DRONE_NAMES.index(label)
        except ValueError:
            self.selected_drone_index = 0
        self._refresh_control_text()


    def on_select_edit_mode(self, label):
        norm = str(label).strip().lower()
        self.edit_mode = 'set_start' if 'start' in norm else 'add_waypoint'
        self._refresh_control_text()

    def _toolbar_active(self):
        toolbar = getattr(self.fig.canvas, 'toolbar', None)
        mode = getattr(toolbar, 'mode', '') if toolbar is not None else ''
        return bool(mode)

    def _point_in_free_space(self, x, y, extra_margin=0.15):
        if x is None or y is None:
            return False
        if not (0.2 <= x <= WORLD_WIDTH_METERS - 0.2 and 0.2 <= y <= WORLD_HEIGHT_METERS - 0.2):
            return False
        if self.truth_occupancy[self._world_to_grid(x, y)[1], self._world_to_grid(x, y)[0]]:
            return False
        for obs in self.obstacles:
            if square_contains_point(obs, x, y, margin=extra_margin):
                return False
        return True

    def _set_drone_start(self, drone, x, y):
        _, _, angle = drone['odometry'].mu
        self._reset_drone_state(
            drone,
            float(x),
            float(y),
            float(angle),
            reset_local_grid=False,
            discovered_flag=True,
        )

    def on_map_click(self, event):
        if event.inaxes != self.ax or event.button != 1:
            return
        if self._toolbar_active():
            return
        x = event.xdata
        y = event.ydata
        if not self._point_in_free_space(x, y):
            return
        if not self.drones:
            return
        drone = self.drones[self.selected_drone_index]
        if self.edit_mode == 'set_start':
            if self.auto_mode:
                self.auto_mode = False
                self._refresh_toggle_button()
            self._set_drone_start(drone, float(x), float(y))
        else:
            drone['path'].append((float(x), float(y)))
            if len(drone['path']) == 1:
                drone['current_target_index'] = 0
            self._update_mission_artist(drone)
            drone['plan_goal_index'] = -1
            drone['last_plan_time'] = -1e9
            drone['just_discovered_obstacle'] = True
        self.ui.refresh_all()
        self.fig.canvas.draw_idle()

    def _update_mission_artist(self, drone):
        path = drone['path']
        if path:
            xs = [pt[0] for pt in path]
            ys = [pt[1] for pt in path]
            drone['mission_line'].set_data(xs, ys)
            if 0 <= drone['current_target_index'] < len(path):
                tx, ty = path[drone['current_target_index']]
                drone['target_marker'].set_data([tx], [ty])
            else:
                drone['target_marker'].set_data([], [])
        else:
            drone['mission_line'].set_data([], [])
            drone['target_marker'].set_data([], [])

    def clear_selected_path(self, event=None):
        if not self.drones:
            return
        drone = self.drones[self.selected_drone_index]
        x_est, y_est, _ = drone['odometry'].mu
        drone['path'] = []
        drone['current_target_index'] = 0
        drone['planned_path'] = [(x_est, y_est)]
        drone['path_progress_index'] = 0
        drone['plan_goal_index'] = -1
        drone['last_plan_time'] = -1e9
        drone['plan_line'].set_data([x_est], [y_est])
        drone['subgoal_marker'].set_data([], [])
        self._update_mission_artist(drone)
        self.ui.refresh_shared_map()
        self.fig.canvas.draw_idle()

    def _save_trajectory_png(self, ts=None):
        self.last_saved_plot_path = save_trajectory_png(
            self.obstacles,
            self.drones,
            ts=ts,
        )
        return self.last_saved_plot_path

    def _save_map_outputs(self, ts=None):
        _, maps_path = save_map_outputs(
            self.shared_known_grid,
            self.drones,
            FREE,
            OCCUPIED,
            ts=ts,
        )
        self.last_saved_map_path = maps_path
        return maps_path

    def _save_outputs(self):
        self._save_trajectory_png()
        self._save_map_outputs()


    def _seed_for_drone(self, drone_index, stream=0):
        return int(self.current_seed * 1000 + 97 * (drone_index + 1) + stream)

    def _start_pose_for_index(self, drone_index, drone_count):
        base_x, base_y, base_angle = DRONE_START_POSE
        if drone_count <= 1:
            return base_x, base_y, base_angle

        offset_index = drone_index - 0.5 * (drone_count - 1)
        offset = offset_index * DRONE_START_SPACING
        angle_rad = math.radians(base_angle)
        perp_x = -math.sin(angle_rad)
        perp_y = math.cos(angle_rad)
        x = float(np.clip(base_x + offset * perp_x, 1.0, WORLD_WIDTH_METERS - 1.0))
        y = float(np.clip(base_y + offset * perp_y, 1.0, WORLD_HEIGHT_METERS - 1.0))
        return x, y, base_angle

    def _load_environment(self, seed):
        self.current_seed = int(seed)
        obstacles, landmark_dicts = generate_environment(self.current_seed)
        self.obstacles = list(obstacles)
        self.landmarks = [Landmark(**lm) for lm in landmark_dicts]
        self.generated_target_sequences = empty_target_sequences()
        self.truth_occupancy = self._build_truth_occupancy_grid()
        self.ax.set_title(
            f'Multi-Drone Search with Shared Observation Map + A* Routing  |  seed={self.current_seed}',
            pad=12,
        )
        self._redraw_environment_artists()

    def _redraw_environment_artists(self):
        for patch in self.obstacle_patches:
            try:
                patch.remove()
            except ValueError:
                pass
        self.obstacle_patches = []
        for patch in self.landmark_patches:
            try:
                patch.remove()
            except ValueError:
                pass
        self.landmark_patches = []

        for obs in self.obstacles:
            half = obs['size'] / 2.0
            patch = patches.Rectangle(
                (obs['x'] - half, obs['y'] - half),
                obs['size'],
                obs['size'],
                facecolor=COLOR_OBSTACLE,
                edgecolor='black',
                linewidth=1.7,
                alpha=0.88,
                zorder=3,
            )
            self.ax.add_patch(patch)
            self.obstacle_patches.append(patch)

        for lm in self.landmarks:
            patch = lm.draw(self.ax)
            if patch is not None:
                self.landmark_patches.append(patch)

    def _build_truth_occupancy_grid(self):
        return self.planner.build_truth_occupancy_grid(self.obstacles)

    def _init_known_grid(self):
        return self.planner.init_known_grid(OCCUPIED)

    def _offsets_for_margin(self, margin_m):
        return self.planner.offsets_for_margin(margin_m)

    def _stamp_obstacle_hit(self, known_grid, gx, gy):
        self.planner.stamp_obstacle_hit(known_grid, gx, gy, OCCUPIED)

    def _planning_occupancy(self, known_grid):
        return self.planner.planning_occupancy(known_grid, OCCUPIED)

    def _known_grid_for_planning(self, drone):
        mode = str(PLANNING_MAP_MODE).strip().lower()
        if mode == 'local':
            return drone['local_known_grid']
        if mode == 'fused':
            return np.maximum(self.shared_known_grid, drone['local_known_grid'])
        return self.shared_known_grid

    def _world_to_grid(self, x, y):
        return self.planner.world_to_grid(x, y)


    def _astar(self, start_xy, goal_xy, known_grid):
        return self.planner.astar(start_xy, goal_xy, known_grid, OCCUPIED)

    def _line_crosses_blocked(self, p0, p1, planning_occ=None, sample_step=None):
        planning_occ = self.truth_occupancy if planning_occ is None else planning_occ
        return self.planner.line_crosses_blocked(p0, p1, planning_occ, sample_step=sample_step)


    def _clear_local_grid_patches(self, drone):
        for patch in drone.get('local_grid_patches', []):
            try:
                patch.remove()
            except ValueError:
                pass
        drone['local_grid_patches'] = []

    def _should_draw_local_grid(self, drone):
        if not SHOW_ASTAR_LOCAL_GRID:
            return False
        if drone.get('drone_index', 0) >= max(0, int(A_STAR_GRID_MAX_DRONES)):
            return False
        stride = max(1, int(A_STAR_GRID_UPDATE_FRAMES))
        return (self.trace_counter % stride) == 0

    def _update_local_grid_visual(self, drone):
        if not SHOW_ASTAR_LOCAL_GRID:
            self._clear_local_grid_patches(drone)
            return
        if drone.get('drone_index', 0) >= max(0, int(A_STAR_GRID_MAX_DRONES)):
            self._clear_local_grid_patches(drone)
            return
        if not self._should_draw_local_grid(drone):
            return

        self._clear_local_grid_patches(drone)

        x_est, y_est, _ = drone['odometry'].mu
        cx, cy = self._world_to_grid(x_est, y_est)
        window = max(1, int(A_STAR_GRID_WINDOW_CELLS))
        x0 = max(0, cx - window)
        x1 = min(self.nx - 1, cx + window)
        y0 = max(0, cy - window)
        y1 = min(self.ny - 1, cy + window)

        known = self._known_grid_for_planning(drone)
        planning_occ = self._planning_occupancy(known)
        path_cells = {self._world_to_grid(px, py) for px, py in drone.get('planned_path', [])}
        color = drone['color']

        for gy in range(y0, y1 + 1):
            for gx in range(x0, x1 + 1):
                # Skip most empty cells to keep the overlay lightweight.
                cell_is_path = (gx, gy) in path_cells
                cell_is_occ = planning_occ[gy, gx]
                if not cell_is_path and not cell_is_occ and known[gy, gx] != UNKNOWN:
                    continue

                xw = gx * self.grid_resolution
                yw = gy * self.grid_resolution
                facecolor = 'none'
                edgecolor = (*color[:3], A_STAR_GRID_LINE_ALPHA * 0.6)
                linewidth = 0.45
                linestyle = ':' if known[gy, gx] == UNKNOWN else '-'

                if cell_is_occ:
                    facecolor = (0.90, 0.25, 0.25, A_STAR_GRID_OCC_ALPHA)
                    edgecolor = (0.55, 0.10, 0.10, min(0.7, A_STAR_GRID_OCC_ALPHA + 0.15))
                    linestyle = '-'
                elif cell_is_path:
                    facecolor = (*color[:3], A_STAR_GRID_PATH_ALPHA)
                    edgecolor = (*color[:3], min(0.8, A_STAR_GRID_PATH_ALPHA + 0.20))
                    linewidth = 1.0
                    linestyle = '-'

                patch = patches.Rectangle(
                    (xw, yw),
                    self.grid_resolution,
                    self.grid_resolution,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    linewidth=linewidth,
                    linestyle=linestyle,
                    zorder=4.8,
                )
                self.ax.add_patch(patch)
                drone['local_grid_patches'].append(patch)

        robot_patch = patches.Rectangle(
            (cx * self.grid_resolution, cy * self.grid_resolution),
            self.grid_resolution,
            self.grid_resolution,
            facecolor='none',
            edgecolor='white',
            linewidth=1.2,
            zorder=5.1,
        )
        self.ax.add_patch(robot_patch)
        drone['local_grid_patches'].append(robot_patch)

    def _create_drone(self, name, color, path, drone_index=0, drone_count=1):
        start_x, start_y, start_angle = self._start_pose_for_index(drone_index, drone_count)
        robot_rng = np.random.default_rng(self._seed_for_drone(drone_index, stream=1))
        robot = Robot(x=start_x, y=start_y, angle=start_angle, rng=robot_rng)
        odometry = OdometryEstimator(
            init_pos=[start_x, start_y, start_angle],
            init_cov=np.diag([0.1, 0.1, 1.0]),
        )

        true_patch = patches.Polygon(
            self.robot_shape_from_pose(start_x, start_y, start_angle, robot.size),
            facecolor=color,
            edgecolor='black',
            linewidth=1.2,
            zorder=10,
        )
        self.ax.add_patch(true_patch)

        est_patch = patches.Polygon(
            self.robot_shape_from_pose(start_x, start_y, start_angle, robot.size * 0.92),
            facecolor='none',
            edgecolor=color,
            linewidth=2.0,
            linestyle='--',
            zorder=11,
        )
        self.ax.add_patch(est_patch)

        ellipse_patch = odometry.draw_uncertainty_ellipse(
            self.ax,
            n_std=2.5,
            edgecolor=color,
            facecolor='none',
            linewidth=1.4,
            linestyle=':',
            alpha=0.9,
        )

        fov_patch = self._make_fov_patch(start_x, start_y, start_angle, color)
        self.ax.add_patch(fov_patch)

        trace_line, = self.ax.plot([start_x], [start_y], color=color, linewidth=1.2, alpha=0.75, zorder=2)
        est_trace_line, = self.ax.plot([start_x], [start_y], color=color, linewidth=1.2, linestyle='--', alpha=0.55, zorder=2)
        mission_line, = self.ax.plot([pt[0] for pt in path], [pt[1] for pt in path], marker='x', linestyle=':', linewidth=1.3, alpha=0.45, color=color, zorder=4)
        plan_line, = self.ax.plot([start_x], [start_y], color=color, linewidth=PATH_LINE_WIDTH, linestyle='-', alpha=0.98, zorder=6)
        plan_line.set_path_effects([pe.Stroke(linewidth=PATH_LINE_WIDTH + 2.2, foreground='black', alpha=0.35), pe.Normal()])

        target_marker, = self.ax.plot([path[0][0]] if path else [], [path[0][1]] if path else [], marker='*', markersize=TARGET_MARKER_SIZE, color=color, mec='black', mew=1.1, zorder=8)
        subgoal_marker, = self.ax.plot([start_x] if path else [], [start_y] if path else [], marker='o', markersize=SUBGOAL_MARKER_SIZE, color=color, mec='white', mew=1.2, zorder=8)

        ray_lines = []
        for _ in range(VISION_RAY_COUNT if SHOW_VISION_RAYS else 0):
            line, = self.ax.plot([start_x, start_x], [start_y, start_y], color=color, alpha=RAY_ALPHA, linewidth=RAY_LINE_WIDTH, zorder=5)
            ray_lines.append(line)

        return {
            'name': name,
            'drone_index': drone_index,
            'rng': np.random.default_rng(self._seed_for_drone(drone_index, stream=2)),
            'color': color,
            'path': list(path),
            'current_target_index': 0,
            'robot': robot,
            'odometry': odometry,
            'true_patch': true_patch,
            'est_patch': est_patch,
            'fov_patch': fov_patch,
            'ellipse_patch': ellipse_patch,
            'trace_x': [start_x],
            'trace_y': [start_y],
            'est_trace_x': [start_x],
            'est_trace_y': [start_y],
            'trace_line': trace_line,
            'est_trace_line': est_trace_line,
            'mission_line': mission_line,
            'ray_lines': ray_lines,
            'local_grid_patches': [],
            'min_clearance': self.view_distance,
            'planned_path': [(start_x, start_y)],
            'plan_line': plan_line,
            'target_marker': target_marker,
            'subgoal_marker': subgoal_marker,
            'plan_goal_index': -1,
            'last_plan_time': -1e9,
            'path_progress_index': 0,
            'local_known_grid': self._init_known_grid(),
            'known_occ_count': 0,
            'local_known_occ_count': 0,
            'just_discovered_obstacle': False,
            'recent_positions': [(0.0, start_x, start_y)],
            'recovery_until': -1e9,
            'recovery_turn_sign': 1.0,
            'last_scan_bias': 1.0,
            'last_command_active': False,
            'stuck_events': 0,
        }

    def robot_shape_from_pose(self, x, y, angle_deg, size):
        angle_rad = math.radians(angle_deg)
        front = (x + size * math.cos(angle_rad), y + size * math.sin(angle_rad))
        left = (
            x + size * 0.6 * math.cos(angle_rad + 2.5),
            y + size * 0.6 * math.sin(angle_rad + 2.5),
        )
        right = (
            x + size * 0.6 * math.cos(angle_rad - 2.5),
            y + size * 0.6 * math.sin(angle_rad - 2.5),
        )
        return [front, left, right]

    def _make_fov_patch(self, x, y, angle_deg, color):
        return patches.Wedge(
            (x, y),
            r=self.view_distance,
            theta1=angle_deg - self.fov_angle / 2.0,
            theta2=angle_deg + self.fov_angle / 2.0,
            facecolor=color,
            edgecolor=color,
            linewidth=1.0,
            linestyle='-',
            alpha=0.10,
            zorder=1,
        )

    def _sync_uncertainty_patch(self, drone):
        drone['ellipse_patch'] = drone['odometry'].draw_uncertainty_ellipse(
            self.ax,
            n_std=2.5,
            ellipse=drone.get('ellipse_patch'),
            edgecolor=drone['color'],
            facecolor='none',
            linewidth=1.4,
            linestyle=':',
            alpha=0.9,
        )

    def _sync_fov_patch(self, drone, x, y, angle_deg):
        patch = drone.get('fov_patch')
        if patch is None:
            patch = self._make_fov_patch(x, y, angle_deg, drone['color'])
            self.ax.add_patch(patch)
            drone['fov_patch'] = patch
            return
        patch.center = (x, y)
        patch.theta1 = angle_deg - self.fov_angle / 2.0
        patch.theta2 = angle_deg + self.fov_angle / 2.0
        patch.r = self.view_distance
        patch.set_facecolor(drone['color'])
        patch.set_edgecolor(drone['color'])
        patch.set_alpha(0.10)

    def _reset_drone_state(self, drone, x, y, angle, *, path=None, reset_local_grid=False, discovered_flag=False):
        robot = drone['robot']
        robot.x = float(x)
        robot.y = float(y)
        robot.angle = float(angle)
        robot.left_motor_speed = 0.0
        robot.right_motor_speed = 0.0

        drone['odometry'].mu = np.array([x, y, angle], dtype=float)
        drone['odometry'].cov = np.diag([0.1, 0.1, 1.0])

        if path is not None:
            drone['path'] = list(path)
        drone['current_target_index'] = 0
        drone['min_clearance'] = self.view_distance
        drone['planned_path'] = [(x, y)]
        drone['plan_goal_index'] = -1
        drone['last_plan_time'] = -1e9
        drone['path_progress_index'] = 0
        drone['plan_line'].set_data([x], [y])

        if drone['path']:
            drone['subgoal_marker'].set_data([x], [y])
            drone['target_marker'].set_data([drone['path'][0][0]], [drone['path'][0][1]])
        else:
            drone['subgoal_marker'].set_data([], [])
            drone['target_marker'].set_data([], [])

        if reset_local_grid:
            drone['local_known_grid'] = self._init_known_grid()
        self._reveal_start_area(drone['local_known_grid'], x, y)
        self._reveal_start_area(self.shared_known_grid, x, y)
        drone['known_occ_count'] = int(np.count_nonzero(self.shared_known_grid == OCCUPIED))
        drone['local_known_occ_count'] = int(np.count_nonzero(drone['local_known_grid'] == OCCUPIED))
        drone['just_discovered_obstacle'] = bool(discovered_flag)

        drone['recent_positions'] = [(self.time_elapsed, x, y)]
        drone['recovery_until'] = -1e9
        drone['recovery_turn_sign'] = 1.0
        drone['last_scan_bias'] = 1.0
        drone['last_command_active'] = False
        drone['stuck_events'] = 0

        drone['trace_x'] = [x]
        drone['trace_y'] = [y]
        drone['est_trace_x'] = [x]
        drone['est_trace_y'] = [y]
        drone['trace_line'].set_data(drone['trace_x'], drone['trace_y'])
        drone['est_trace_line'].set_data(drone['est_trace_x'], drone['est_trace_y'])

        self._clear_local_grid_patches(drone)
        drone['true_patch'].set_xy(self.robot_shape_from_pose(x, y, angle, robot.size))
        drone['est_patch'].set_xy(self.robot_shape_from_pose(x, y, angle, robot.size * 0.92))
        self._sync_uncertainty_patch(drone)
        self._sync_fov_patch(drone, x, y, angle)
        for line in drone['ray_lines']:
            line.set_data([x, x], [y, y])
        self._update_mission_artist(drone)

    def _reveal_start_area(self, grid, x, y, radius_m=1.1):
        offsets = self._offsets_for_margin(radius_m)
        gx, gy = self._world_to_grid(x, y)
        for dx, dy in offsets:
            nx = gx + dx
            ny = gy + dy
            if 0 <= nx < self.nx and 0 <= ny < self.ny and grid[ny, nx] == UNKNOWN:
                grid[ny, nx] = FREE

    def apply_seed_from_box(self, event=None):
        try:
            seed = int(str(self.ui.seed_box.text).strip())
        except Exception:
            self.ui.seed_box.set_val(str(self.current_seed))
            return
        self._load_environment(seed)
        self.auto_mode = False
        self._refresh_toggle_button()
        self._refresh_control_text()
        for idx, drone in enumerate(self.drones):
            new_path = list(self.generated_target_sequences[idx]) if idx < len(self.generated_target_sequences) else []
            drone['path'] = new_path
        self.reset_simulation()
        self.ui.refresh_shared_map()

    def toggle_auto_mode(self, event=None):
        self.auto_mode = not self.auto_mode
        self._refresh_toggle_button()
        self._refresh_control_text()
        if not self.auto_mode:
            for drone in self.drones:
                drone['robot'].set_motor_speeds(0.0, 0.0)

    def reset_simulation(self, event=None):
        self.auto_mode = False
        self._refresh_toggle_button()
        self._refresh_control_text()
        self.trace_counter = 0
        self.time_elapsed = 0.0
        self.ui.seed_box.set_val(str(self.current_seed))
        self.shared_known_grid = self._init_known_grid()

        for idx, drone in enumerate(self.drones):
            start_x, start_y, start_angle = self._start_pose_for_index(idx, len(self.drones))
            path = list(self.generated_target_sequences[idx]) if idx < len(self.generated_target_sequences) else []
            drone['robot'].rng = np.random.default_rng(self._seed_for_drone(idx, stream=1))
            drone['rng'] = np.random.default_rng(self._seed_for_drone(idx, stream=2))
            self._reset_drone_state(
                drone,
                start_x,
                start_y,
                start_angle,
                path=path,
                reset_local_grid=True,
                discovered_flag=False,
            )

        self.ui.refresh_shared_map()

    def _apply_scan_to_grid(self, grid, pose, scan):
        apply_scan_to_grid(
            grid, pose, scan, self._world_to_grid, self._stamp_obstacle_hit, self.grid_resolution, self.view_distance, UNKNOWN, FREE
        )

    def _update_known_map_from_scan(self, drone, scan):
        update_known_map_from_scan(
            drone, self.shared_known_grid, scan, self._apply_scan_to_grid, OCCUPIED
        )

    def _clearance_groups(self, scan):
        return clearance_groups(scan)

    def _maybe_replan(self, drone):
        path = drone['path']
        if not path:
            drone['planned_path'] = []
            return
        target_idx = drone['current_target_index']
        if target_idx >= len(path):
            drone['planned_path'] = []
            return

        x_est, y_est, _ = drone['odometry'].mu
        goal = path[target_idx]
        known_grid = self._known_grid_for_planning(drone)
        planning_occ = self._planning_occupancy(known_grid)
        need_replan = False
        if drone['plan_goal_index'] != target_idx:
            need_replan = True
        elif (self.time_elapsed - drone['last_plan_time']) >= A_STAR_REPLAN_SECONDS:
            need_replan = True
        elif not drone['planned_path']:
            need_replan = True
        elif KNOWN_MAP_REPLAN_ON_NEW_OBS and drone['just_discovered_obstacle']:
            need_replan = True
        elif drone['path_progress_index'] < len(drone['planned_path']):
            next_pt = drone['planned_path'][drone['path_progress_index']]
            if self._line_crosses_blocked((x_est, y_est), next_pt, planning_occ):
                need_replan = True

        if need_replan:
            old_path = list(drone['planned_path'])
            new_path = self._astar((x_est, y_est), goal, known_grid)
            drone['plan_goal_index'] = target_idx
            drone['last_plan_time'] = self.time_elapsed

            if new_path:
                drone['planned_path'] = new_path
                drone['path_progress_index'] = 1 if len(new_path) > 1 else 0
            elif old_path:
                drone['planned_path'] = old_path
                drone['path_progress_index'] = min(drone['path_progress_index'], max(0, len(old_path) - 1))
            else:
                drone['planned_path'] = []
                drone['path_progress_index'] = 0

            xs = [p[0] for p in drone['planned_path']]
            ys = [p[1] for p in drone['planned_path']]
            drone['plan_line'].set_data(xs, ys)

    def _next_subgoal(self, drone):
        path_pts = drone['planned_path']
        if not path_pts:
            return None
        x_est, y_est, _ = drone['odometry'].mu
        prog = drone['path_progress_index']
        planning_occ = self._planning_occupancy(self._known_grid_for_planning(drone))

        while prog < len(path_pts) - 1 and math.hypot(path_pts[prog][0] - x_est, path_pts[prog][1] - y_est) < A_STAR_GOAL_TOLERANCE:
            prog += 1

        furthest_visible = prog
        max_idx = min(len(path_pts) - 1, prog + max(1, A_STAR_LOOKAHEAD_STEPS))
        for j in range(prog, max_idx + 1):
            if self._line_crosses_blocked((x_est, y_est), path_pts[j], planning_occ):
                break
            furthest_visible = j

        drone['path_progress_index'] = prog
        return path_pts[furthest_visible]

    def _update_progress_history(self, drone):
        robot = drone['robot']
        hist = drone['recent_positions']
        hist.append((self.time_elapsed, robot.x, robot.y))
        cutoff = self.time_elapsed - STUCK_WINDOW_SECONDS
        while len(hist) > 2 and hist[1][0] < cutoff:
            hist.pop(0)

    def _should_trigger_recovery(self, drone):
        if self.time_elapsed < drone['recovery_until']:
            return False
        if not drone['last_command_active']:
            return False
        if drone['min_clearance'] > max(1.0, 0.95 * OBSTACLE_AVOID_DISTANCE):
            return False
        hist = drone['recent_positions']
        if len(hist) < 2:
            return False
        t0, x0, y0 = hist[0]
        t1, x1, y1 = hist[-1]
        if (t1 - t0) < 0.85 * STUCK_WINDOW_SECONDS:
            return False
        moved = math.hypot(x1 - x0, y1 - y0)
        return moved < STUCK_PROGRESS_EPS

    def _start_recovery(self, drone):
        drone['recovery_until'] = self.time_elapsed + STUCK_RECOVERY_SECONDS
        drone['recovery_turn_sign'] = drone['last_scan_bias'] if abs(drone['last_scan_bias']) > 1e-6 else 1.0
        drone['stuck_events'] += 1
        drone['plan_goal_index'] = -1
        drone['last_plan_time'] = -1e9
        drone['just_discovered_obstacle'] = True

    def _compute_control(self, drone):
        robot = drone['robot']
        odometry = drone['odometry']
        path = drone['path']
        target_idx = drone['current_target_index']
        L = robot.motor_distance

        scan = robot.scan_obstacles(self.obstacles, self.fov_angle, self.view_distance, ray_count=VISION_RAY_COUNT)
        self._update_known_map_from_scan(drone, scan)
        left, center, right = self._clearance_groups(scan)
        min_center = float(np.min(center)) if len(center) else self.view_distance
        mean_left = float(np.mean(left)) if len(left) else self.view_distance
        mean_right = float(np.mean(right)) if len(right) else self.view_distance
        drone['min_clearance'] = min_center
        drone['last_scan_bias'] = 1.0 if mean_left >= mean_right else -1.0

        if drone['ray_lines']:
            for ray_info, line in zip(scan, drone['ray_lines']):
                line.set_data([robot.x, ray_info['hit_x']], [robot.y, ray_info['hit_y']])

        if not path or target_idx >= len(path):
            drone['plan_line'].set_data([], [])
            drone['subgoal_marker'].set_data([], [])
            drone['last_command_active'] = False
            return 0.0, 0.0

        if self.time_elapsed < drone['recovery_until']:
            turn_sign = drone['recovery_turn_sign']
            drone['last_command_active'] = True
            return -STUCK_REVERSE_SPEED - turn_sign * 0.05, -STUCK_REVERSE_SPEED + turn_sign * 0.05

        self._maybe_replan(drone)
        subgoal = self._next_subgoal(drone)
        if subgoal is None:
            drone['subgoal_marker'].set_data([], [])
            drone['last_command_active'] = False
            return 0.0, 0.0

        x_est, y_est, theta_est = odometry.mu
        goal_x, goal_y = path[target_idx]
        sg_x, sg_y = subgoal
        dist_goal = math.hypot(goal_x - x_est, goal_y - y_est)
        dist_sub = math.hypot(sg_x - x_est, sg_y - y_est)

        if dist_goal < A_STAR_GOAL_TOLERANCE:
            drone['current_target_index'] += 1
            if drone['current_target_index'] >= len(path):
                drone['target_marker'].set_data([], [])
                drone['subgoal_marker'].set_data([], [])
                drone['planned_path'] = []
                drone['plan_line'].set_data([], [])
            drone['plan_goal_index'] = -1
            drone['last_command_active'] = False
            self._update_mission_artist(drone)
            return 0.0, 0.0

        target_angle = math.degrees(math.atan2(sg_y - y_est, sg_x - x_est)) % 360.0
        angle_diff = (target_angle - theta_est + 540.0) % 360.0 - 180.0

        heading_scale = max(0.35, 1.0 - min(abs(angle_diff), 120.0) / 140.0)
        v = (1.0 if dist_sub > 0.35 else 0.55) * heading_scale
        omega = np.clip(angle_diff / 4.2, -3.2, 3.2)

        if min_center < 0.9:
            v *= 0.15
            turn_sign = 1.0 if mean_left >= mean_right else -1.0
            omega = turn_sign * max(abs(omega), OBSTACLE_TURN_GAIN)
        elif min_center < OBSTACLE_AVOID_DISTANCE:
            v *= max(0.35, min_center / max(OBSTACLE_AVOID_DISTANCE, 1e-6))

        drone['last_command_active'] = abs(v) > 0.05 or abs(omega) > 0.1
        drone['subgoal_marker'].set_data([sg_x], [sg_y])
        drone['target_marker'].set_data([goal_x], [goal_y])

        vl = v - omega * L / 2.0
        vr = v + omega * L / 2.0
        return vl, vr

    def _update_drone(self, drone, dt):
        robot = drone['robot']
        odometry = drone['odometry']

        if dt > 0.0 and self.auto_mode:
            vl, vr = self._compute_control(drone)
            robot.set_motor_speeds(vl, vr)
        else:
            robot.set_motor_speeds(0.0, 0.0)

        if dt > 0.0:
            robot.update(dt, obstacles=self.obstacles)
            self._update_progress_history(drone)
            if self.auto_mode and self._should_trigger_recovery(drone):
                self._start_recovery(drone)

            L = robot.motor_distance
            linear_velocity = (robot.left_motor_speed + robot.right_motor_speed) / 2.0
            angular_velocity = (robot.right_motor_speed - robot.left_motor_speed) / L

            odometry.predict(
                [linear_velocity, angular_velocity],
                dt,
                np.diag([
                    PREDICTION_NOISE[0] ** 2,
                    PREDICTION_NOISE[1] ** 2,
                    PREDICTION_NOISE[2] ** 2,
                ]),
            )

        detected = robot.detect_landmarks(self.landmarks, self.fov_angle, self.view_distance, obstacles=self.obstacles)
        measurements = []
        for lm in detected:
            dx = lm.x - robot.x
            dy = lm.y - robot.y
            r = np.hypot(dx, dy) + drone['rng'].normal(0.0, MEASUREMENT_NOISE[0])

            true_bearing = (np.degrees(np.arctan2(dy, dx)) - robot.angle) % 360.0
            if true_bearing > 180.0:
                true_bearing -= 360.0
            b = true_bearing + drone['rng'].normal(0.0, MEASUREMENT_NOISE[1])
            measurements.append((r, b, lm.x, lm.y))

        if dt > 0.0 and measurements:
            odometry.correct(
                measurements,
                np.diag([MEASUREMENT_NOISE[0] ** 2, MEASUREMENT_NOISE[1] ** 2]),
                alpha=MEASUREMENT_ALPHA,
            )

        est_x, est_y, est_theta = odometry.mu
        drone['true_patch'].set_xy(self.robot_shape_from_pose(robot.x, robot.y, robot.angle, robot.size))
        drone['est_patch'].set_xy(self.robot_shape_from_pose(est_x, est_y, est_theta, robot.size * 0.92))

        self._sync_uncertainty_patch(drone)
        self._sync_fov_patch(drone, robot.x, robot.y, robot.angle)
        self._update_local_grid_visual(drone)

        return detected

    def update(self, frame):
        dt = TIME_STEP if self.auto_mode else 0.0
        if dt > 0.0:
            self.time_elapsed += dt

        artists = []

        for drone in self.drones:
            self._update_drone(drone, dt)
            robot = drone['robot']
            est_x, est_y, _ = drone['odometry'].mu

            if dt > 0.0 and self.trace_counter % self.trace_interval == 0:
                drone['trace_x'].append(robot.x)
                drone['trace_y'].append(robot.y)
                drone['est_trace_x'].append(est_x)
                drone['est_trace_y'].append(est_y)
                drone['trace_line'].set_data(drone['trace_x'], drone['trace_y'])
                drone['est_trace_line'].set_data(drone['est_trace_x'], drone['est_trace_y'])

            artists.extend([
                drone['true_patch'],
                drone['est_patch'],
                drone['fov_patch'],
                drone['ellipse_patch'],
                drone['trace_line'],
                drone['est_trace_line'],
                drone['mission_line'],
                drone['plan_line'],
                drone['target_marker'],
                drone['subgoal_marker'],
            ])
            artists.extend(drone['ray_lines'])

        if dt > 0.0:
            self.trace_counter += 1
        self.ui.refresh_status_text()
        self.ui.refresh_shared_map()
        artists.append(self.ui.status_text)
        if self.ui.shared_map_image is not None:
            artists.append(self.ui.shared_map_image)
        if self.ui.shared_robot_scatter is not None:
            artists.append(self.ui.shared_robot_scatter)
        return artists

    def run(self):
        self.ani = FuncAnimation(
            self.fig,
            self.update,
            interval=max(1, int(1000 / RENDER_FPS)),
            blit=False,
            cache_frame_data=False,
        )
        plt.show()
        self._save_outputs()

if __name__ == '__main__':
    Simulator().run()

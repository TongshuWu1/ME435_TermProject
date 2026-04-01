from datetime import datetime
import json
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.animation import FuncAnimation

from config import (
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
    A_STAR_MIN_REPLAN_GAP_SECONDS,
    AUTO_EXPLORE_REPLAN_SECONDS,
    AUTO_GOAL_MIN_HOLD_SECONDS,
    AUTO_GOAL_STALE_SECONDS,
    AUTO_GOAL_MIN_SWITCH_DISTANCE,
    AUTO_FRONTIER_INFO_GAIN,
    AUTO_FRONTIER_MIN_COMPONENT_CELLS,
    AUTO_FRONTIER_PARTITION_PENALTY,
    AUTO_FRONTIER_PROGRESS_WEIGHT,
    AUTO_FRONTIER_MIN_GOAL_DISTANCE,
    AUTO_FRONTIER_TEAMMATE_PENALTY,
    AUTO_FRONTIER_TEAMMATE_RADIUS,
    AUTO_FRONTIER_TOP_K_CANDIDATES,
    AUTO_LAUNCH_DISPERSAL_DISTANCE,
    AUTO_LAUNCH_DISPERSAL_ENABLED,
    AUTO_LAUNCH_DISPERSAL_GOAL_TOLERANCE,
    AUTO_LAUNCH_DISPERSAL_MAX_SECONDS,
    AUTO_STOP_WHEN_FINISHED,
    AUTO_FINISH_HOLD_SECONDS,
    AUTO_DENSITY_FRONTIER_WEIGHT,
    AUTO_DENSITY_UNKNOWN_WEIGHT,
    AUTO_DENSITY_FREE_WEIGHT,
    AUTO_DENSITY_SMOOTHING_PASSES,
    AUTO_FRONTIER_CENTROID_WEIGHT,
    AUTO_FRONTIER_DENSITY_VALUE_WEIGHT,
    AUTO_COVERAGE_CENTROID_PULL,
    AUTO_COVERAGE_DENSITY_GAIN,
    AUTO_COVERAGE_FALLBACK_GLOBAL,
    AUTO_COVERAGE_FRONTIER_BONUS,
    AUTO_COVERAGE_FRONTIER_CLUSTER_GAIN,
    AUTO_COVERAGE_FRONTIER_CONTACT_GAIN,
    AUTO_COVERAGE_FRONTIER_PROXIMITY_GAIN,
    AUTO_COVERAGE_GLOBAL_FALLBACK_PENALTY,
    AUTO_COVERAGE_LOCAL_MASS_GAIN,
    AUTO_COVERAGE_NONMAX_RADIUS_CELLS,
    AUTO_COVERAGE_PROGRESS_WEIGHT,
    AUTO_COVERAGE_ROBOT_DISTANCE_WEIGHT,
    AUTO_COVERAGE_TEAMMATE_PENALTY,
    AUTO_COVERAGE_TEAMMATE_RADIUS,
    AUTO_COVERAGE_TOP_K_CANDIDATES,
    DEFAULT_AUTO_POLICY,
    AUTO_GLOBAL_FRONTIER_FALLBACK,
    AUTO_PARTITION_EPSILON_METERS,
    COLOR_BOUNDARY,
    COLOR_OBSTACLE,
    DEFAULT_MISSION_MODE,
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
    PREDICTION_NOISE,
    RANDOM_SEED,
    RAY_ALPHA,
    RAY_LINE_WIDTH,
    SHOW_ASTAR_LOCAL_GRID,
    SHOW_DENSITY_OVERLAY_BY_DEFAULT,
    SHOW_PARTITION_OVERLAY_BY_DEFAULT,
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
from .auto_explore import partition_generators_from_positions
from .controllers import FrontierController, WeightedCoverageController
from .landmark import Landmark
from .mapping_utils import apply_scan_to_grid, clearance_groups, reveal_start_area, update_known_map_from_scan
from .output_utils import save_map_outputs, save_trajectory_png, write_run_metadata
from .reporting import compute_coverage_metrics, polyline_length, save_run_reports
from .paths import output_root, run_output_dir
from .planner import GridPlanner
from .sim.drone_factory import create_drone
from .sim.partition_state import compute_partition_state
from .sim.rendering import (
    create_uncertainty_ellipse,
    make_fov_patch,
    robot_shape_from_pose,
    update_fov_patch,
    update_uncertainty_ellipse_patch,
)
from .ui.simulator_ui import SimulatorUI

RENDER_FPS = 30
UNKNOWN = 0
FREE = 1
OCCUPIED = 2

class Simulator:
    def __init__(self):
        self.current_seed = RANDOM_SEED
        self.run_index = 0
        self.current_run_label = ''
        self.current_run_dir = None
        self._saved_current_run = False
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
        self.auto_finished = False
        self.auto_finish_candidate_time = None

        self.grid_resolution = A_STAR_GRID_RESOLUTION
        self.planner = GridPlanner(A_STAR_GRID_RESOLUTION, A_STAR_INFLATION_MARGIN)
        self.nx = self.planner.nx
        self.ny = self.planner.ny
        self.truth_occupancy = self._build_truth_occupancy_grid()
        self.shared_known_grid = self._init_known_grid()
        self.mission_mode = DEFAULT_MISSION_MODE
        self.auto_policy = DEFAULT_AUTO_POLICY
        self.show_partition_overlay = SHOW_PARTITION_OVERLAY_BY_DEFAULT
        self.show_density_overlay = SHOW_DENSITY_OVERLAY_BY_DEFAULT
        self.partition_labels = -np.ones((self.ny, self.nx), dtype=int)
        self.partition_rgba = np.zeros((self.ny, self.nx, 4), dtype=float)
        self.density_map = np.zeros((self.ny, self.nx), dtype=float)
        self.density_rgba = np.zeros((self.ny, self.nx, 4), dtype=float)
        self.partition_generators_xy = np.zeros((0, 2), dtype=float)
        self.partition_centroids_xy = np.zeros((0, 2), dtype=float)
        self.partition_generator_colors = []
        self.frontier_components = []
        self._partition_dirty = True
        self._last_partition_generators = None
        self.auto_finished = False
        self.auto_finish_candidate_time = None

        self.selected_drone_index = 0
        self.edit_mode = 'add_waypoint'
        self.last_saved_plot_path = ''
        self.last_saved_map_path = ''
        self.event_log = []
        self.coverage_history = []
        self._last_coverage_sample_time = -1e9
        self._last_console_snapshot_time = -1e9
        self._event_seq = 0
        self._live_metrics = None

        self.frontier_controller = FrontierController(
            fallback_global=AUTO_GLOBAL_FRONTIER_FALLBACK,
            top_k_candidates=AUTO_FRONTIER_TOP_K_CANDIDATES,
            info_gain=AUTO_FRONTIER_INFO_GAIN,
            partition_penalty=AUTO_FRONTIER_PARTITION_PENALTY,
            teammate_radius=AUTO_FRONTIER_TEAMMATE_RADIUS,
            teammate_penalty=AUTO_FRONTIER_TEAMMATE_PENALTY,
            progress_weight=AUTO_FRONTIER_PROGRESS_WEIGHT,
            min_goal_distance=max(AUTO_FRONTIER_MIN_GOAL_DISTANCE, 1.05 * A_STAR_GOAL_TOLERANCE),
            centroid_weight=AUTO_FRONTIER_CENTROID_WEIGHT,
            density_value_weight=AUTO_FRONTIER_DENSITY_VALUE_WEIGHT,
        )
        self.coverage_controller = WeightedCoverageController(
            fallback_global=AUTO_COVERAGE_FALLBACK_GLOBAL,
            top_k_candidates=AUTO_COVERAGE_TOP_K_CANDIDATES,
            density_gain=AUTO_COVERAGE_DENSITY_GAIN,
            centroid_pull=AUTO_COVERAGE_CENTROID_PULL,
            robot_distance_weight=AUTO_COVERAGE_ROBOT_DISTANCE_WEIGHT,
            teammate_radius=AUTO_COVERAGE_TEAMMATE_RADIUS,
            teammate_penalty=AUTO_COVERAGE_TEAMMATE_PENALTY,
            progress_weight=AUTO_COVERAGE_PROGRESS_WEIGHT,
            frontier_bonus=AUTO_COVERAGE_FRONTIER_BONUS,
            min_goal_distance=max(AUTO_FRONTIER_MIN_GOAL_DISTANCE, 1.05 * A_STAR_GOAL_TOLERANCE),
            free_density_baseline=AUTO_DENSITY_FREE_WEIGHT,
            frontier_contact_gain=AUTO_COVERAGE_FRONTIER_CONTACT_GAIN,
            frontier_cluster_gain=AUTO_COVERAGE_FRONTIER_CLUSTER_GAIN,
            local_mass_gain=AUTO_COVERAGE_LOCAL_MASS_GAIN,
            frontier_proximity_gain=AUTO_COVERAGE_FRONTIER_PROXIMITY_GAIN,
            nonmax_radius_cells=AUTO_COVERAGE_NONMAX_RADIUS_CELLS,
            global_fallback_penalty=AUTO_COVERAGE_GLOBAL_FALLBACK_PENALTY,
        )

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

        if len(DRONE_NAMES) <= 10:
            self.colors = plt.cm.tab10(np.linspace(0, 1, len(DRONE_NAMES)))
        elif len(DRONE_NAMES) <= 20:
            self.colors = plt.cm.tab20(np.linspace(0, 1, len(DRONE_NAMES)))
        else:
            self.colors = plt.cm.hsv(np.linspace(0, 1, len(DRONE_NAMES), endpoint=False))
        self._load_environment(self.current_seed)
        self._start_new_run()

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
        self._sync_paths_for_mode()
        self._refresh_partition_state()
        self.ui.refresh_all()

    def _refresh_toggle_button(self):
        self.ui.refresh_toggle_button()

    def _refresh_partition_button(self):
        self.ui.refresh_partition_button()

    def _refresh_control_text(self):
        self.ui.refresh_status_text()

    def _mark_partition_dirty(self):
        self._partition_dirty = True

    def _reset_reporting_state(self):
        self.event_log = []
        self.coverage_history = []
        self._last_coverage_sample_time = -1e9
        self._last_console_snapshot_time = -1e9
        self._event_seq = 0
        self._live_metrics = None

    def _log_event(self, event_type, message, drone=None, **data):
        self._event_seq += 1
        robot_name = '' if drone is None else str(drone.get('name', ''))
        record = {
            'seq': int(self._event_seq),
            'time_seconds': round(float(self.time_elapsed), 3),
            'event_type': str(event_type),
            'robot_name': robot_name,
            'message': str(message),
            'data_json': json.dumps(data, sort_keys=True) if data else '',
        }
        self.event_log.append(record)
        prefix = f"[{self.current_run_label or 'run'} t={self.time_elapsed:6.2f}s]"
        if robot_name:
            prefix += f" [{robot_name}]"
        print(f"{prefix} {message}")

    def _compute_live_metrics(self):
        metrics = compute_coverage_metrics(
            self.shared_known_grid,
            self.truth_occupancy,
            unknown_value=UNKNOWN,
            free_value=FREE,
            occupied_value=OCCUPIED,
        )
        metrics['frontier_count'] = int(len(getattr(self, 'frontier_components', [])))
        metrics['active_goals'] = int(sum(1 for drone in getattr(self, 'drones', []) if drone.get('auto_goal_xy') is not None))
        metrics['running'] = bool(self.auto_mode)
        metrics['auto_finished'] = bool(self.auto_finished)
        metrics['selected_robot'] = self.drones[self.selected_drone_index]['name'] if getattr(self, 'drones', None) else ''
        metrics['mission_mode'] = self.mission_mode
        metrics['auto_policy'] = self.auto_policy
        self._live_metrics = metrics
        return metrics

    def _record_coverage_snapshot(self, force=False):
        if not force and (self.time_elapsed - self._last_coverage_sample_time) < 0.5:
            return
        metrics = self._compute_live_metrics()
        row = {
            'time_seconds': round(float(self.time_elapsed), 3),
            'known_ratio': round(float(metrics['known_ratio']), 6),
            'free_coverage_ratio': round(float(metrics['free_coverage_ratio']), 6),
            'occupied_recall_ratio': round(float(metrics['occupied_recall_ratio']), 6),
            'known_cells': int(metrics['known_cells']),
            'covered_free_cells': int(metrics['covered_free_cells']),
            'mapped_occupied_cells': int(metrics['mapped_occupied_cells']),
            'frontier_count': int(metrics['frontier_count']),
            'active_goals': int(metrics['active_goals']),
            'running': int(bool(metrics['running'])),
            'auto_finished': int(bool(metrics['auto_finished'])),
            'selected_robot': metrics['selected_robot'],
            'auto_policy': metrics['auto_policy'],
            'mission_mode': metrics['mission_mode'],
        }
        if self.coverage_history and self.coverage_history[-1] == row:
            return
        self.coverage_history.append(row)
        self._last_coverage_sample_time = self.time_elapsed

    def _maybe_print_live_snapshot(self, force=False):
        if not force and (self.time_elapsed - self._last_console_snapshot_time) < 2.0:
            return
        metrics = self._compute_live_metrics()
        mission_name = 'auto explore' if self.mission_mode == 'auto_explore' else 'manual click'
        policy_name = 'weighted coverage' if self.auto_policy == 'weighted_coverage' else 'frontier'
        print(
            f"[{self.current_run_label or 'run'} | t={self.time_elapsed:6.2f}s] "
            f"mode={mission_name}, policy={policy_name}, "
            f"map_known={100.0 * metrics['known_ratio']:5.1f}%, "
            f"free_covered={100.0 * metrics['free_coverage_ratio']:5.1f}%, "
            f"obstacles_found={100.0 * metrics['occupied_recall_ratio']:5.1f}%, "
            f"frontier_groups={metrics['frontier_count']:d}, active_goals={metrics['active_goals']:d}"
        )
        self._last_console_snapshot_time = self.time_elapsed

    def _goal_type_for_drone(self, drone):
        if self.mission_mode == 'manual_click':
            return 'manual'
        meta = drone.get('auto_goal_meta') or {}
        if self.auto_policy == 'weighted_coverage':
            return str(meta.get('goal_flavor', 'coverage-fill'))
        if meta.get('used_fallback'):
            return 'frontier-fallback'
        return 'frontier'

    def build_status_text(self):
        metrics = self._live_metrics if self._live_metrics is not None else self._compute_live_metrics()
        drone = None
        if getattr(self, 'drones', None):
            drone = self.drones[self.selected_drone_index]
        sim_state = 'Finished' if (self.auto_finished and self.mission_mode == 'auto_explore') else ('Running' if self.auto_mode else 'Paused')
        mission_name = 'Auto Explore' if self.mission_mode == 'auto_explore' else 'Manual Click'
        policy_name = 'Weighted Coverage' if self.auto_policy == 'weighted_coverage' else 'Frontier'
        lines = [
            f'State   : {sim_state}',
            f'Mission : {mission_name}',
            f'Policy  : {policy_name}',
            f'Time    : {self.time_elapsed:5.1f} s    Seed: {self.current_seed}',
            f'Map     : {100.0 * metrics["known_ratio"]:5.1f}% known',
            f'Free    : {100.0 * metrics["free_coverage_ratio"]:5.1f}% covered',
            f'Obs     : {100.0 * metrics["occupied_recall_ratio"]:5.1f}% found',
            f'Frontier: {metrics["frontier_count"]} groups    Goals: {metrics["active_goals"]}',
        ]
        if drone is not None:
            est_x, est_y, _ = drone['odometry'].mu
            goal_xy = drone.get('auto_goal_xy')
            remaining_path = drone.get('planned_path', [])
            remaining_length = polyline_length(remaining_path[max(0, int(drone.get('path_progress_index', 0))):])
            goal_text = 'none' if goal_xy is None else f'({goal_xy[0]:4.1f}, {goal_xy[1]:4.1f})'
            last_landmark = '-' if drone.get('last_landmark_update_time') is None else f"{float(drone['last_landmark_update_time']):4.1f}s"
            lines.extend([
                '',
                f'Robot   : {drone["name"]}    Phase: {drone.get("auto_phase", "manual")}',
                f'Goal    : {self._goal_type_for_drone(drone)} -> {goal_text}',
                f'Est xy  : ({est_x:4.1f}, {est_y:4.1f})    Path: {remaining_length:4.1f} m',
                f'Goals   : {drone.get("goal_reached_count", 0)} reached / {drone.get("goal_assignment_count", 0)} assigned',
                f'Replans : {drone.get("replan_count", 0)}    Stuck: {drone.get("stuck_events", 0)}',
                f'Landmark: {drone.get("measurement_update_count", 0)} updates    Last: {last_landmark}',
            ])
        return '\n'.join(lines)

    def _sync_paths_for_mode(self):
        for drone in getattr(self, 'drones', []):
            if self.mission_mode == 'manual_click':
                drone['path'] = list(drone.get('manual_path', []))
                drone['current_target_index'] = min(drone.get('current_target_index', 0), len(drone['path']))
                drone['auto_goal_xy'] = None
                drone['auto_goal_last_set_time'] = -1e9
                drone['auto_phase'] = 'launch'
            else:
                drone['path'] = [] if drone.get('auto_goal_xy') is None else [tuple(drone['auto_goal_xy'])]
                drone['current_target_index'] = 0
            self._update_mission_artist(drone)
        self.auto_finished = False
        self.auto_finish_candidate_time = None
        self._mark_partition_dirty()

    def on_select_mission_mode(self, label):
        norm = str(label).strip().lower()
        self.mission_mode = 'auto_explore' if 'auto' in norm else 'manual_click'
        self._sync_paths_for_mode()
        self.auto_finished = False
        self.auto_finish_candidate_time = None
        self._log_event('mission_mode_changed', f"Mission mode set to {self.mission_mode.replace('_', ' ')}")
        self._refresh_control_text()
        self._refresh_partition_state(force=True)
        self.fig.canvas.draw_idle()

    def on_select_auto_policy(self, label):
        norm = str(label).strip().lower()
        self.auto_policy = 'weighted_coverage' if 'weight' in norm else 'frontier'
        for drone in getattr(self, 'drones', []):
            if self.mission_mode == 'auto_explore':
                drone['auto_goal_xy'] = None
                drone['auto_goal_last_set_time'] = -1e9
                drone['auto_goal_meta'] = None
                drone['path'] = []
                drone['planned_path'] = []
                drone['plan_goal_index'] = -1
                self._update_mission_artist(drone)
        self.auto_finished = False
        self.auto_finish_candidate_time = None
        self._log_event('auto_policy_changed', f"Auto policy set to {self.auto_policy.replace('_', ' ')}")
        self._refresh_control_text()
        self.fig.canvas.draw_idle()

    def on_select_robot(self, label):
        try:
            self.selected_drone_index = DRONE_NAMES.index(label)
        except ValueError:
            self.selected_drone_index = 0
        self._refresh_control_text()

    def on_select_edit_mode(self, label):
        norm = str(label).strip().lower()
        self.edit_mode = 'set_start' if 'start' in norm else 'add_waypoint'
        self._log_event('edit_mode_changed', f"Edit mode set to {self.edit_mode.replace('_', ' ')}")
        self._refresh_control_text()

    def toggle_partition_overlay(self, event=None):
        self.show_partition_overlay = not self.show_partition_overlay
        self._refresh_partition_button()
        self.ui.refresh_partition_overlay()
        self.fig.canvas.draw_idle()

    def toggle_density_overlay(self, event=None):
        self.show_density_overlay = not self.show_density_overlay
        self._refresh_density_button()
        self.ui.refresh_partition_overlay()
        self.fig.canvas.draw_idle()

    def _refresh_density_button(self):
        self.ui.refresh_density_button()

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
        robot = drone['robot']
        _, _, angle = drone['odometry'].mu
        robot.x = float(x)
        robot.y = float(y)
        robot.angle = float(angle)
        robot.left_motor_speed = 0.0
        robot.right_motor_speed = 0.0

        drone['odometry'].mu = np.array([x, y, angle], dtype=float)
        drone['odometry'].cov = np.diag([0.1, 0.1, 1.0])
        drone['current_target_index'] = 0
        drone['auto_goal_xy'] = None
        drone['auto_goal_last_set_time'] = -1e9
        drone['auto_goal_meta'] = None
        drone['auto_phase'] = 'launch'
        drone['planned_path'] = [(x, y)]
        drone['plan_goal_index'] = -1
        drone['last_plan_time'] = -1e9
        drone['path_progress_index'] = 0
        drone['plan_line'].set_data([x], [y])
        drone['subgoal_marker'].set_data([x] if drone['path'] else [], [y] if drone['path'] else [])
        self._reveal_start_area(drone['local_known_grid'], x, y)
        self._reveal_start_area(self.shared_known_grid, x, y)
        drone['trace_x'] = [x]
        drone['trace_y'] = [y]
        drone['est_trace_x'] = [x]
        drone['est_trace_y'] = [y]
        drone['trace_line'].set_data(drone['trace_x'], drone['trace_y'])
        drone['est_trace_line'].set_data(drone['est_trace_x'], drone['est_trace_y'])
        drone['recent_positions'] = [(self.time_elapsed, x, y)]
        drone['just_discovered_obstacle'] = True
        drone['last_command_active'] = False
        drone['distance_travelled'] = 0.0
        drone['idle_time'] = 0.0
        drone['visited_cells'] = {self._world_to_grid(x, y)}
        self._update_mission_artist(drone)
        drone['true_patch'].set_xy(self.robot_shape_from_pose(x, y, angle, robot.size))
        drone['est_patch'].set_xy(self.robot_shape_from_pose(x, y, angle, robot.size * 0.92))
        update_uncertainty_ellipse_patch(drone['ellipse_patch'], drone['odometry'], n_std=2.5)
        update_fov_patch(drone['fov_patch'], x, y, angle, view_distance=self.view_distance, fov_angle=self.fov_angle)
        for line in drone['ray_lines']:
            line.set_data([x, x], [y, y])
        self._saved_current_run = False
        self._mark_partition_dirty()
        self._refresh_partition_state(force=True)

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
            self._log_event('start_reset', f"Start pose moved to ({float(x):.2f}, {float(y):.2f})", drone=drone, x=float(x), y=float(y))
        else:
            if self.mission_mode != 'manual_click':
                return
            drone['manual_path'].append((float(x), float(y)))
            drone['path'] = list(drone['manual_path'])
            drone['last_goal_type'] = 'manual'
            if len(drone['path']) == 1:
                drone['current_target_index'] = 0
            self._update_mission_artist(drone)
            drone['plan_goal_index'] = -1
            drone['last_plan_time'] = -1e9
            drone['just_discovered_obstacle'] = True
            self._log_event('manual_waypoint_added', f"Manual waypoint added at ({float(x):.2f}, {float(y):.2f})", drone=drone, x=float(x), y=float(y), waypoint_count=len(drone['manual_path']))
        self._saved_current_run = False
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
        drone['manual_path'] = []
        drone['auto_goal_xy'] = None
        drone['auto_goal_last_set_time'] = -1e9
        drone['auto_goal_meta'] = None
        drone['auto_phase'] = 'launch'
        drone['path'] = []
        drone['current_target_index'] = 0
        drone['planned_path'] = [(x_est, y_est)]
        drone['path_progress_index'] = 0
        drone['plan_goal_index'] = -1
        drone['last_plan_time'] = -1e9
        drone['plan_line'].set_data([x_est], [y_est])
        drone['subgoal_marker'].set_data([], [])
        drone['last_goal_type'] = 'none'
        self._update_mission_artist(drone)
        self._saved_current_run = False
        self._mark_partition_dirty()
        self._refresh_partition_state(force=True)
        self.ui.refresh_shared_map()
        self._log_event('path_cleared', 'Cleared path and current goal', drone=drone)
        self.fig.canvas.draw_idle()

    def _save_outputs(self):
        self._finalize_current_run()

    def _run_metadata(self):
        return {
            'run_label': self.current_run_label,
            'seed': int(self.current_seed),
            'mission_mode': self.mission_mode,
            'auto_policy': self.auto_policy,
            'robot_count': len(getattr(self, 'drones', [])),
            'robot_names': [drone['name'] for drone in getattr(self, 'drones', [])],
            'sim_time_seconds': float(self.time_elapsed),
            'auto_mode_enabled_at_save': bool(self.auto_mode),
            'saved_at': datetime.now().isoformat(timespec='seconds'),
        }

    def _run_has_progress(self):
        if getattr(self, 'time_elapsed', 0.0) > 1e-9:
            return True
        for drone in getattr(self, 'drones', []):
            if len(drone.get('trace_x', [])) > 1 or len(drone.get('trace_y', [])) > 1:
                return True
        return False

    def _start_new_run(self):
        self.run_index += 1
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.current_run_label = f'run_{self.run_index:03d}_seed_{int(self.current_seed)}_{ts}'
        self.current_run_dir = output_root() / self.current_run_label
        self._saved_current_run = False
        self._reset_reporting_state()
        self._log_event('run_started', f'Started {self.current_run_label}', seed=int(self.current_seed), mission_mode=self.mission_mode, auto_policy=self.auto_policy)

    def _finalize_current_run(self):
        if self.current_run_dir is None or self._saved_current_run:
            return
        if not self._run_has_progress():
            return
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._record_coverage_snapshot(force=True)
        self.last_saved_plot_path = save_trajectory_png(self.obstacles, self.drones, self.current_run_dir, ts=ts)
        _, self.last_saved_map_path = save_map_outputs(self.shared_known_grid, self.drones, FREE, OCCUPIED, self.current_run_dir, ts=ts)
        metadata = self._run_metadata()
        write_run_metadata(self.current_run_dir, metadata)
        metrics = self._compute_live_metrics()
        robot_rows = []
        for drone in self.drones:
            est_x, est_y, _ = drone['odometry'].mu
            robot_rows.append({
                'name': drone['name'],
                'phase': drone.get('auto_phase', 'manual'),
                'goal_type': drone.get('last_goal_type', 'none'),
                'goal_assignments': int(drone.get('goal_assignment_count', 0)),
                'goals_reached': int(drone.get('goal_reached_count', 0)),
                'replans': int(drone.get('replan_count', 0)),
                'replans_blocked': int(drone.get('blocked_replan_count', 0)),
                'stuck_events': int(drone.get('stuck_events', 0)),
                'measurement_updates': int(drone.get('measurement_update_count', 0)),
                'last_landmark_update_time': '' if drone.get('last_landmark_update_time') is None else round(float(drone.get('last_landmark_update_time')), 3),
                'distance_travelled_m': round(float(drone.get('distance_travelled', 0.0)), 3),
                'idle_time_s': round(float(drone.get('idle_time', 0.0)), 3),
                'visited_cell_count': int(len(drone.get('visited_cells', []))),
                'path_waypoint_count': int(len(drone.get('manual_path', []))),
                'remaining_path_points': int(len(drone.get('planned_path', []))),
                'remaining_path_length_m': round(polyline_length(drone.get('planned_path', [])[max(0, int(drone.get('path_progress_index', 0))):]), 3),
                'final_true_x': round(float(drone['robot'].x), 3),
                'final_true_y': round(float(drone['robot'].y), 3),
                'final_est_x': round(float(est_x), 3),
                'final_est_y': round(float(est_y), 3),
            })
        summary = {
            'known_ratio': round(float(metrics['known_ratio']), 6),
            'free_coverage_ratio': round(float(metrics['free_coverage_ratio']), 6),
            'occupied_recall_ratio': round(float(metrics['occupied_recall_ratio']), 6),
            'frontier_count': int(metrics['frontier_count']),
            'active_goals': int(metrics['active_goals']),
            'event_count': int(len(self.event_log)),
            'coverage_samples': int(len(self.coverage_history)),
            'auto_finished': bool(self.auto_finished),
            'last_saved_plot_path': self.last_saved_plot_path,
            'last_saved_map_path': self.last_saved_map_path,
        }
        self._log_event('run_saved', f"Saved outputs to {self.current_run_dir}", output_dir=str(self.current_run_dir), known_ratio=summary['known_ratio'], free_coverage_ratio=summary['free_coverage_ratio'])
        save_run_reports(
            out_dir=self.current_run_dir,
            metadata=metadata,
            coverage_history=self.coverage_history,
            event_log=self.event_log,
            robot_rows=robot_rows,
            summary=summary,
            ts=ts,
        )
        self._saved_current_run = True

    def _save_trajectory_png(self, ts=None):
        out_dir = self.current_run_dir or run_output_dir('run_manual')
        self.last_saved_plot_path = save_trajectory_png(
            self.obstacles,
            self.drones,
            out_dir,
            ts=ts,
        )
        return self.last_saved_plot_path

    def _save_map_outputs(self, ts=None):
        out_dir = self.current_run_dir or run_output_dir('run_manual')
        _, maps_path = save_map_outputs(
            self.shared_known_grid,
            self.drones,
            FREE,
            OCCUPIED,
            out_dir,
            ts=ts,
        )
        self.last_saved_map_path = maps_path
        return maps_path

    def _prepare_for_new_run(self):
        self._finalize_current_run()
        self._start_new_run()

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
        self._mark_partition_dirty()
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

    def _world_to_grid(self, x, y):
        return self.planner.world_to_grid(x, y)

    def _grid_to_world(self, cell):
        return self.planner.grid_to_world(cell)

    def _generator_positions(self):
        est_positions = []
        for drone in self.drones:
            x_est, y_est, _ = drone['odometry'].mu
            est_positions.append((x_est, y_est))
        return partition_generators_from_positions(est_positions, DRONE_START_POSE, epsilon_m=AUTO_PARTITION_EPSILON_METERS)

    def _refresh_partition_state(self, force=False):
        if not getattr(self, 'drones', None):
            return
        generators = np.asarray(self._generator_positions(), dtype=float)
        need_refresh = bool(force or self._partition_dirty)
        if not need_refresh:
            if self._last_partition_generators is None:
                need_refresh = True
            elif generators.shape != self._last_partition_generators.shape:
                need_refresh = True
            elif generators.size and np.max(np.linalg.norm(generators - self._last_partition_generators, axis=1)) > 0.10:
                need_refresh = True
        if not need_refresh:
            return

        state = compute_partition_state(
            self.shared_known_grid,
            generators,
            [drone['color'] for drone in self.drones],
            self.grid_resolution,
            UNKNOWN,
            FREE,
            OCCUPIED,
            frontier_weight=AUTO_DENSITY_FRONTIER_WEIGHT,
            unknown_weight=AUTO_DENSITY_UNKNOWN_WEIGHT,
            free_weight=AUTO_DENSITY_FREE_WEIGHT,
            smoothing_passes=AUTO_DENSITY_SMOOTHING_PASSES,
            min_frontier_cells=AUTO_FRONTIER_MIN_COMPONENT_CELLS,
        )
        self.partition_labels = state.labels
        self.density_map = state.density_map
        self.partition_centroids_xy = state.centroids_xy
        self.partition_generators_xy = state.generators_xy
        self.partition_generator_colors = state.generator_colors
        self.partition_rgba = state.partition_rgba
        self.density_rgba = state.density_rgba
        self.frontier_components = state.frontier_components
        self._last_partition_generators = generators.copy()
        self._partition_dirty = False

    def _assign_auto_goal(self, drone, goal_xy, meta=None):
        x_est, y_est, _ = drone['odometry'].mu
        prev_goal = drone.get('auto_goal_xy')
        prev_goal_type = drone.get('last_goal_type', 'none')
        drone['auto_goal_xy'] = None if goal_xy is None else tuple(goal_xy)
        drone['auto_goal_last_set_time'] = self.time_elapsed
        drone['current_target_index'] = 0
        drone['path'] = [] if goal_xy is None else [tuple(goal_xy)]
        drone['plan_goal_index'] = -1
        drone['last_plan_time'] = -1e9
        drone['planned_path'] = [(x_est, y_est)] if goal_xy is not None else []
        meta_dict = {} if meta is None else dict(meta)
        if goal_xy is not None:
            meta_dict['policy'] = self.auto_policy
            meta_dict['goal_type'] = self._goal_type_for_drone({'auto_goal_meta': meta_dict}) if meta_dict else ('coverage-fill' if self.auto_policy == 'weighted_coverage' else 'frontier')
        drone['auto_goal_meta'] = meta_dict or None
        if goal_xy is not None:
            drone['last_goal_type'] = self._goal_type_for_drone(drone)
            goal_changed = prev_goal is None or math.hypot(float(goal_xy[0]) - float(prev_goal[0]), float(goal_xy[1]) - float(prev_goal[1])) > 0.35 or prev_goal_type != drone['last_goal_type']
            if goal_changed:
                drone['goal_assignment_count'] = int(drone.get('goal_assignment_count', 0)) + 1
                self._log_event(
                    'goal_assigned',
                    f"Assigned {drone['last_goal_type']} goal at ({goal_xy[0]:.2f}, {goal_xy[1]:.2f})",
                    drone=drone,
                    goal_x=round(float(goal_xy[0]), 3),
                    goal_y=round(float(goal_xy[1]), 3),
                    goal_type=drone['last_goal_type'],
                    score=round(float(meta_dict.get('score', 0.0)), 4) if 'score' in meta_dict else None,
                )
        else:
            drone['last_goal_type'] = 'none'
            if prev_goal is not None:
                self._log_event('goal_cleared', 'No valid exploration goal available', drone=drone)
        self._update_mission_artist(drone)

    def _launch_phase_expired(self, drone):
        return self.time_elapsed >= float(AUTO_LAUNCH_DISPERSAL_MAX_SECONDS)

    def _teammate_context(self, drone):
        teammate_positions = []
        teammate_goal_positions = []
        for other in self.drones:
            if other is drone:
                continue
            ox, oy, _ = other['odometry'].mu
            teammate_positions.append((ox, oy))
            if other.get('auto_goal_xy') is not None and other.get('auto_phase', 'explore') == 'explore':
                teammate_goal_positions.append(tuple(other['auto_goal_xy']))
        return teammate_positions, teammate_goal_positions

    def _auto_centroid_for_drone(self, drone):
        idx = int(drone.get('drone_index', 0))
        if idx < len(self.partition_centroids_xy):
            centroid = np.asarray(self.partition_centroids_xy[idx], dtype=float)
            if np.all(np.isfinite(centroid)):
                return centroid
        return None

    def _current_goal_valid(self, drone):
        goal_xy = drone.get('auto_goal_xy')
        if goal_xy is None:
            return False
        x_est, y_est, _ = drone['odometry'].mu
        goal_tol = AUTO_LAUNCH_DISPERSAL_GOAL_TOLERANCE if (self.mission_mode == 'auto_explore' and drone.get('auto_phase', 'explore') == 'launch') else A_STAR_GOAL_TOLERANCE
        return math.hypot(float(goal_xy[0]) - x_est, float(goal_xy[1]) - y_est) > 1.05 * goal_tol

    def _should_keep_current_auto_goal(self, drone, candidate_goal_xy, candidate_meta):
        current_goal = drone.get('auto_goal_xy')
        if current_goal is None:
            return False
        goal_age = float(self.time_elapsed - drone.get('auto_goal_last_set_time', -1e9))
        current_valid = self._current_goal_valid(drone)
        if not current_valid:
            return False
        if not drone.get('path'):
            return False
        if goal_age < float(AUTO_GOAL_MIN_HOLD_SECONDS):
            return True
        if candidate_goal_xy is None:
            return goal_age < float(AUTO_GOAL_STALE_SECONDS)
        goal_sep = math.hypot(float(candidate_goal_xy[0]) - float(current_goal[0]), float(candidate_goal_xy[1]) - float(current_goal[1]))
        if goal_sep < float(AUTO_GOAL_MIN_SWITCH_DISTANCE):
            return True
        current_type = str(drone.get('last_goal_type', ''))
        candidate_type = '' if not candidate_meta else str(candidate_meta.get('goal_flavor') or candidate_meta.get('goal_type') or '')
        if goal_age < float(AUTO_GOAL_STALE_SECONDS) and candidate_type == current_type:
            return True
        return False

    def _choose_auto_goal(self, drone):
        x_est, y_est, _ = drone['odometry'].mu
        teammate_positions, teammate_goal_positions = self._teammate_context(drone)
        centroid_xy = self._auto_centroid_for_drone(drone)
        common = dict(
            robot_index=drone['drone_index'],
            partition_labels=self.partition_labels,
            robot_xy=(x_est, y_est),
            centroid_xy=centroid_xy,
            teammate_positions=teammate_positions,
            teammate_goal_positions=teammate_goal_positions,
        )
        if self.auto_policy == 'weighted_coverage':
            return self.coverage_controller.choose_goal(
                density_map=self.density_map,
                known_grid=self.shared_known_grid,
                free_value=FREE,
                occupied_value=OCCUPIED,
                unknown_value=UNKNOWN,
                planner=self.planner,
                grid_to_world_fn=self._grid_to_world,
                world_to_grid_fn=self._world_to_grid,
                frontier_components=self.frontier_components,
                visited_cells=drone.get('visited_cells', set()),
                **common,
            )
        return self.frontier_controller.choose_goal(
            frontier_components=self.frontier_components,
            grid_to_world_fn=self._grid_to_world,
            known_grid=self.shared_known_grid,
            planner=self.planner,
            occupied_value=OCCUPIED,
            unknown_value=UNKNOWN,
            density_map=self.density_map,
            **common,
        )

    def _any_assignable_frontier_goal(self):
        if self.mission_mode != 'auto_explore':
            return False
        if len(self.frontier_components) <= 0:
            return False
        for drone in self.drones:
            if drone.get('auto_phase', 'explore') != 'explore':
                continue
            goal_xy, _meta = self._choose_auto_goal(drone)
            if goal_xy is not None:
                return True
        return False

    def _auto_explore_done_now(self):
        if self.mission_mode != 'auto_explore':
            return False
        for drone in self.drones:
            if drone.get('auto_phase', 'explore') != 'explore':
                return False
            if drone.get('auto_goal_xy') is not None:
                return False
            if drone.get('path'):
                return False
            if drone.get('planned_path'):
                return False
            if drone.get('last_command_active', False):
                return False
        if len(self.frontier_components) <= 0:
            return True
        return not self._any_assignable_frontier_goal()

    def _maybe_finish_auto_explore(self):
        if not self.auto_mode or self.mission_mode != 'auto_explore' or not AUTO_STOP_WHEN_FINISHED:
            self.auto_finish_candidate_time = None
            return
        if not self._auto_explore_done_now():
            self.auto_finish_candidate_time = None
            return
        if self.auto_finish_candidate_time is None:
            self.auto_finish_candidate_time = self.time_elapsed
            return
        if (self.time_elapsed - self.auto_finish_candidate_time) < float(AUTO_FINISH_HOLD_SECONDS):
            return
        self.auto_mode = False
        self.auto_finished = True
        self.auto_finish_candidate_time = None
        for drone in self.drones:
            drone['robot'].set_motor_speeds(0.0, 0.0)
        self._refresh_toggle_button()
        self._refresh_control_text()
        self._log_event('auto_finished', 'Automatic exploration reached completion condition')
        self._finalize_current_run()

    def _choose_launch_staging_goal(self, drone):
        if not AUTO_LAUNCH_DISPERSAL_ENABLED:
            return None
        idx = int(drone.get('drone_index', 0))
        start_x, start_y, start_angle = self._start_pose_for_index(idx, len(self.drones))
        ref = None
        if idx < len(self.partition_centroids_xy):
            cand = np.asarray(self.partition_centroids_xy[idx], dtype=float)
            if np.all(np.isfinite(cand)):
                ref = cand
        if ref is None and idx < len(self.partition_generators_xy):
            cand = np.asarray(self.partition_generators_xy[idx], dtype=float)
            if np.all(np.isfinite(cand)):
                ref = cand
        if ref is None:
            ang = math.radians(start_angle)
            ref = np.array([start_x + math.cos(ang), start_y + math.sin(ang)], dtype=float)

        direction = np.asarray(ref, dtype=float) - np.array([start_x, start_y], dtype=float)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            ang = math.radians(start_angle)
            direction = np.array([math.cos(ang), math.sin(ang)], dtype=float)
            norm = 1.0
        direction = direction / norm

        planning_occ = self._planning_occupancy(self.shared_known_grid)
        x_est, y_est, _ = drone['odometry'].mu
        nominal = max(float(AUTO_LAUNCH_DISPERSAL_DISTANCE), 1.4 * A_STAR_GOAL_TOLERANCE)
        distances = [nominal, 0.86 * nominal, 0.72 * nominal, 0.58 * nominal, 0.44 * nominal]
        tried = []
        for dist in distances:
            cand_xy = np.array([start_x, start_y], dtype=float) + direction * dist
            cand_xy[0] = float(np.clip(cand_xy[0], 0.8, WORLD_WIDTH_METERS - 0.8))
            cand_xy[1] = float(np.clip(cand_xy[1], 0.8, WORLD_HEIGHT_METERS - 0.8))
            tried.append(tuple(cand_xy))

        if ref is not None:
            tried.append((float(np.clip(ref[0], 0.8, WORLD_WIDTH_METERS - 0.8)), float(np.clip(ref[1], 0.8, WORLD_HEIGHT_METERS - 0.8))))

        seen = set()
        for cand in tried:
            if cand in seen:
                continue
            seen.add(cand)
            cell = self.planner.nearest_free_cell(self._world_to_grid(*cand), planning_occ)
            goal_xy = self._grid_to_world(cell)
            if math.hypot(goal_xy[0] - start_x, goal_xy[1] - start_y) < 1.1 * A_STAR_GOAL_TOLERANCE:
                continue
            if self._line_crosses_blocked((x_est, y_est), goal_xy, planning_occ):
                path = self._astar((x_est, y_est), goal_xy, self.shared_known_grid)
                if not path:
                    continue
            return goal_xy
        return None

    def _ensure_auto_goal(self, drone):
        if self.mission_mode != 'auto_explore':
            return

        if drone.get('auto_phase', 'launch') == 'launch' and self._launch_phase_expired(drone):
            drone['auto_phase'] = 'explore'
            drone['auto_goal_xy'] = None
            drone['path'] = []

        current_goal = drone.get('auto_goal_xy')
        goal_age = float(self.time_elapsed - drone.get('auto_goal_last_set_time', -1e9))
        immediate_reassign = (
            current_goal is None
            or not drone.get('path')
            or not self._current_goal_valid(drone)
        )
        periodic_review = (not immediate_reassign) and (goal_age >= float(AUTO_EXPLORE_REPLAN_SECONDS))
        if not immediate_reassign and not periodic_review:
            return

        if drone.get('auto_phase', 'launch') == 'launch':
            stage_goal = self._choose_launch_staging_goal(drone)
            if stage_goal is not None:
                if self._should_keep_current_auto_goal(drone, stage_goal, None):
                    return
                self._assign_auto_goal(drone, stage_goal)
                return
            drone['auto_phase'] = 'explore'

        goal_xy, meta = self._choose_auto_goal(drone)
        if self._should_keep_current_auto_goal(drone, goal_xy, meta):
            return
        self._assign_auto_goal(drone, goal_xy, meta=meta)

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

        known = self.shared_known_grid
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
        start_pose = self._start_pose_for_index(drone_index, drone_count)
        drone = create_drone(
            ax=self.ax,
            name=name,
            color=color,
            path=path,
            drone_index=drone_index,
            start_pose=start_pose,
            robot_seed_rng=np.random.default_rng(self._seed_for_drone(drone_index, stream=1)),
            planner_init_known_grid=self._init_known_grid,
        )
        drone['rng'] = np.random.default_rng(self._seed_for_drone(drone_index, stream=2))
        drone['min_clearance'] = self.view_distance
        return drone

    def robot_shape_from_pose(self, x, y, angle_deg, size):
        return robot_shape_from_pose(x, y, angle_deg, size)

    def _make_fov_patch(self, x, y, angle_deg, color):
        return make_fov_patch(x, y, angle_deg, color, view_distance=self.view_distance, fov_angle=self.fov_angle)

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
        self._finalize_current_run()
        self._load_environment(seed)
        self._start_new_run()
        self.auto_mode = False
        self._refresh_toggle_button()
        self._refresh_control_text()
        for idx, drone in enumerate(self.drones):
            new_path = list(self.generated_target_sequences[idx]) if idx < len(self.generated_target_sequences) else []
            drone['manual_path'] = new_path
            drone['path'] = list(new_path) if self.mission_mode == 'manual_click' else []
            drone['auto_goal_xy'] = None
            drone['auto_goal_last_set_time'] = -1e9
            drone['auto_phase'] = 'launch'
        self.reset_simulation()
        self._saved_current_run = False
        self._mark_partition_dirty()
        self._refresh_partition_state(force=True)
        self.ui.refresh_shared_map()

    def toggle_auto_mode(self, event=None):
        self.auto_mode = not self.auto_mode
        if self.auto_mode:
            self.auto_finished = False
            self.auto_finish_candidate_time = None
        self._refresh_toggle_button()
        self._refresh_control_text()
        self._mark_partition_dirty()
        self._refresh_partition_state(force=True)
        if self.auto_mode and self.mission_mode == 'auto_explore':
            for drone in self.drones:
                drone['auto_goal_xy'] = None
                drone['auto_goal_last_set_time'] = -1e9
                drone['auto_goal_meta'] = None
                drone['auto_phase'] = 'launch'
                drone['path'] = []
                drone['planned_path'] = []
                drone['plan_goal_index'] = -1
                drone['last_plan_time'] = -1e9
                self._update_mission_artist(drone)
        if not self.auto_mode:
            for drone in self.drones:
                drone['robot'].set_motor_speeds(0.0, 0.0)
        self._log_event('auto_mode_toggled', f"Simulation {'started' if self.auto_mode else 'paused'}", auto_mode=bool(self.auto_mode))

    def reset_simulation(self, event=None):
        self._prepare_for_new_run()
        self.auto_mode = False
        self._refresh_toggle_button()
        self._refresh_control_text()
        self.trace_counter = 0
        self.time_elapsed = 0.0
        self.ui.seed_box.set_val(str(self.current_seed))
        self.shared_known_grid = self._init_known_grid()

        for idx, drone in enumerate(self.drones):
            start_x, start_y, start_angle = self._start_pose_for_index(idx, len(self.drones))
            drone['manual_path'] = list(self.generated_target_sequences[idx]) if idx < len(self.generated_target_sequences) else []
            drone['path'] = list(drone['manual_path']) if self.mission_mode == 'manual_click' else []
            drone['auto_goal_xy'] = None
            drone['auto_goal_last_set_time'] = -1e9
            drone['auto_phase'] = 'launch'
            robot = drone['robot']
            robot.rng = np.random.default_rng(self._seed_for_drone(idx, stream=1))
            drone['rng'] = np.random.default_rng(self._seed_for_drone(idx, stream=2))
            robot.x = start_x
            robot.y = start_y
            robot.angle = start_angle
            robot.left_motor_speed = 0.0
            robot.right_motor_speed = 0.0

            drone['odometry'].mu = np.array([start_x, start_y, start_angle], dtype=float)
            drone['odometry'].cov = np.diag([0.1, 0.1, 1.0])
            drone['current_target_index'] = 0
            drone['min_clearance'] = self.view_distance
            drone['planned_path'] = [(start_x, start_y)]
            drone['plan_goal_index'] = -1
            drone['last_plan_time'] = -1e9
            drone['path_progress_index'] = 0
            drone['plan_line'].set_data([start_x], [start_y])
            if drone['path']:
                drone['subgoal_marker'].set_data([start_x], [start_y])
                drone['target_marker'].set_data([drone['path'][0][0]], [drone['path'][0][1]])
            else:
                drone['subgoal_marker'].set_data([], [])
                drone['target_marker'].set_data([], [])
            drone['local_known_grid'] = self._init_known_grid()
            self._reveal_start_area(drone['local_known_grid'], start_x, start_y)
            self._reveal_start_area(self.shared_known_grid, start_x, start_y)
            drone['known_occ_count'] = int(np.count_nonzero(self.shared_known_grid == OCCUPIED))
            drone['local_known_occ_count'] = int(np.count_nonzero(drone['local_known_grid'] == OCCUPIED))
            drone['just_discovered_obstacle'] = False
            drone['recent_positions'] = [(0.0, start_x, start_y)]
            drone['recovery_until'] = -1e9
            drone['recovery_turn_sign'] = 1.0
            drone['last_scan_bias'] = 1.0
            drone['last_command_active'] = False
            drone['stuck_events'] = 0
            drone['replan_count'] = 0
            drone['blocked_replan_count'] = 0
            drone['goal_assignment_count'] = 0
            drone['goal_reached_count'] = 0
            drone['measurement_update_count'] = 0
            drone['last_landmark_update_time'] = None
            drone['distance_travelled'] = 0.0
            drone['idle_time'] = 0.0
            drone['last_goal_type'] = 'none'
            drone['visited_cells'] = {self._world_to_grid(start_x, start_y)}

            drone['trace_x'] = [start_x]
            drone['trace_y'] = [start_y]
            drone['est_trace_x'] = [start_x]
            drone['est_trace_y'] = [start_y]
            drone['trace_line'].set_data(drone['trace_x'], drone['trace_y'])
            drone['est_trace_line'].set_data(drone['est_trace_x'], drone['est_trace_y'])

            self._clear_local_grid_patches(drone)
            drone['true_patch'].set_xy(self.robot_shape_from_pose(start_x, start_y, start_angle, robot.size))
            drone['est_patch'].set_xy(self.robot_shape_from_pose(start_x, start_y, start_angle, robot.size * 0.92))
            update_uncertainty_ellipse_patch(drone['ellipse_patch'], drone['odometry'], n_std=2.5)
            update_fov_patch(drone['fov_patch'], start_x, start_y, start_angle, view_distance=self.view_distance, fov_angle=self.fov_angle)

            for line in drone['ray_lines']:
                line.set_data([start_x, start_x], [start_y, start_y])
            self._update_mission_artist(drone)

        self._saved_current_run = False
        self._mark_partition_dirty()
        self._refresh_partition_state(force=True)
        self.ui.refresh_shared_map()
        self._record_coverage_snapshot(force=True)
        self._log_event('run_reset', 'Simulation reset to initial conditions', seed=int(self.current_seed))

    def _apply_scan_to_grid(self, grid, robot, scan):
        return apply_scan_to_grid(
            grid, robot, scan, self._world_to_grid, self._stamp_obstacle_hit, self.grid_resolution, self.view_distance, UNKNOWN, FREE
        )

    def _update_known_map_from_scan(self, drone, scan):
        changed = update_known_map_from_scan(
            drone, self.shared_known_grid, scan, self._apply_scan_to_grid, OCCUPIED
        )
        if changed:
            self._mark_partition_dirty()

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
        planning_occ = self._planning_occupancy(self.shared_known_grid)
        plan_age = float(self.time_elapsed - drone['last_plan_time'])
        goal_changed = drone['plan_goal_index'] != target_idx
        missing_path = not drone['planned_path']

        goal_gx, goal_gy = self._world_to_grid(*goal)
        goal_blocked = bool(planning_occ[goal_gy, goal_gx])
        allow_blocked_refresh = goal_blocked and plan_age >= max(A_STAR_MIN_REPLAN_GAP_SECONDS, 1.5)

        need_replan = goal_changed or missing_path or allow_blocked_refresh

        if need_replan:
            old_path = list(drone['planned_path'])
            new_path = self._astar((x_est, y_est), goal, self.shared_known_grid)
            drone['plan_goal_index'] = target_idx
            drone['last_plan_time'] = self.time_elapsed
            drone['replan_count'] = int(drone.get('replan_count', 0)) + 1

            if new_path:
                drone['planned_path'] = new_path
                drone['path_progress_index'] = 1 if len(new_path) > 1 else 0
                old_len = polyline_length(old_path)
                new_len = polyline_length(new_path)
                if goal_changed:
                    reason = 'goal update'
                elif goal_blocked:
                    reason = 'goal blocked'
                else:
                    reason = 'path refresh'
                if (not old_path) or abs(new_len - old_len) > 0.75 or drone['replan_count'] in (1, 5, 10):
                    self._log_event('replan', f"Replanned path with {len(new_path)} points ({reason})", drone=drone, path_length=round(new_len, 3))
            elif old_path:
                drone['planned_path'] = old_path
                drone['path_progress_index'] = min(drone['path_progress_index'], max(0, len(old_path) - 1))
                drone['blocked_replan_count'] = int(drone.get('blocked_replan_count', 0)) + 1
                self._log_event('replan_failed', 'Replan blocked; keeping previous path', drone=drone)
            else:
                drone['planned_path'] = []
                drone['path_progress_index'] = 0
                drone['blocked_replan_count'] = int(drone.get('blocked_replan_count', 0)) + 1
                self._log_event('replan_failed', 'Replan failed; no fallback path available', drone=drone)

            xs = [p[0] for p in drone['planned_path']]
            ys = [p[1] for p in drone['planned_path']]
            drone['plan_line'].set_data(xs, ys)

    def _next_subgoal(self, drone):
        path_pts = drone['planned_path']
        if not path_pts:
            return None
        x_est, y_est, _ = drone['odometry'].mu
        prog = drone['path_progress_index']
        planning_occ = self._planning_occupancy(self.shared_known_grid)

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
        self._log_event('stuck_recovery', 'Triggered stuck recovery maneuver', drone=drone, stuck_events=int(drone['stuck_events']))

    def _compute_control(self, drone):
        robot = drone['robot']
        odometry = drone['odometry']
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

        if self.mission_mode == 'auto_explore':
            self._ensure_auto_goal(drone)
        path = drone['path']
        target_idx = drone['current_target_index']

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

        goal_tol = AUTO_LAUNCH_DISPERSAL_GOAL_TOLERANCE if (self.mission_mode == 'auto_explore' and drone.get('auto_phase', 'explore') == 'launch') else A_STAR_GOAL_TOLERANCE

        if dist_goal < goal_tol:
            drone['goal_reached_count'] = int(drone.get('goal_reached_count', 0)) + 1
            if self.mission_mode == 'auto_explore':
                reached_goal = drone.get('auto_goal_xy')
                if drone.get('auto_phase', 'explore') == 'launch':
                    drone['auto_phase'] = 'explore'
                drone['auto_goal_xy'] = None
                drone['auto_goal_last_set_time'] = self.time_elapsed
                drone['path'] = []
                drone['current_target_index'] = 0
                drone['target_marker'].set_data([], [])
                drone['subgoal_marker'].set_data([], [])
                drone['planned_path'] = []
                drone['plan_line'].set_data([], [])
                if reached_goal is not None:
                    self._log_event('goal_reached', f"Reached {drone.get('last_goal_type', 'goal')} at ({reached_goal[0]:.2f}, {reached_goal[1]:.2f})", drone=drone)
            else:
                reached_goal = path[target_idx] if target_idx < len(path) else None
                drone['current_target_index'] += 1
                if drone['current_target_index'] >= len(path):
                    drone['target_marker'].set_data([], [])
                    drone['subgoal_marker'].set_data([], [])
                    drone['planned_path'] = []
                    drone['plan_line'].set_data([], [])
                if reached_goal is not None:
                    self._log_event('manual_waypoint_reached', f"Reached manual waypoint at ({reached_goal[0]:.2f}, {reached_goal[1]:.2f})", drone=drone)
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
            prev_x = float(robot.x)
            prev_y = float(robot.y)
            robot.update(dt, obstacles=self.obstacles)
            drone['distance_travelled'] = float(drone.get('distance_travelled', 0.0)) + math.hypot(float(robot.x) - prev_x, float(robot.y) - prev_y)
            drone.setdefault('visited_cells', set()).add(self._world_to_grid(robot.x, robot.y))
            if not drone.get('last_command_active', False):
                drone['idle_time'] = float(drone.get('idle_time', 0.0)) + float(dt)
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
            drone['measurement_update_count'] = int(drone.get('measurement_update_count', 0)) + 1
            drone['last_landmark_update_time'] = float(self.time_elapsed)

        est_x, est_y, est_theta = odometry.mu
        drone['true_patch'].set_xy(self.robot_shape_from_pose(robot.x, robot.y, robot.angle, robot.size))
        drone['est_patch'].set_xy(self.robot_shape_from_pose(est_x, est_y, est_theta, robot.size * 0.92))
        update_uncertainty_ellipse_patch(drone['ellipse_patch'], odometry, n_std=2.5)
        update_fov_patch(drone['fov_patch'], robot.x, robot.y, robot.angle, view_distance=self.view_distance, fov_angle=self.fov_angle)
        self._update_local_grid_visual(drone)

        return detected

    def update(self, frame):
        dt = TIME_STEP if self.auto_mode else 0.0
        if dt > 0.0:
            self.time_elapsed += dt
            self._saved_current_run = False

        self._refresh_partition_state(force=False)
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
            self._mark_partition_dirty()
        self._refresh_partition_state(force=False)
        self._record_coverage_snapshot(force=(self.time_elapsed <= 1e-9))
        if dt > 0.0 and self.auto_mode:
            self._maybe_print_live_snapshot(force=False)
        self._maybe_finish_auto_explore()
        self.ui.refresh_status_text()
        self._saved_current_run = False
        self.ui.refresh_shared_map()
        self.ui.refresh_partition_overlay()
        artists.append(self.ui.status_text)
        if self.ui.shared_map_image is not None:
            artists.append(self.ui.shared_map_image)
        if self.ui.shared_robot_scatter is not None:
            artists.append(self.ui.shared_robot_scatter)
        if self.ui.partition_image is not None:
            artists.append(self.ui.partition_image)
        if self.ui.partition_generator_scatter is not None:
            artists.append(self.ui.partition_generator_scatter)
        if self.ui.partition_centroid_scatter is not None:
            artists.append(self.ui.partition_centroid_scatter)
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
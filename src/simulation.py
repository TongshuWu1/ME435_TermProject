import json
import math
from collections import deque
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
    A_STAR_GOAL_CLEARANCE_CELLS,
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
    AUTO_STARTUP_DISPERSION_MEAN_PAIRWISE_DISTANCE,
    AUTO_STARTUP_PARTITION_PENALTY_SCALE,
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
    HOME_BASE_DRAW_ALPHA,
    HOME_BASE_HEIGHT,
    HOME_BASE_MARKER_COLOR,
    HOME_BASE_MARKER_SHAPE,
    HOME_BASE_MARKER_SIZE,
    HOME_BASE_SPAWN_SLOT_MARGIN_X,
    HOME_BASE_SPAWN_FRONT_OFFSET,
    HOME_BASE_SPAWN_HEADING_FAN_DEGREES,
    HOME_BASE_STAGING_DISTANCE,
    HOME_BASE_WIDTH,
    FOV_ANGLE,
    KNOWN_MAP_REPLAN_ON_NEW_OBS,
    MEASUREMENT_ALPHA,
    MEASUREMENT_NOISE,
    MAP_REPLAY_MIN_HEADING_SHIFT_DEG,
    MAP_REPLAY_MIN_POSITION_SHIFT,
    MAP_REPLAY_MIN_TRACE_IMPROVEMENT,
    ENABLE_RECENT_SCAN_REPLAY,
    MAP_SCAN_BUFFER_SIZE,
    MAP_REPLAY_MAX_SCANS,
    COMMUNICATION_RADIUS,
    ROBOT_AVOIDANCE_RADIUS,
    ROBOT_AVOIDANCE_GAIN,
    ROBOT_RADIUS,
    PATH_HISTORY_PACKET_POINTS,
    TEAMMATE_PACKET_STALE_SECONDS,
    TEAMMATE_STALE_DECAY,
    TEAMMATE_TRACE_SCALE,
    TEAMMATE_MIN_RELIABILITY,
    INTERROBOT_USE_AS_LANDMARK,
    INTERROBOT_MAX_REFERENCE_TRACE,
    INTERROBOT_MEASUREMENT_NOISE,
    INTERROBOT_NOISE_FROM_REFERENCE_TRACE_GAIN,
    PATH_LINE_WIDTH,
    PREDICTION_NOISE,
    RANDOM_SEED,
    RAY_ALPHA,
    RAY_LINE_WIDTH,
    SHOW_ASTAR_LOCAL_GRID,
    SHOW_DENSITY_OVERLAY_BY_DEFAULT,
    SHOW_UNCERTAINTY_OVERLAY_BY_DEFAULT,
    MAP_UNCERTAINTY_VIS_METERS_SCALE,
    AUTO_FINISH_MIN_TOTAL_FRONTIER_CELLS,
    SHOW_PARTITION_OVERLAY_BY_DEFAULT,
    SHOW_VISION_RAYS,
    RETURN_HOME_FINISH_HOLD_SECONDS,
    RETURN_HOME_GOAL_TOLERANCE,
    START_SIMULATION_RUNNING,
    STUCK_PROGRESS_EPS,
    STUCK_RECOVERY_SECONDS,
    STUCK_REVERSE_SPEED,
    STUCK_TURN_SPEED,
    STUCK_WINDOW_SECONDS,
    SUBGOAL_MARKER_SIZE,
    SUBGOAL_OBSTACLE_CLEARANCE_METERS,
    SUBGOAL_CLEARANCE_PREFERENCE,
    TARGET_MARKER_SIZE,
    UI_HEAVY_REFRESH_EVERY_FRAMES,
    UI_SHARED_REFRESH_EVERY_FRAMES,
    UI_MONITOR_REFRESH_EVERY_FRAMES,
    UI_OVERLAY_REFRESH_EVERY_FRAMES,
    TIME_STEP,
    VIEW_DISTANCE,
    VISION_RAY_COUNT,
    WORLD_HEIGHT_METERS,
    WORLD_WIDTH_METERS,
)
from .environment import empty_target_sequences, generate_environment, square_contains_point, home_base_region
from .auto_explore import partition_generators_from_positions
from .controllers import FrontierController, WeightedCoverageController
from .landmark import Landmark
from .mapping_utils import (
    apply_scan_to_grid,
    clearance_groups,
    initialize_belief_grids,
    initialize_pose_uncertainty_grid,
    reveal_start_area,
    reveal_start_area_with_belief,
    scan_buffer_entry_from_scan,
    scan_from_buffer_entry,
    shifted_scan_buffer_entry,
    update_known_map_from_scan,
)
from .metrics import compute_coverage_metrics, polyline_length
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
from .sim_ui import SimulatorUI

RENDER_FPS = 20
UNKNOWN = 0
FREE = 1
OCCUPIED = 2

class Simulator:
    def __init__(self):
        self.current_seed = RANDOM_SEED
        self.run_index = 0
        self.session_label = 'session'
        self.landmark_patches = []
        self.obstacle_patches = []
        self.landmarks = []
        self.shared_known_landmarks = {}
        self.obstacles = []
        self.home_base = home_base_region()
        self.home_base_patch = None
        self.fov_angle = FOV_ANGLE
        self.view_distance = VIEW_DISTANCE
        self.auto_mode = START_SIMULATION_RUNNING
        self.trace_interval = 5
        self.trace_counter = 0
        self.time_elapsed = 0.0
        self.auto_finished = False
        self.auto_finish_candidate_time = None
        self.return_phase_active = False
        self.return_finish_candidate_time = None
        self.shared_los_segments = []
        self._planning_occupancy_version = 0
        self._planning_occ_cache_frame = {}

        self.grid_resolution = A_STAR_GRID_RESOLUTION
        self.planner = GridPlanner(A_STAR_GRID_RESOLUTION, A_STAR_INFLATION_MARGIN)
        self.nx = self.planner.nx
        self.ny = self.planner.ny
        self.truth_occupancy = self._build_truth_occupancy_grid()
        self.shared_known_grid = self._init_known_grid()
        self.shared_logodds_grid, self.shared_confidence_grid = self._init_belief_layers(self.shared_known_grid)
        self.shared_pose_uncertainty_grid = self._init_pose_uncertainty_grid(self.shared_known_grid)
        self.mission_mode = DEFAULT_MISSION_MODE
        self.auto_policy = DEFAULT_AUTO_POLICY
        self.show_partition_overlay = SHOW_PARTITION_OVERLAY_BY_DEFAULT
        self.show_density_overlay = SHOW_DENSITY_OVERLAY_BY_DEFAULT
        self.show_uncertainty_overlay = SHOW_UNCERTAINTY_OVERLAY_BY_DEFAULT
        self.partition_labels = -np.ones((self.ny, self.nx), dtype=int)
        self.partition_rgba = np.zeros((self.ny, self.nx, 4), dtype=float)
        self.density_map = np.zeros((self.ny, self.nx), dtype=float)
        self.density_rgba = np.zeros((self.ny, self.nx, 4), dtype=float)
        self.uncertainty_rgba = np.zeros((self.ny, self.nx, 4), dtype=float)
        self._shared_uncertainty_dirty = True
        self.partition_generators_xy = np.zeros((0, 2), dtype=float)
        self.partition_centroids_xy = np.zeros((0, 2), dtype=float)
        self.partition_generator_colors = []
        self.frontier_components = []
        self._partition_dirty = True
        self._last_partition_generators = None
        self.auto_finished = False
        self.auto_finish_candidate_time = None
        self.shared_los_segments = []

        self.selected_drone_index = 0
        self.edit_mode = 'add_waypoint'
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
            drone['local_pose_uncertainty_grid'] = self._init_pose_uncertainty_grid(drone['local_known_grid'])
            self._reveal_start_area_with_belief(
                drone['local_known_grid'],
                drone['local_logodds_grid'],
                drone['local_confidence_grid'],
                drone['local_pose_uncertainty_grid'],
                drone['robot'].x,
                drone['robot'].y,
            )
            self._reveal_start_area_with_belief(
                self.shared_known_grid,
                self.shared_logodds_grid,
                self.shared_confidence_grid,
                self.shared_pose_uncertainty_grid,
                drone['robot'].x,
                drone['robot'].y,
            )
            self.drones.append(drone)

        legend_handles = [
            patches.Patch(facecolor=self.colors[i], edgecolor='black', label=self.drones[i]['name'])
            for i in range(len(self.drones))
        ]
        self.ax.legend(handles=legend_handles, loc='upper left')
        self._sync_paths_for_mode()
        self._mark_shared_uncertainty_dirty()
        for _dr in self.drones:
            self._mark_local_uncertainty_dirty(_dr)
        self._refresh_partition_state()
        self._update_los_and_packets()
        self.ui.build_robot_monitor(self.drones)
        self.ui.refresh_all()
        self.ui.refresh_robot_monitor()

    def _refresh_toggle_button(self):
        self.ui.refresh_toggle_button()

    def _refresh_partition_button(self):
        self.ui.refresh_partition_button()

    def _refresh_control_text(self):
        self.ui.refresh_status_text()

    def _mark_partition_dirty(self):
        self._partition_dirty = True

    def _mark_shared_uncertainty_dirty(self):
        self._shared_uncertainty_dirty = True

    @staticmethod
    def _mark_local_uncertainty_dirty(drone):
        drone['local_uncertainty_dirty'] = True

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
        prefix = f"[{self.session_label} t={self.time_elapsed:6.2f}s]"
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
        metrics['landmarks_total'] = int(len(getattr(self, 'landmarks', [])))
        metrics['landmarks_discovered'] = int(len(getattr(self, 'shared_known_landmarks', {})))
        metrics['landmark_recall_ratio'] = (metrics['landmarks_discovered'] / metrics['landmarks_total']) if metrics['landmarks_total'] else 0.0
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
            'landmarks_total': int(metrics.get('landmarks_total', 0)),
            'landmarks_discovered': int(metrics.get('landmarks_discovered', 0)),
            'landmark_recall_ratio': round(float(metrics.get('landmark_recall_ratio', 0.0)), 6),
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
            f"[{self.session_label} | t={self.time_elapsed:6.2f}s] "
            f"mode={mission_name}, policy={policy_name}, "
            f"map_known={100.0 * metrics['known_ratio']:5.1f}%, "
            f"free_covered={100.0 * metrics['free_coverage_ratio']:5.1f}%, "
            f"obstacles_found={100.0 * metrics['occupied_recall_ratio']:5.1f}%, "
            f"landmarks={metrics.get('landmarks_discovered', 0):d}/{metrics.get('landmarks_total', 0):d}, "
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


    @staticmethod
    def _landmark_key(landmark):
        return f"{str(landmark.shape)}|{str(landmark.color_name)}|{float(landmark.x):.3f}|{float(landmark.y):.3f}"

    @staticmethod
    def _serialize_landmark(landmark, *, seen_time=0.0, robot_name=''):
        return {
            'shape': str(landmark.shape),
            'color_name': str(landmark.color_name),
            'x': round(float(landmark.x), 3),
            'y': round(float(landmark.y), 3),
            'size': round(float(getattr(landmark, 'size', 0.8)), 3),
            'seen_count': 1,
            'first_seen_time': round(float(seen_time), 3),
            'last_seen_time': round(float(seen_time), 3),
            'first_seen_by': str(robot_name),
            'last_seen_by': str(robot_name),
            'seen_by': {str(robot_name)} if robot_name else set(),
        }

    def _remember_landmarks(self, drone, detected):
        robot_name = str(drone.get('name', ''))
        seen_time = float(self.time_elapsed)
        new_shared = []
        visible_keys = []
        local_memory = drone.setdefault('known_landmarks', {})
        for landmark in detected:
            key = self._landmark_key(landmark)
            visible_keys.append(key)
            if key not in local_memory:
                local_memory[key] = self._serialize_landmark(landmark, seen_time=seen_time, robot_name=robot_name)
            else:
                entry = local_memory[key]
                entry['seen_count'] = int(entry.get('seen_count', 0)) + 1
                entry['last_seen_time'] = round(seen_time, 3)
                entry['last_seen_by'] = robot_name
                entry.setdefault('seen_by', set()).add(robot_name)
            if key not in self.shared_known_landmarks:
                self.shared_known_landmarks[key] = self._serialize_landmark(landmark, seen_time=seen_time, robot_name=robot_name)
                new_shared.append(key)
            else:
                entry = self.shared_known_landmarks[key]
                entry['seen_count'] = int(entry.get('seen_count', 0)) + 1
                entry['last_seen_time'] = round(seen_time, 3)
                entry['last_seen_by'] = robot_name
                entry.setdefault('seen_by', set()).add(robot_name)
        drone['last_detected_landmarks'] = visible_keys
        return new_shared

    def _landmark_rows(self):
        rows = []
        for idx, (key, row) in enumerate(sorted(self.shared_known_landmarks.items(), key=lambda kv: (float(kv[1].get('first_seen_time', 0.0)), kv[0])), start=1):
            seen_by = sorted(str(x) for x in row.get('seen_by', set()) if str(x))
            rows.append({
                'landmark_id': idx,
                'landmark_key': key,
                'shape': row.get('shape', ''),
                'color_name': row.get('color_name', row.get('color', '')),
                'x_m': row.get('x', ''),
                'y_m': row.get('y', ''),
                'size_m': row.get('size', ''),
                'seen_count': row.get('seen_count', 0),
                'first_seen_time_s': row.get('first_seen_time', 0.0),
                'last_seen_time_s': row.get('last_seen_time', 0.0),
                'first_seen_by': row.get('first_seen_by', ''),
                'last_seen_by': row.get('last_seen_by', ''),
                'seen_by_count': len(seen_by),
                'seen_by_robots': ', '.join(seen_by),
            })
        return rows

    def _remaining_path_length(self, drone):
        path = list(drone.get('planned_path', []))
        if not path:
            return 0.0
        progress_index = max(0, int(drone.get('path_progress_index', 0)))
        if progress_index >= len(path):
            return 0.0
        remaining = path[progress_index:]
        est_x, est_y, _ = drone['odometry'].mu
        total = polyline_length(remaining)
        first_x, first_y = remaining[0]
        total += math.hypot(float(first_x) - float(est_x), float(first_y) - float(est_y))
        return total

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
            f'Landmk  : {int(metrics.get("landmarks_discovered", 0))}/{int(metrics.get("landmarks_total", 0))} discovered',
            f'Frontier: {metrics["frontier_count"]} groups    Goals: {metrics["active_goals"]}',
        ]
        if drone is not None:
            est_x, est_y, _ = drone['odometry'].mu
            goal_xy = drone.get('auto_goal_xy')
            remaining_length = self._remaining_path_length(drone)
            goal_text = 'none' if goal_xy is None else f'({goal_xy[0]:4.1f}, {goal_xy[1]:4.1f})'
            last_landmark = '-' if drone.get('last_landmark_update_time') is None else f"{float(drone['last_landmark_update_time']):4.1f}s"
            lines.extend([
                '',
                f'Robot   : {drone["name"]}    Phase: {drone.get("auto_phase", "manual")}',
                f'Goal    : {self._goal_type_for_drone(drone)} -> {goal_text}',
                f'Est xy  : ({est_x:4.1f}, {est_y:4.1f})    Path: {remaining_length:4.1f} m',
                f'Goals   : {drone.get("goal_reached_count", 0)} reached / {drone.get("goal_assignment_count", 0)} assigned',
                f'Replan attempts: {drone.get("replan_count", 0)}    Stuck: {drone.get("stuck_events", 0)}',
                f'Landmk  : {len(drone.get("known_landmarks", {}))} local, {len(drone.get("last_detected_landmarks", []))} vis',
                f'Updates : {drone.get("measurement_update_count", 0)} corr    Last: {last_landmark}',
                f'LOS Peers: {len(drone.get("visible_teammates", []))} -> {', '.join(drone.get("visible_teammates", [])[:3]) if drone.get("visible_teammates", []) else '-'}',
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
        self.return_phase_active = False
        self.return_finish_candidate_time = None
        self.shared_los_segments = []
        self._mark_partition_dirty()

    def on_select_mission_mode(self, label):
        norm = str(label).strip().lower()
        self.mission_mode = 'auto_explore' if 'auto' in norm else 'manual_click'
        self._sync_paths_for_mode()
        self.auto_finished = False
        self.auto_finish_candidate_time = None
        self.shared_los_segments = []
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
        self.shared_los_segments = []
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

    def toggle_uncertainty_overlay(self, event=None):
        self.show_uncertainty_overlay = not self.show_uncertainty_overlay
        self._refresh_uncertainty_button()
        self.ui.refresh_partition_overlay()
        self.ui.refresh_robot_monitor()
        self.fig.canvas.draw_idle()

    def _refresh_density_button(self):
        self.ui.refresh_density_button()

    def _refresh_uncertainty_button(self):
        self.ui.refresh_uncertainty_button()

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
        drone['local_logodds_grid'], drone['local_confidence_grid'] = self._init_belief_layers(drone['local_known_grid'])
        drone['local_pose_uncertainty_grid'] = self._init_pose_uncertainty_grid(drone['local_known_grid'])
        drone['recent_scan_buffer'] = deque(maxlen=MAP_SCAN_BUFFER_SIZE)
        self._reveal_start_area_with_belief(drone['local_known_grid'], drone['local_logodds_grid'], drone['local_confidence_grid'], drone['local_pose_uncertainty_grid'], x, y)
        self._reveal_start_area_with_belief(self.shared_known_grid, self.shared_logodds_grid, self.shared_confidence_grid, self.shared_pose_uncertainty_grid, x, y)
        drone['trace_x'] = [x]
        drone['trace_y'] = [y]
        drone['est_trace_x'] = [x]
        drone['est_trace_y'] = [y]
        drone['trace_line'].set_data(drone['trace_x'], drone['trace_y'])
        drone['est_trace_line'].set_data(drone['est_trace_x'], drone['est_trace_y'])
        drone['recent_positions'] = [(self.time_elapsed, *self._estimated_xy(drone))]
        drone['just_discovered_obstacle'] = True
        drone['last_command_active'] = False
        drone['distance_travelled'] = 0.0
        drone['idle_time'] = 0.0
        drone['visited_cells'] = {self._estimated_grid_cell(drone)}
        drone['last_pose_trace'] = float(np.trace(drone['odometry'].cov[:2, :2]))
        self._update_mission_artist(drone)
        drone['true_patch'].set_xy(self.robot_shape_from_pose(x, y, angle, robot.size))
        drone['est_patch'].set_xy(self.robot_shape_from_pose(x, y, angle, robot.size * 0.92))
        update_uncertainty_ellipse_patch(drone['ellipse_patch'], drone['odometry'], n_std=2.5)
        update_fov_patch(drone['fov_patch'], x, y, angle, view_distance=self.view_distance, fov_angle=self.fov_angle)
        for line in drone['ray_lines']:
            line.set_data([x, x], [y, y])
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
        drone['planned_path'] = []
        drone['path_progress_index'] = 0
        drone['plan_goal_index'] = -1
        drone['last_plan_time'] = -1e9
        drone['plan_line'].set_data([], [])
        drone['subgoal_marker'].set_data([], [])
        drone['last_goal_type'] = 'none'
        self._update_mission_artist(drone)
        self._mark_partition_dirty()
        self._refresh_partition_state(force=True)
        self._update_los_and_packets()
        self.ui.refresh_shared_map()
        self.ui.refresh_robot_monitor()
        self._log_event('path_cleared', 'Cleared path and current goal', drone=drone)
        self.fig.canvas.draw_idle()

    def _save_outputs(self):
        return None

    def _start_new_run(self):
        self._reset_reporting_state()
        self.session_label = f'session_seed_{int(self.current_seed)}'
        self._log_event('session_started', 'Started session', seed=int(self.current_seed), mission_mode=self.mission_mode, auto_policy=self.auto_policy)

    def _finalize_current_run(self):
        return None

    def _prepare_for_new_run(self):
        self._reset_reporting_state()
        self.session_label = f'session_seed_{int(self.current_seed)}'
        self._log_event('session_reset', 'Reset simulation state', seed=int(self.current_seed))

    def _seed_for_drone(self, drone_index, stream=0):
        return int(self.current_seed * 1000 + 97 * (drone_index + 1) + stream)

    def _start_pose_for_index(self, drone_index, drone_count):
        base = self.home_base
        cx = float(base["cx"])
        cy = float(base["cy"])
        base_angle = float(base["heading_deg"])
        # Spawn robots in stable slots across the home-base width instead of a tight arc.
        # This reduces startup symmetry and gives each robot a clear initial lane out of home.
        usable_half_w = max(0.3, float(HOME_BASE_WIDTH) / 2.0 - float(HOME_BASE_SPAWN_SLOT_MARGIN_X) - float(ROBOT_RADIUS))
        front_y = cy + max(0.0, float(HOME_BASE_HEIGHT) / 2.0 - float(HOME_BASE_SPAWN_FRONT_OFFSET) - float(ROBOT_RADIUS))
        if drone_count <= 1:
            x = cx
            y = front_y
            return float(x), float(y), base_angle

        xs = np.linspace(cx - usable_half_w, cx + usable_half_w, int(drone_count))
        x = float(xs[int(np.clip(drone_index, 0, drone_count - 1))])
        y = float(front_y)
        frac = drone_index / max(1, drone_count - 1)
        # Fan headings outward from the base center so left slots head left/outward
        # and right slots head right/outward instead of crossing through the base.
        heading_deg = base_angle + 0.5 * float(HOME_BASE_SPAWN_HEADING_FAN_DEGREES) - float(HOME_BASE_SPAWN_HEADING_FAN_DEGREES) * frac
        # Keep the spawn strictly inside the home base footprint.
        half_w = float(HOME_BASE_WIDTH) / 2.0 - float(ROBOT_RADIUS) - 0.05
        half_h = float(HOME_BASE_HEIGHT) / 2.0 - float(ROBOT_RADIUS) - 0.05
        x = float(np.clip(x, cx - half_w, cx + half_w))
        y = float(np.clip(y, cy - half_h, cy + half_h))
        x = float(np.clip(x, 1.0, WORLD_WIDTH_METERS - 1.0))
        y = float(np.clip(y, 1.0, WORLD_HEIGHT_METERS - 1.0))
        return x, y, heading_deg

    def _load_environment(self, seed):
        self.current_seed = int(seed)
        self.home_base = home_base_region()
        obstacles, landmark_dicts = generate_environment(self.current_seed)
        self.obstacles = list(obstacles)
        self.landmarks = [Landmark(**lm) for lm in landmark_dicts]
        self.landmarks.append(self._home_marker())
        self.shared_known_landmarks = {}
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
        if self.home_base_patch is not None:
            try:
                self.home_base_patch.remove()
            except ValueError:
                pass
            self.home_base_patch = None

        base = self.home_base
        self.home_base_patch = patches.Rectangle(
            (base["cx"] - base["width"] / 2.0, base["cy"] - base["height"] / 2.0),
            base["width"],
            base["height"],
            facecolor=(0.18, 0.60, 0.32, float(HOME_BASE_DRAW_ALPHA)),
            edgecolor=(0.12, 0.45, 0.22, 0.95),
            linewidth=1.4,
            linestyle='--',
            zorder=1.8,
        )
        self.ax.add_patch(self.home_base_patch)

        for obs in self.obstacles:
            half = obs['size'] / 2.0
            patch = patches.Rectangle(
                (obs['x'] - half, obs['y'] - half),
                obs['size'],
                obs['size'],
                facecolor=COLOR_OBSTACLE,
                edgecolor=COLOR_OBSTACLE,
                linewidth=0.6,
                alpha=0.92,
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

    def _init_belief_layers(self, known_grid=None):
        base_grid = self._init_known_grid() if known_grid is None else known_grid
        return initialize_belief_grids(base_grid, OCCUPIED)

    def _init_pose_uncertainty_grid(self, known_grid=None):
        base_grid = self._init_known_grid() if known_grid is None else known_grid
        return initialize_pose_uncertainty_grid(base_grid, OCCUPIED)

    def _reveal_start_area_with_belief(self, grid, logodds_grid, confidence_grid, pose_uncertainty_grid, x, y, radius_m=1.1):
        reveal_start_area_with_belief(
            grid,
            logodds_grid,
            confidence_grid,
            pose_uncertainty_grid,
            x,
            y,
            radius_m,
            self._world_to_grid,
            self._offsets_for_margin,
            self.nx,
            self.ny,
            UNKNOWN,
            FREE,
        )

    def _offsets_for_margin(self, margin_m):
        return self.planner.offsets_for_margin(margin_m)

    def _stamp_obstacle_hit(self, known_grid, gx, gy):
        return self.planner.stamp_obstacle_hit(known_grid, gx, gy, OCCUPIED)


    def _invalidate_planning_cache(self):
        self._planning_occupancy_version = int(getattr(self, '_planning_occupancy_version', 0)) + 1
        cache = getattr(self, '_planning_occ_cache_frame', None)
        if isinstance(cache, dict):
            cache.clear()

    def _planning_occupancy(self, known_grid, exclude_drone=None):
        exclude_idx = -1 if exclude_drone is None else int(exclude_drone.get('drone_index', -1))
        pos_signature = tuple(self._estimated_grid_cell(dr) for dr in getattr(self, 'drones', []))
        cache_key = (int(getattr(self, '_planning_occupancy_version', 0)), exclude_idx, pos_signature)
        cache = getattr(self, '_planning_occ_cache_frame', None)
        if isinstance(cache, dict) and cache_key in cache:
            return cache[cache_key]

        occ = np.array(self.planner.planning_occupancy(known_grid, OCCUPIED), copy=True)
        for other in getattr(self, 'drones', []):
            if exclude_drone is not None and other is exclude_drone:
                continue
            ox, oy = self._estimated_xy(other)
            radius_m = float(getattr(other['robot'], 'size', 0.0)) / 2.0
            for dx, dy in self._offsets_for_margin(radius_m):
                gx, gy = self._world_to_grid(ox + dx * self.grid_resolution, oy + dy * self.grid_resolution)
                if 0 <= gx < self.nx and 0 <= gy < self.ny:
                    occ[gy, gx] = True
        if isinstance(cache, dict):
            cache[cache_key] = occ
        return occ

    def _world_to_grid(self, x, y):
        return self.planner.world_to_grid(x, y)

    @staticmethod
    def _estimated_xy(drone):
        x_est, y_est, _ = drone['odometry'].mu
        return float(x_est), float(y_est)

    def _estimated_grid_cell(self, drone):
        x_est, y_est = self._estimated_xy(drone)
        return self._world_to_grid(x_est, y_est)

    def _recent_estimated_path(self, drone, max_points=PATH_HISTORY_PACKET_POINTS):
        xs = list(drone.get('est_trace_x', []))
        ys = list(drone.get('est_trace_y', []))
        pts = list(zip(xs, ys))
        if not pts:
            return []
        if len(pts) > max_points:
            step = max(1, len(pts) // max_points)
            pts = pts[::step]
            if pts[-1] != (xs[-1], ys[-1]):
                pts.append((xs[-1], ys[-1]))
        return [(float(x), float(y)) for x, y in pts[-max_points:]]

    def _packet_from_drone(self, drone):
        pose = tuple(float(v) for v in drone['odometry'].mu)
        pose_cov = np.asarray(drone['odometry'].cov, dtype=float)
        pose_trace = float(np.trace(pose_cov[:2, :2]))
        role = str(drone.get('team_role') or drone.get('auto_phase') or ('explorer' if self.mission_mode == 'auto_explore' else 'manual'))
        past_waypoints = self._recent_estimated_path(drone)
        return {
            'name': str(drone.get('name', '')),
            'timestamp': float(self.time_elapsed),
            'pose': pose,
            'pose_cov': pose_cov.tolist(),
            'pose_trace': pose_trace,
            'goal': tuple(drone['auto_goal_xy']) if drone.get('auto_goal_xy') is not None else None,
            'role': role,
            'path_history': past_waypoints,
            'past_waypoints': past_waypoints,
        }

    def _exchange_packet(self, receiver, sender):
        receiver.setdefault('teammate_memory', {})[sender['name']] = self._packet_from_drone(sender)

    def _packet_reliability(self, packet):
        age = max(0.0, float(self.time_elapsed) - float(packet.get('timestamp', -1e9)))
        stale_weight = math.exp(-float(TEAMMATE_STALE_DECAY) * age)
        pose_trace = max(0.0, float(packet.get('pose_trace', 0.0)))
        trace_scale = max(1e-6, float(TEAMMATE_TRACE_SCALE))
        trace_weight = 1.0 / (1.0 + pose_trace / trace_scale)
        return max(float(TEAMMATE_MIN_RELIABILITY), min(1.0, stale_weight * trace_weight))

    def _valid_teammate_packets(self, drone):
        packets = []
        stale_limit = float(TEAMMATE_PACKET_STALE_SECONDS)
        for packet in drone.get('teammate_memory', {}).values():
            if not isinstance(packet, dict):
                continue
            age = float(self.time_elapsed) - float(packet.get('timestamp', -1e9))
            if age > stale_limit:
                continue
            pkt = dict(packet)
            pkt['age'] = age
            pkt['reliability'] = self._packet_reliability(pkt)
            packets.append(pkt)
        return packets

    @staticmethod
    def _sanitize_xy_cov(cov_xy, fallback_trace=1e-3):
        cov_xy = np.asarray(cov_xy, dtype=float)
        if cov_xy.shape != (2, 2):
            fallback_trace = max(float(fallback_trace), 1e-6)
            cov_xy = np.eye(2, dtype=float) * (0.5 * fallback_trace)
        cov_xy = 0.5 * (cov_xy + cov_xy.T)
        eigvals, eigvecs = np.linalg.eigh(cov_xy)
        eigvals = np.maximum(eigvals, 1e-9)
        return eigvecs @ np.diag(eigvals) @ eigvecs.T

    def _packet_xy_cov(self, packet):
        pose_cov = np.asarray(packet.get('pose_cov', []), dtype=float)
        pose_trace = max(0.0, float(packet.get('pose_trace', 0.0)))
        if pose_cov.shape == (3, 3):
            xy_cov = pose_cov[:2, :2]
        elif pose_cov.shape == (2, 2):
            xy_cov = pose_cov
        else:
            xy_cov = np.eye(2, dtype=float) * max(1e-6, 0.5 * pose_trace)
        xy_cov = self._sanitize_xy_cov(xy_cov, fallback_trace=pose_trace)

        age = max(0.0, float(packet.get('age', 0.0)))
        drift_std = max(float(PREDICTION_NOISE[0]), float(PREDICTION_NOISE[1])) * age
        drift_cov = np.eye(2, dtype=float) * (drift_std ** 2)
        reference_floor_cov = np.eye(2, dtype=float) * (0.5 * float(INTERROBOT_NOISE_FROM_REFERENCE_TRACE_GAIN) * pose_trace)
        return self._sanitize_xy_cov(xy_cov + drift_cov + reference_floor_cov, fallback_trace=pose_trace)

    def _home_base_measurement(self, drone):
        robot = drone['robot']
        base_x = float(self.home_base['cx'])
        base_y = float(self.home_base['cy'])
        dx = base_x - float(robot.x)
        dy = base_y - float(robot.y)
        distance = math.hypot(dx, dy)
        if distance > float(self.view_distance):
            return None
        target_angle = np.degrees(np.arctan2(dy, dx)) % 360.0
        if not robot._angle_in_fov(target_angle, float(robot.angle) % 360.0, float(self.fov_angle)):
            return None
        if robot.line_of_sight_blocked(base_x, base_y, self.obstacles):
            return None

        range_noise = float(MEASUREMENT_NOISE[0])
        bearing_noise = float(MEASUREMENT_NOISE[1])
        r = distance + drone['rng'].normal(0.0, range_noise)
        true_bearing = (np.degrees(np.arctan2(dy, dx)) - float(robot.angle)) % 360.0
        if true_bearing > 180.0:
            true_bearing -= 360.0
        b = true_bearing + drone['rng'].normal(0.0, bearing_noise)
        return (r, b, base_x, base_y)

    def _home_marker(self):
        return Landmark(
            x=float(self.home_base['cx']),
            y=float(self.home_base['cy']),
            shape=str(HOME_BASE_MARKER_SHAPE),
            color=str(HOME_BASE_MARKER_COLOR),
            size=float(HOME_BASE_MARKER_SIZE),
        )

    def _home_return_goal_for_drone(self, drone):
        idx = int(drone.get('drone_index', 0))
        x, y, _heading = self._start_pose_for_index(idx, max(1, len(self.drones)))
        return float(x), float(y)

    def _drone_home_pose_error(self, drone):
        gx, gy = self._home_return_goal_for_drone(drone)
        x_est, y_est, _ = drone['odometry'].mu
        return math.hypot(float(gx) - float(x_est), float(gy) - float(y_est))

    def _drone_is_home(self, drone):
        robot = drone['robot']
        base_half_w = 0.5 * float(self.home_base['width']) - 0.2 * float(robot.size)
        base_half_h = 0.5 * float(self.home_base['height']) - 0.2 * float(robot.size)
        in_box = (
            abs(float(robot.x) - float(self.home_base['cx'])) <= max(0.0, base_half_w)
            and abs(float(robot.y) - float(self.home_base['cy'])) <= max(0.0, base_half_h)
        )
        return in_box and self._drone_home_pose_error(drone) <= max(float(RETURN_HOME_GOAL_TOLERANCE), 0.55)

    def _begin_return_phase(self):
        if self.return_phase_active:
            return
        self.return_phase_active = True
        self.return_finish_candidate_time = None
        self.auto_finish_candidate_time = None
        for drone in self.drones:
            drone['auto_phase'] = 'return'
            drone['auto_goal_xy'] = None
            drone['auto_goal_last_set_time'] = -1e9
            drone['auto_goal_meta'] = {'goal_flavor': 'return-home'}
            drone['path'] = []
            drone['current_target_index'] = 0
            drone['planned_path'] = []
            drone['plan_goal_index'] = -1
            drone['last_plan_time'] = -1e9
            drone['path_progress_index'] = 0
            drone['team_role'] = 'return'
            drone['target_marker'].set_data([], [])
            drone['subgoal_marker'].set_data([], [])
            drone['plan_line'].set_data([], [])
            self._update_mission_artist(drone)
        self._log_event('return_phase_started', 'Exploration complete; robots returning to home base')

    def _all_robots_home_and_idle(self):
        for drone in self.drones:
            if drone.get('auto_phase') not in ('return', 'returned'):
                return False
            if not self._drone_is_home(drone):
                return False
            if drone.get('auto_goal_xy') is not None:
                return False
            if drone.get('path') or drone.get('planned_path'):
                return False
            if drone.get('last_command_active', False):
                return False
        return True

    def _interrobot_measurements(self, drone):
        if not bool(INTERROBOT_USE_AS_LANDMARK):
            return []
        robot = drone['robot']
        measurements = []
        visible_names = set(drone.get('visible_teammates', []))
        for packet in self._valid_teammate_packets(drone):
            name = str(packet.get('name', ''))
            if name not in visible_names:
                continue
            ref_trace = float(packet.get('pose_trace', 1e9))
            if ref_trace > float(INTERROBOT_MAX_REFERENCE_TRACE):
                continue
            ref_pose = packet.get('pose', None)
            if ref_pose is None or len(ref_pose) < 2:
                continue
            other = next((d for d in self.drones if d.get('name') == name), None)
            if other is None:
                continue

            dx = float(other['robot'].x) - float(robot.x)
            dy = float(other['robot'].y) - float(robot.y)
            r = np.hypot(dx, dy) + drone['rng'].normal(0.0, float(INTERROBOT_MEASUREMENT_NOISE[0]))
            true_bearing = (np.degrees(np.arctan2(dy, dx)) - float(robot.angle)) % 360.0
            if true_bearing > 180.0:
                true_bearing -= 360.0
            b = true_bearing + drone['rng'].normal(0.0, float(INTERROBOT_MEASUREMENT_NOISE[1]))

            ref_x = float(ref_pose[0])
            ref_y = float(ref_pose[1])
            ref_cov_xy = self._packet_xy_cov(packet)
            reliability = float(packet.get('reliability', 1.0))
            sensor_R = np.diag([
                float(INTERROBOT_MEASUREMENT_NOISE[0]) ** 2,
                float(INTERROBOT_MEASUREMENT_NOISE[1]) ** 2,
            ])
            measurements.append(((r, b, ref_x, ref_y, ref_cov_xy), sensor_R, reliability, name))
        return measurements

    def _update_los_and_packets(self):
        self.shared_los_segments = []
        for drone in getattr(self, 'drones', []):
            drone['visible_teammates'] = []
            drone['visible_segments_est'] = []
        for i, a in enumerate(getattr(self, 'drones', [])):
            for b in self.drones[i + 1:]:
                ra = a['robot']
                rb = b['robot']
                if math.hypot(float(ra.x) - float(rb.x), float(ra.y) - float(rb.y)) > float(COMMUNICATION_RADIUS):
                    continue
                if ra.line_of_sight_blocked(rb.x, rb.y, self.obstacles):
                    continue
                ax, ay = self._estimated_xy(a)
                bx, by = self._estimated_xy(b)
                seg = [(ax, ay), (bx, by)]
                self.shared_los_segments.append(seg)
                a['visible_teammates'].append(b['name'])
                b['visible_teammates'].append(a['name'])
                a['visible_segments_est'].append(seg)
                b['visible_segments_est'].append(seg)
                self._exchange_packet(a, b)
                self._exchange_packet(b, a)

    @staticmethod
    def _wrap_angle_deg(angle_deg):
        angle = float(angle_deg)
        while angle > 180.0:
            angle -= 360.0
        while angle <= -180.0:
            angle += 360.0
        return angle

    def _record_recent_scan(self, drone, robot, scan):
        drone.setdefault('recent_scan_buffer', deque(maxlen=MAP_SCAN_BUFFER_SIZE)).append(
            scan_buffer_entry_from_scan(robot, scan, drone['odometry'].mu, drone['odometry'].cov, self.time_elapsed)
        )

    def _should_replay_recent_scans(self, pre_mu, pre_cov, post_mu, post_cov):
        pre_trace = float(np.trace(pre_cov[:2, :2]))
        post_trace = float(np.trace(post_cov[:2, :2]))
        trace_gain = pre_trace - post_trace
        pos_shift = math.hypot(float(post_mu[0]) - float(pre_mu[0]), float(post_mu[1]) - float(pre_mu[1]))
        heading_shift = abs(self._wrap_angle_deg(float(post_mu[2]) - float(pre_mu[2])))
        return bool(
            trace_gain >= MAP_REPLAY_MIN_TRACE_IMPROVEMENT
            or pos_shift >= MAP_REPLAY_MIN_POSITION_SHIFT
            or heading_shift >= MAP_REPLAY_MIN_HEADING_SHIFT_DEG
        )

    def _replay_recent_scans(self, drone, delta_pose):
        buffer = list(drone.get('recent_scan_buffer', []))
        if not buffer:
            return
        replay_limit = max(1, int(MAP_REPLAY_MAX_SCANS))
        if len(buffer) > replay_limit:
            buffer = buffer[-replay_limit:]
        drone['recent_scan_buffer'] = deque(
            (shifted_scan_buffer_entry(entry, delta_pose) for entry in buffer),
            maxlen=MAP_SCAN_BUFFER_SIZE,
        )
        start_x, start_y, _ = self._start_pose_for_index(drone.get('drone_index', 0), len(self.drones))
        drone['local_known_grid'] = self._init_known_grid()
        drone['local_logodds_grid'], drone['local_confidence_grid'] = self._init_belief_layers(drone['local_known_grid'])
        drone['local_pose_uncertainty_grid'] = self._init_pose_uncertainty_grid(drone['local_known_grid'])
        self._mark_local_uncertainty_dirty(drone)
        self._reveal_start_area_with_belief(
            drone['local_known_grid'],
            drone['local_logodds_grid'],
            drone['local_confidence_grid'],
            drone['local_pose_uncertainty_grid'],
            start_x,
            start_y,
        )
        for entry in drone['recent_scan_buffer']:
            pose_x, pose_y, pose_theta = (float(v) for v in entry['pose'])
            pose_proxy = type('PoseProxy', (), {'x': pose_x, 'y': pose_y, 'angle': pose_theta})()
            pose_cov = np.diag([max(0.02, float(entry.get('pose_trace', 0.0)) / 2.0)] * 2 + [1.0])
            replay_scan = scan_from_buffer_entry(entry)
            self._apply_scan_to_grid(
                drone['local_known_grid'],
                drone['local_logodds_grid'],
                drone['local_confidence_grid'],
                drone['local_pose_uncertainty_grid'],
                pose_proxy,
                replay_scan,
                pose_cov,
            )
        drone['local_known_occ_count'] = int(np.count_nonzero(drone['local_known_grid'] == OCCUPIED))
        self._merge_local_into_shared_if_better(drone)
        self._update_local_grid_visual(drone)

    def _merge_local_into_shared_if_better(self, drone):
        local_unc = drone['local_pose_uncertainty_grid']
        shared_unc = self.shared_pose_uncertainty_grid
        local_conf = drone['local_confidence_grid']
        shared_conf = self.shared_confidence_grid
        local_finite = np.isfinite(local_unc)
        shared_finite = np.isfinite(shared_unc)
        lower_unc = local_finite & (~shared_finite | (local_unc + 1e-6 < shared_unc))
        same_unc = np.zeros_like(local_finite, dtype=bool)
        finite_both = local_finite & shared_finite
        same_unc[finite_both] = np.abs(local_unc[finite_both] - shared_unc[finite_both]) <= 1e-6
        tie_break_conf = same_unc & (local_conf > shared_conf)
        better_mask = lower_unc | tie_break_conf
        if not np.any(better_mask):
            return
        self.shared_logodds_grid[better_mask] = drone['local_logodds_grid'][better_mask]
        self.shared_confidence_grid[better_mask] = local_conf[better_mask]
        self.shared_pose_uncertainty_grid[better_mask] = local_unc[better_mask]
        self.shared_known_grid[better_mask] = drone['local_known_grid'][better_mask]
        self._invalidate_planning_cache()
        self._mark_partition_dirty()
        self._mark_shared_uncertainty_dirty()

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

    def _belief_uncertainty_rgba(self, logodds_grid, pose_uncertainty_grid, known_grid):
        known = np.asarray(known_grid) != UNKNOWN
        pose_unc = np.asarray(pose_uncertainty_grid, dtype=float)
        finite_vals = pose_unc[np.isfinite(pose_unc) & known]
        if finite_vals.size:
            robust_scale = max(1e-6, float(np.percentile(finite_vals, 90)))
        else:
            robust_scale = max(1e-6, float(MAP_UNCERTAINTY_VIS_METERS_SCALE))
        robust_scale = max(robust_scale, float(MAP_UNCERTAINTY_VIS_METERS_SCALE))
        pose_term = np.clip(np.where(np.isfinite(pose_unc), pose_unc / robust_scale, 0.0), 0.0, 1.0)
        logodds = np.clip(np.asarray(logodds_grid, dtype=float), -60.0, 60.0)
        probs = 1.0 / (1.0 + np.exp(-logodds))
        probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
        entropy = -(probs * np.log(probs) + (1.0 - probs) * np.log(1.0 - probs)) / math.log(2.0)
        combined = np.clip(0.82 * pose_term + 0.18 * entropy, 0.0, 1.0)
        rgba = plt.cm.turbo(combined)
        rgba = np.asarray(rgba, dtype=float)
        rgba[..., 3] = (0.18 + 0.58 * combined) * known.astype(float)
        rgba[~known, 3] = 0.0
        return rgba

    def _shared_uncertainty_rgba(self, force=False):
        if force or self._shared_uncertainty_dirty:
            self.uncertainty_rgba = self._belief_uncertainty_rgba(self.shared_logodds_grid, self.shared_pose_uncertainty_grid, self.shared_known_grid)
            self._shared_uncertainty_dirty = False
        return self.uncertainty_rgba

    def _local_uncertainty_rgba(self, drone, force=False):
        if force or drone.get('local_uncertainty_dirty', True):
            drone['local_uncertainty_rgba_cache'] = self._belief_uncertainty_rgba(drone['local_logodds_grid'], drone['local_pose_uncertainty_grid'], drone['local_known_grid'])
            drone['local_uncertainty_dirty'] = False
        return drone['local_uncertainty_rgba_cache']

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


    def _estimated_team_centroid(self):
        pts = []
        for drone in self.drones:
            try:
                x_est, y_est, _ = drone['odometry'].mu
                pts.append((float(x_est), float(y_est)))
            except Exception:
                continue
        if not pts:
            starts = [self._start_pose_for_index(i, len(self.drones))[:2] for i in range(len(self.drones))]
            pts = [(float(x), float(y)) for x, y in starts]
        arr = np.asarray(pts, dtype=float)
        return np.mean(arr, axis=0)

    def _mean_pairwise_estimated_distance(self):
        pts = []
        for drone in self.drones:
            try:
                x_est, y_est, _ = drone['odometry'].mu
                pts.append((float(x_est), float(y_est)))
            except Exception:
                continue
        if len(pts) < 2:
            return 0.0
        total = 0.0
        count = 0
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                total += math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])
                count += 1
        return total / max(1, count)

    def _startup_dispersion_active(self):
        if not AUTO_LAUNCH_DISPERSAL_ENABLED:
            return False
        if self.time_elapsed >= float(AUTO_LAUNCH_DISPERSAL_MAX_SECONDS):
            return False
        return self._mean_pairwise_estimated_distance() < float(AUTO_STARTUP_DISPERSION_MEAN_PAIRWISE_DISTANCE)

    def _partition_penalty_scale(self):
        return float(AUTO_STARTUP_PARTITION_PENALTY_SCALE) if self._startup_dispersion_active() else 1.0

    def _launch_phase_expired(self, drone):
        return (self.time_elapsed >= float(AUTO_LAUNCH_DISPERSAL_MAX_SECONDS)) or (not self._startup_dispersion_active())

    def _teammate_context(self, drone):
        teammate_positions = []
        teammate_goal_positions = []
        teammate_path_histories = []
        for packet in self._valid_teammate_packets(drone):
            reliability = float(packet.get('reliability', 1.0))
            pose = packet.get('pose', None)
            if pose is not None and len(pose) >= 2:
                teammate_positions.append((float(pose[0]), float(pose[1]), reliability))
            goal = packet.get('goal', None)
            role = str(packet.get('role', 'explorer')).lower()
            if goal is not None and role in {'explorer', 'explore', 'launch'}:
                teammate_goal_positions.append((float(goal[0]), float(goal[1]), reliability))
            hist = packet.get('past_waypoints') or packet.get('path_history')
            if hist:
                teammate_path_histories.append({
                    'points': [(float(px), float(py)) for px, py in hist],
                    'weight': reliability,
                    'timestamp': float(packet.get('timestamp', self.time_elapsed)),
                })
        return teammate_positions, teammate_goal_positions, teammate_path_histories

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
        if self.mission_mode == 'auto_explore' and drone.get('auto_phase', 'explore') == 'launch':
            goal_tol = AUTO_LAUNCH_DISPERSAL_GOAL_TOLERANCE
        elif self.mission_mode == 'auto_explore' and drone.get('auto_phase', 'explore') == 'return':
            goal_tol = RETURN_HOME_GOAL_TOLERANCE
        else:
            goal_tol = A_STAR_GOAL_TOLERANCE
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
        teammate_positions, teammate_goal_positions, teammate_path_histories = self._teammate_context(drone)
        centroid_xy = self._auto_centroid_for_drone(drone)
        common = dict(
            robot_index=drone['drone_index'],
            partition_labels=self.partition_labels,
            robot_xy=(x_est, y_est),
            centroid_xy=centroid_xy,
            teammate_positions=teammate_positions,
            teammate_goal_positions=teammate_goal_positions,
            own_recent_path=self._recent_estimated_path(drone),
            teammate_path_histories=teammate_path_histories,
            partition_penalty_scale=self._partition_penalty_scale(),
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

    def _remaining_frontier_mass(self):
        total = 0
        for comp in self.frontier_components:
            if isinstance(comp, dict):
                total += int(comp.get('size', len(comp.get('cells', comp.get('points', [])))))
            else:
                total += int(len(comp))
        return total

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
        if self._remaining_frontier_mass() <= int(AUTO_FINISH_MIN_TOTAL_FRONTIER_CELLS):
            return True
        return not self._any_assignable_frontier_goal()

    def _maybe_finish_auto_explore(self):
        if not self.auto_mode or self.mission_mode != 'auto_explore' or not AUTO_STOP_WHEN_FINISHED:
            self.auto_finish_candidate_time = None
            self.return_finish_candidate_time = None
            return
        if not self.return_phase_active:
            if not self._auto_explore_done_now():
                self.auto_finish_candidate_time = None
                return
            if self.auto_finish_candidate_time is None:
                self.auto_finish_candidate_time = self.time_elapsed
                return
            if (self.time_elapsed - self.auto_finish_candidate_time) < float(AUTO_FINISH_HOLD_SECONDS):
                return
            self._begin_return_phase()
            return
        if not self._all_robots_home_and_idle():
            self.return_finish_candidate_time = None
            return
        if self.return_finish_candidate_time is None:
            self.return_finish_candidate_time = self.time_elapsed
            return
        if (self.time_elapsed - self.return_finish_candidate_time) < float(RETURN_HOME_FINISH_HOLD_SECONDS):
            return
        self.auto_mode = False
        self.auto_finished = True
        self.auto_finish_candidate_time = None
        self.return_finish_candidate_time = None
        for drone in self.drones:
            drone['robot'].set_motor_speeds(0.0, 0.0)
            drone['auto_phase'] = 'returned'
            drone['team_role'] = 'home'
        self._refresh_toggle_button()
        self._refresh_control_text()
        self._log_event('auto_finished', 'Exploration and return-to-home completed')
        self._finalize_current_run()

    def _choose_launch_staging_goal(self, drone):
        if not AUTO_LAUNCH_DISPERSAL_ENABLED:
            return None
        start_x, start_y, start_heading = self._start_pose_for_index(int(drone.get('drone_index', 0)), max(1, len(self.drones)))
        heading = math.radians(start_heading)
        planning_occ = self._planning_occupancy(self.shared_known_grid, exclude_drone=drone)
        x_est, y_est, _ = drone['odometry'].mu
        nominal = max(float(HOME_BASE_STAGING_DISTANCE), float(AUTO_LAUNCH_DISPERSAL_DISTANCE), 1.4 * A_STAR_GOAL_TOLERANCE)
        candidate_xy = []
        for dist_scale in (1.0, 0.85, 0.7):
            dist = nominal * dist_scale
            for ang_offset in (0.0, -0.18, 0.18, -0.36, 0.36):
                ang = heading + ang_offset
                cand = np.array([start_x + dist * math.cos(ang), start_y + dist * math.sin(ang)], dtype=float)
                cand[0] = float(np.clip(cand[0], 0.8, WORLD_WIDTH_METERS - 0.8))
                cand[1] = float(np.clip(cand[1], 0.8, WORLD_HEIGHT_METERS - 0.8))
                candidate_xy.append((float(cand[0]), float(cand[1])))
        seen = set()
        for cand in candidate_xy:
            if cand in seen:
                continue
            seen.add(cand)
            cell = self.planner.nearest_free_cell(self._world_to_grid(*cand), planning_occ, clearance_cells=A_STAR_GOAL_CLEARANCE_CELLS)
            goal_xy = self._grid_to_world(cell)
            if math.hypot(goal_xy[0] - x_est, goal_xy[1] - y_est) < 1.25 * A_STAR_GOAL_TOLERANCE:
                continue
            path = self._astar((x_est, y_est), goal_xy, self.shared_known_grid, exclude_drone=drone)
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

        phase = drone.get('auto_phase', 'launch')
        if phase == 'returned':
            drone['auto_goal_xy'] = None
            drone['path'] = []
            return
        if phase == 'return':
            return_goal = self._home_return_goal_for_drone(drone)
            if self._should_keep_current_auto_goal(drone, return_goal, {'goal_flavor': 'return-home'}):
                return
            self._assign_auto_goal(drone, return_goal, meta={'goal_flavor': 'return-home'})
            drone['team_role'] = 'return'
            return

        if phase == 'launch':
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

    def _astar(self, start_xy, goal_xy, known_grid, exclude_drone=None):
        planning_occ = self._planning_occupancy(known_grid, exclude_drone=exclude_drone)
        return self.planner.astar_on_occupancy(start_xy, goal_xy, planning_occ, goal_clearance_cells=A_STAR_GOAL_CLEARANCE_CELLS)

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
        planning_occ = self._planning_occupancy(known, exclude_drone=drone)
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
            planner_world_to_grid=self._world_to_grid,
        )
        drone['rng'] = np.random.default_rng(self._seed_for_drone(drone_index, stream=2))
        drone['min_clearance'] = self.view_distance
        drone['local_logodds_grid'], drone['local_confidence_grid'] = self._init_belief_layers(drone['local_known_grid'])
        drone['local_pose_uncertainty_grid'] = self._init_pose_uncertainty_grid(drone['local_known_grid'])
        drone['recent_scan_buffer'] = deque(maxlen=MAP_SCAN_BUFFER_SIZE)
        drone['last_pose_trace'] = float(np.trace(drone['odometry'].cov[:2, :2]))
        drone['teammate_memory'] = {}
        drone['visible_teammates'] = []
        drone['visible_segments_est'] = []
        drone['local_uncertainty_rgba_cache'] = np.zeros((self.ny, self.nx, 4), dtype=float)
        drone['local_uncertainty_dirty'] = True
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
        self._load_environment(seed)
        self._prepare_for_new_run()
        self.auto_mode = False
        self._refresh_toggle_button()
        self._refresh_control_text()
        for idx, drone in enumerate(self.drones):
            new_path = list(self.generated_target_sequences[idx]) if idx < len(self.generated_target_sequences) else []
            drone['manual_path'] = new_path
            drone['path'] = list(new_path) if self.mission_mode == 'manual_click' else []
            drone['auto_goal_xy'] = None
            drone['auto_goal_last_set_time'] = -1e9
            drone['auto_goal_meta'] = None
            drone['auto_phase'] = 'launch'
            drone['team_role'] = 'launch'
        self.reset_simulation()
        self._mark_partition_dirty()
        self._refresh_partition_state(force=True)
        self._update_los_and_packets()
        self.ui.refresh_shared_map()

    def toggle_auto_mode(self, event=None):
        self.auto_mode = not self.auto_mode
        if self.auto_mode:
            self.auto_finished = False
            self.auto_finish_candidate_time = None
            self.return_phase_active = False
            self.return_finish_candidate_time = None
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
                drone['team_role'] = 'launch'
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
        self.shared_logodds_grid, self.shared_confidence_grid = self._init_belief_layers(self.shared_known_grid)
        self.shared_pose_uncertainty_grid = self._init_pose_uncertainty_grid(self.shared_known_grid)
        self._mark_shared_uncertainty_dirty()
        self.shared_known_landmarks = {}
        self._ui_frame_counter = 0

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
            drone['local_logodds_grid'], drone['local_confidence_grid'] = self._init_belief_layers(drone['local_known_grid'])
            drone['local_pose_uncertainty_grid'] = self._init_pose_uncertainty_grid(drone['local_known_grid'])
            self._mark_local_uncertainty_dirty(drone)
            drone['recent_scan_buffer'] = deque(maxlen=MAP_SCAN_BUFFER_SIZE)
            self._reveal_start_area_with_belief(drone['local_known_grid'], drone['local_logodds_grid'], drone['local_confidence_grid'], drone['local_pose_uncertainty_grid'], start_x, start_y)
            self._reveal_start_area_with_belief(self.shared_known_grid, self.shared_logodds_grid, self.shared_confidence_grid, self.shared_pose_uncertainty_grid, start_x, start_y)
            drone['known_occ_count'] = int(np.count_nonzero(self.shared_known_grid == OCCUPIED))
            drone['local_known_occ_count'] = int(np.count_nonzero(drone['local_known_grid'] == OCCUPIED))
            drone['just_discovered_obstacle'] = False
            drone['recent_positions'] = [(0.0, *self._estimated_xy(drone))]
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
            drone['known_landmarks'] = {}
            drone['last_detected_landmarks'] = []
            drone['distance_travelled'] = 0.0
            drone['idle_time'] = 0.0
            drone['last_goal_type'] = 'none'
            drone['visited_cells'] = {self._estimated_grid_cell(drone)}
            drone['last_pose_trace'] = float(np.trace(drone['odometry'].cov[:2, :2]))
            drone['teammate_memory'] = {}
            drone['visible_teammates'] = []
            drone['visible_segments_est'] = []

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

        self._mark_partition_dirty()
        self._refresh_partition_state(force=True)
        self._update_los_and_packets()
        self.ui.refresh_shared_map()
        self._record_coverage_snapshot(force=True)
        self._log_event('run_reset', 'Simulation reset to initial conditions', seed=int(self.current_seed))

    def _apply_scan_to_grid(self, grid, logodds_grid, confidence_grid, pose_uncertainty_grid, robot, scan, pose_cov):
        return apply_scan_to_grid(
            grid,
            logodds_grid,
            confidence_grid,
            pose_uncertainty_grid,
            robot,
            scan,
            self._world_to_grid,
            self._stamp_obstacle_hit,
            self.grid_resolution,
            self.view_distance,
            UNKNOWN,
            FREE,
            OCCUPIED,
            pose_cov,
        )

    def _update_known_map_from_scan(self, drone, scan):
        changed = update_known_map_from_scan(
            drone,
            self.shared_known_grid,
            self.shared_logodds_grid,
            self.shared_confidence_grid,
            self.shared_pose_uncertainty_grid,
            scan,
            self._apply_scan_to_grid,
            OCCUPIED,
        )
        if changed:
            self._invalidate_planning_cache()
            self._mark_local_uncertainty_dirty(drone)
            self._mark_shared_uncertainty_dirty()
            self._mark_partition_dirty()

    def _clearance_groups(self, scan):
        return clearance_groups(scan)

    def _path_blocked(self, drone, planning_occ, max_segments=8):
        path = list(drone.get('planned_path', []))
        if len(path) < 2:
            return False
        prog = max(0, min(int(drone.get('path_progress_index', 0)), len(path) - 1))
        x_est, y_est, _ = drone['odometry'].mu
        prev = (float(x_est), float(y_est))
        checked = 0
        for pt in path[prog:]:
            seg_end = (float(pt[0]), float(pt[1]))
            if self._line_crosses_blocked(prev, seg_end, planning_occ):
                return True
            prev = seg_end
            checked += 1
            if checked >= max_segments:
                break
        return False

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
        planning_occ = self._planning_occupancy(self.shared_known_grid, exclude_drone=drone)
        plan_age = float(self.time_elapsed - drone['last_plan_time'])
        goal_changed = drone['plan_goal_index'] != target_idx
        missing_path = not drone['planned_path']

        goal_gx, goal_gy = self._world_to_grid(*goal)
        goal_blocked = bool(planning_occ[goal_gy, goal_gx])
        path_blocked = self._path_blocked(drone, planning_occ) if KNOWN_MAP_REPLAN_ON_NEW_OBS else False
        allow_blocked_refresh = (goal_blocked or path_blocked) and plan_age >= max(A_STAR_MIN_REPLAN_GAP_SECONDS, 1.5)

        need_replan = goal_changed or missing_path or allow_blocked_refresh

        if need_replan:
            old_path = list(drone['planned_path'])
            new_path = self._astar((x_est, y_est), goal, self.shared_known_grid, exclude_drone=drone)
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
                elif path_blocked:
                    reason = 'path blocked'
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
        planning_occ = self._planning_occupancy(self.shared_known_grid, exclude_drone=drone)

        while prog < len(path_pts) - 1 and math.hypot(path_pts[prog][0] - x_est, path_pts[prog][1] - y_est) < A_STAR_GOAL_TOLERANCE:
            prog += 1

        visible_candidates = []
        max_idx = min(len(path_pts) - 1, prog + max(1, A_STAR_LOOKAHEAD_STEPS))
        for j in range(prog, max_idx + 1):
            if self._line_crosses_blocked((x_est, y_est), path_pts[j], planning_occ):
                break
            clearance_m = self.planner.clearance_distance_world(path_pts[j], planning_occ)
            visible_candidates.append((j, clearance_m, path_pts[j]))

        drone['path_progress_index'] = prog
        if not visible_candidates:
            return None

        safe_candidates = [item for item in visible_candidates if item[1] >= float(SUBGOAL_OBSTACLE_CLEARANCE_METERS)]
        if safe_candidates:
            safe_candidates.sort(key=lambda item: (item[0] + SUBGOAL_CLEARANCE_PREFERENCE * item[1], item[1]), reverse=True)
            return safe_candidates[0][2]

        visible_candidates.sort(key=lambda item: (item[1], item[0]), reverse=True)
        return visible_candidates[0][2]

    def _update_progress_history(self, drone):
        hist = drone['recent_positions']
        hist.append((self.time_elapsed, *self._estimated_xy(drone)))
        cutoff = self.time_elapsed - STUCK_WINDOW_SECONDS
        while len(hist) > 2 and hist[1][0] < cutoff:
            hist.pop(0)

    def _should_trigger_recovery(self, drone):
        if self.time_elapsed < drone['recovery_until']:
            return False
        if not drone['last_command_active']:
            return False
        emergency_clearance = max(0.55, 1.25 * (float(drone['robot'].size) / 2.0))
        if drone['min_clearance'] > emergency_clearance:
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
        self._record_recent_scan(drone, robot, scan)
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

        if self.mission_mode == 'auto_explore' and drone.get('auto_phase', 'explore') == 'launch':
            goal_tol = AUTO_LAUNCH_DISPERSAL_GOAL_TOLERANCE
        elif self.mission_mode == 'auto_explore' and drone.get('auto_phase', 'explore') == 'return':
            goal_tol = RETURN_HOME_GOAL_TOLERANCE
        else:
            goal_tol = A_STAR_GOAL_TOLERANCE

        if dist_goal < goal_tol:
            drone['goal_reached_count'] = int(drone.get('goal_reached_count', 0)) + 1
            if self.mission_mode == 'auto_explore':
                reached_goal = drone.get('auto_goal_xy')
                phase = drone.get('auto_phase', 'explore')
                if phase == 'launch':
                    drone['auto_phase'] = 'explore'
                    drone['team_role'] = 'explorer'
                elif phase == 'return':
                    drone['auto_phase'] = 'returned'
                    drone['team_role'] = 'home'
                drone['auto_goal_xy'] = None
                drone['auto_goal_last_set_time'] = self.time_elapsed
                drone['path'] = []
                drone['current_target_index'] = 0
                drone['target_marker'].set_data([], [])
                drone['subgoal_marker'].set_data([], [])
                drone['planned_path'] = []
                drone['plan_line'].set_data([], [])
                if reached_goal is not None:
                    event_name = 'returned_home' if phase == 'return' else 'goal_reached'
                    label = 'home base' if phase == 'return' else drone.get('last_goal_type', 'goal')
                    self._log_event(event_name, f"Reached {label} at ({reached_goal[0]:.2f}, {reached_goal[1]:.2f})", drone=drone)
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

        # Let A* and clearance-aware goal/subgoal selection handle obstacle avoidance.
        # The controller only slows or stops on immediate collision risk instead of
        # creating a long-range reactive "soft wall" around obstacles.
        emergency_clearance = max(0.55, 1.25 * (float(robot.size) / 2.0))
        caution_clearance = max(0.8, emergency_clearance + 0.25)
        if min_center < emergency_clearance:
            v = 0.0
        elif min_center < caution_clearance:
            v *= max(0.45, min_center / max(caution_clearance, 1e-6))

        for other in self.drones:
            if other is drone:
                continue
            ox = float(other['robot'].x) - float(robot.x)
            oy = float(other['robot'].y) - float(robot.y)
            dist = math.hypot(ox, oy)
            if dist < 1e-6 or dist > ROBOT_AVOIDANCE_RADIUS:
                continue
            rel_angle = self._wrap_angle_deg(math.degrees(math.atan2(oy, ox)) - theta_est)
            if abs(rel_angle) <= 95.0:
                v *= max(0.15, dist / max(ROBOT_AVOIDANCE_RADIUS, 1e-6))
                omega += -math.copysign(ROBOT_AVOIDANCE_GAIN * (1.0 - dist / max(ROBOT_AVOIDANCE_RADIUS, 1e-6)), rel_angle)

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
            other_robots = [other['robot'] for other in self.drones if other is not drone]
            robot.update(dt, obstacles=self.obstacles, other_robots=other_robots)
            drone['distance_travelled'] = float(drone.get('distance_travelled', 0.0)) + math.hypot(float(robot.x) - prev_x, float(robot.y) - prev_y)
            drone.setdefault('visited_cells', set()).add(self._estimated_grid_cell(drone))
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
        new_shared_landmarks = self._remember_landmarks(drone, detected)
        for key in new_shared_landmarks:
            row = self.shared_known_landmarks.get(key, {})
            self._log_event(
                'landmark_discovered',
                f"Discovered {row.get('color_name', '')} {row.get('shape', '')} landmark at ({float(row.get('x', 0.0)):.1f}, {float(row.get('y', 0.0)):.1f})",
                drone=drone,
                landmark_shape=row.get('shape', ''),
                landmark_color=row.get('color_name', ''),
                landmark_x=row.get('x', 0.0),
                landmark_y=row.get('y', 0.0),
                landmark_total=len(self.shared_known_landmarks),
            )
        measurements = []
        home_base_measurement = self._home_base_measurement(drone)
        if home_base_measurement is not None:
            measurements.append(home_base_measurement)

        home_marker_key = self._landmark_key(self._home_marker())
        for lm in detected:
            if self._landmark_key(lm) == home_marker_key:
                continue
            dx = lm.x - robot.x
            dy = lm.y - robot.y
            r = np.hypot(dx, dy) + drone['rng'].normal(0.0, MEASUREMENT_NOISE[0])

            true_bearing = (np.degrees(np.arctan2(dy, dx)) - robot.angle) % 360.0
            if true_bearing > 180.0:
                true_bearing -= 360.0
            b = true_bearing + drone['rng'].normal(0.0, MEASUREMENT_NOISE[1])
            measurements.append((r, b, lm.x, lm.y))

        teammate_measurements = self._interrobot_measurements(drone)
        if dt > 0.0 and (measurements or teammate_measurements):
            pre_correct_mu = np.array(odometry.mu, dtype=float)
            pre_correct_cov = np.array(odometry.cov, dtype=float)
            if measurements:
                odometry.correct(
                    measurements,
                    np.diag([MEASUREMENT_NOISE[0] ** 2, MEASUREMENT_NOISE[1] ** 2]),
                    alpha=MEASUREMENT_ALPHA,
                )
                drone['measurement_update_count'] = int(drone.get('measurement_update_count', 0)) + len(measurements)
                drone['last_landmark_update_time'] = float(self.time_elapsed)
            for teammate_meas, teammate_R, teammate_alpha, _teammate_name in teammate_measurements:
                odometry.correct_with_uncertain_landmarks([teammate_meas], teammate_R, alpha=teammate_alpha)
                drone['measurement_update_count'] = int(drone.get('measurement_update_count', 0)) + 1
            if ENABLE_RECENT_SCAN_REPLAY and self._should_replay_recent_scans(pre_correct_mu, pre_correct_cov, odometry.mu, odometry.cov):
                delta_pose = (
                    float(odometry.mu[0] - pre_correct_mu[0]),
                    float(odometry.mu[1] - pre_correct_mu[1]),
                    self._wrap_angle_deg(float(odometry.mu[2] - pre_correct_mu[2])),
                )
                self._replay_recent_scans(drone, delta_pose)

        est_x, est_y, est_theta = odometry.mu
        drone['true_patch'].set_xy(self.robot_shape_from_pose(robot.x, robot.y, robot.angle, robot.size))
        drone['est_patch'].set_xy(self.robot_shape_from_pose(est_x, est_y, est_theta, robot.size * 0.92))
        update_uncertainty_ellipse_patch(drone['ellipse_patch'], odometry, n_std=2.5)
        update_fov_patch(drone['fov_patch'], robot.x, robot.y, robot.angle, view_distance=self.view_distance, fov_angle=self.fov_angle)
        self._update_local_grid_visual(drone)

        return detected

    def update(self, frame):
        if isinstance(getattr(self, '_planning_occ_cache_frame', None), dict):
            self._planning_occ_cache_frame.clear()
        dt = TIME_STEP if self.auto_mode else 0.0
        if dt > 0.0:
            self.time_elapsed += dt
    
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

        self._update_los_and_packets()
        if dt > 0.0:
            self.trace_counter += 1
            self._mark_partition_dirty()
        self._refresh_partition_state(force=False)
        self._record_coverage_snapshot(force=(self.time_elapsed <= 1e-9))
        if dt > 0.0 and self.auto_mode:
            self._maybe_print_live_snapshot(force=False)
        self._maybe_finish_auto_explore()
        self._ui_frame_counter = int(getattr(self, '_ui_frame_counter', 0)) + 1
        shared_refresh = (not self.auto_mode) or (self._ui_frame_counter % max(1, int(UI_SHARED_REFRESH_EVERY_FRAMES)) == 0)
        heavy_refresh = (not self.auto_mode) or (self._ui_frame_counter % max(1, int(UI_HEAVY_REFRESH_EVERY_FRAMES)) == 0)
        overlay_refresh = (not self.auto_mode) or (self._ui_frame_counter % max(1, int(UI_OVERLAY_REFRESH_EVERY_FRAMES)) == 0)
        monitor_refresh = (not self.auto_mode) or (self._ui_frame_counter % max(1, int(UI_MONITOR_REFRESH_EVERY_FRAMES)) == 0)
        self.ui.refresh_status_text()
        if shared_refresh:
            self.ui.refresh_shared_map()
        if heavy_refresh and overlay_refresh:
            self.ui.refresh_partition_overlay()
        if heavy_refresh and monitor_refresh:
            self.ui.refresh_robot_monitor()
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

if __name__ == '__main__':
    Simulator().run()
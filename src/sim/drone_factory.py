import matplotlib.patheffects as pe
import matplotlib.patches as patches
import numpy as np

from config import (
    PATH_LINE_WIDTH,
    RAY_ALPHA,
    RAY_LINE_WIDTH,
    SHOW_VISION_RAYS,
    SUBGOAL_MARKER_SIZE,
    TARGET_MARKER_SIZE,
    VISION_RAY_COUNT,
)
from ..localization import OdometryEstimator
from ..robot import Robot
from .rendering import create_uncertainty_ellipse, make_fov_patch, robot_shape_from_pose


def create_drone(ax, name, color, path, drone_index, start_pose, robot_seed_rng, planner_init_known_grid):
    start_x, start_y, start_angle = start_pose
    robot = Robot(x=start_x, y=start_y, angle=start_angle, rng=robot_seed_rng)
    odometry = OdometryEstimator(
        init_pos=[start_x, start_y, start_angle],
        init_cov=np.diag([0.1, 0.1, 1.0]),
    )

    true_patch = patches.Polygon(
        robot_shape_from_pose(start_x, start_y, start_angle, robot.size),
        facecolor=color,
        edgecolor='black',
        linewidth=1.2,
        zorder=10,
    )
    ax.add_patch(true_patch)

    est_patch = patches.Polygon(
        robot_shape_from_pose(start_x, start_y, start_angle, robot.size * 0.92),
        facecolor='none',
        edgecolor=color,
        linewidth=2.0,
        linestyle='--',
        zorder=11,
    )
    ax.add_patch(est_patch)

    ellipse_patch = create_uncertainty_ellipse(
        ax,
        odometry,
        n_std=2.5,
        edgecolor=color,
        facecolor='none',
        linewidth=1.4,
        linestyle=':',
        alpha=0.9,
    )

    fov_patch = make_fov_patch(start_x, start_y, start_angle, color)
    ax.add_patch(fov_patch)

    trace_line, = ax.plot([start_x], [start_y], color=color, linewidth=1.2, alpha=0.75, zorder=2)
    est_trace_line, = ax.plot([start_x], [start_y], color=color, linewidth=1.2, linestyle='--', alpha=0.55, zorder=2)
    mission_line, = ax.plot(
        [pt[0] for pt in path],
        [pt[1] for pt in path],
        marker='x',
        linestyle=':',
        linewidth=1.3,
        alpha=0.45,
        color=color,
        zorder=4,
    )
    plan_line, = ax.plot([start_x], [start_y], color=color, linewidth=PATH_LINE_WIDTH, linestyle='-', alpha=0.98, zorder=6)
    plan_line.set_path_effects([
        pe.Stroke(linewidth=PATH_LINE_WIDTH + 2.2, foreground='black', alpha=0.35),
        pe.Normal(),
    ])

    target_marker, = ax.plot(
        [path[0][0]] if path else [],
        [path[0][1]] if path else [],
        marker='*',
        markersize=TARGET_MARKER_SIZE,
        color=color,
        mec='black',
        mew=1.1,
        zorder=8,
    )
    subgoal_marker, = ax.plot(
        [start_x] if path else [],
        [start_y] if path else [],
        marker='o',
        markersize=SUBGOAL_MARKER_SIZE,
        color=color,
        mec='white',
        mew=1.2,
        zorder=8,
    )

    ray_lines = []
    for _ in range(VISION_RAY_COUNT if SHOW_VISION_RAYS else 0):
        line, = ax.plot([start_x, start_x], [start_y, start_y], color=color, alpha=RAY_ALPHA, linewidth=RAY_LINE_WIDTH, zorder=5)
        ray_lines.append(line)

    return {
        'name': name,
        'drone_index': drone_index,
        'color': color,
        'manual_path': list(path),
        'path': list(path),
        'current_target_index': 0,
        'auto_goal_xy': None,
        'auto_goal_last_set_time': -1e9,
        'auto_goal_meta': None,
        'auto_phase': 'launch',
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
        'min_clearance': 0.0,
        'planned_path': [(start_x, start_y)],
        'plan_line': plan_line,
        'target_marker': target_marker,
        'subgoal_marker': subgoal_marker,
        'plan_goal_index': -1,
        'last_plan_time': -1e9,
        'path_progress_index': 0,
        'local_known_grid': planner_init_known_grid(),
        'known_occ_count': 0,
        'local_known_occ_count': 0,
        'just_discovered_obstacle': False,
        'recent_positions': [(0.0, start_x, start_y)],
        'recovery_until': -1e9,
        'recovery_turn_sign': 1.0,
        'last_scan_bias': 1.0,
        'last_command_active': False,
        'stuck_events': 0,
        'replan_count': 0,
        'blocked_replan_count': 0,
        'goal_assignment_count': 0,
        'goal_reached_count': 0,
        'measurement_update_count': 0,
        'last_landmark_update_time': None,
        'known_landmarks': {},
        'last_detected_landmarks': [],
        'distance_travelled': 0.0,
        'idle_time': 0.0,
        'last_goal_type': 'none',
        'visited_cells': {(int(round(start_x)), int(round(start_y)))},
    }

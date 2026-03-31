import os
from datetime import datetime

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from .config import COLOR_BOUNDARY, COLOR_OBSTACLE, OUTPUT_FOLDER_NAME, WORLD_HEIGHT_METERS, WORLD_WIDTH_METERS
from .paths import output_dir


def make_output_dir():
    return str(output_dir(OUTPUT_FOLDER_NAME))


def grid_to_rgb(grid, free_value, occupied_value):
    rgb = np.ones((grid.shape[0], grid.shape[1], 3), dtype=float)
    rgb[:] = np.array([0.78, 0.78, 0.78])
    rgb[grid == free_value] = np.array([0.98, 0.98, 0.98])
    rgb[grid == occupied_value] = np.array([0.18, 0.18, 0.18])
    return rgb


def draw_map_panel(ax, grid, drones, title, free_value, occupied_value, show_trajectories=False):
    rgb = grid_to_rgb(grid, free_value, occupied_value)
    ax.imshow(rgb, origin='lower', extent=[0, WORLD_WIDTH_METERS, 0, WORLD_HEIGHT_METERS], interpolation='nearest')
    ax.set_xlim(0, WORLD_WIDTH_METERS)
    ax.set_ylim(0, WORLD_HEIGHT_METERS)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.grid(False)
    ax.add_patch(patches.Rectangle((0, 0), WORLD_WIDTH_METERS, WORLD_HEIGHT_METERS, linewidth=1.6, edgecolor='black', facecolor='none'))
    for drone in drones:
        if show_trajectories:
            ax.plot(drone['trace_x'], drone['trace_y'], color=drone['color'], linewidth=1.5, alpha=0.9)
        ex, ey, _ = drone['odometry'].mu
        ax.plot([ex], [ey], marker='o', markersize=5.5, color=drone['color'], mec='black', mew=0.8)


def save_trajectory_png(obstacles, drones, ts=None):
    out_dir = make_output_dir()
    ts = ts or datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(out_dir, f'trajectories_{ts}.png')

    fig, ax = plt.subplots(figsize=(8.2, 7.8))
    ax.set_xlim(-0.4, WORLD_WIDTH_METERS + 0.4)
    ax.set_ylim(-0.4, WORLD_HEIGHT_METERS + 0.4)
    ax.set_aspect('equal')
    ax.set_title('Robot trajectories')
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.grid(True, alpha=0.15)
    ax.add_patch(patches.Rectangle((0, 0), WORLD_WIDTH_METERS, WORLD_HEIGHT_METERS, linewidth=2.0, edgecolor=COLOR_BOUNDARY, facecolor='none'))
    for obs in obstacles:
        half = obs['size'] / 2.0
        ax.add_patch(patches.Rectangle((obs['x'] - half, obs['y'] - half), obs['size'], obs['size'], facecolor=COLOR_OBSTACLE, edgecolor='black', linewidth=1.2, alpha=0.8))
    for drone in drones:
        ax.plot(drone['trace_x'], drone['trace_y'], color=drone['color'], linewidth=2.0, label=drone['name'])
        if drone['trace_x'] and drone['trace_y']:
            ax.plot([drone['trace_x'][0]], [drone['trace_y'][0]], marker='o', color=drone['color'], mec='black')
            ax.plot([drone['trace_x'][-1]], [drone['trace_y'][-1]], marker='s', color=drone['color'], mec='black')
        if drone['path']:
            xs = [p[0] for p in drone['path']]
            ys = [p[1] for p in drone['path']]
            ax.plot(xs, ys, linestyle=':', marker='x', linewidth=1.2, alpha=0.55, color=drone['color'])
    ax.legend(loc='upper left')
    fig.tight_layout(pad=0.8)
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    return out_path


def save_map_outputs(shared_known_grid, drones, free_value, occupied_value, ts=None):
    out_dir = make_output_dir()
    ts = ts or datetime.now().strftime('%Y%m%d_%H%M%S')
    shared_path = os.path.join(out_dir, f'shared_map_{ts}.png')
    fig, ax = plt.subplots(figsize=(8.0, 7.4))
    draw_map_panel(ax, shared_known_grid, drones, 'Shared observation map', free_value, occupied_value, show_trajectories=True)
    fig.tight_layout(pad=0.8)
    fig.savefig(shared_path, dpi=170, bbox_inches='tight')
    plt.close(fig)

    panel_count = len(drones) + 1
    ncols = 2
    nrows = int(np.ceil(panel_count / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.2, 4.6 * nrows))
    axes = np.atleast_1d(axes).ravel()
    draw_map_panel(axes[0], shared_known_grid, drones, 'Shared observation map', free_value, occupied_value, show_trajectories=True)
    for idx, drone in enumerate(drones, start=1):
        draw_map_panel(axes[idx], drone['local_known_grid'], drones, f"{drone['name']} local map", free_value, occupied_value, show_trajectories=True)
    for ax in axes[panel_count:]:
        ax.axis('off')
    fig.tight_layout(pad=0.9)
    out_path = os.path.join(out_dir, f'observed_maps_{ts}.png')
    fig.savefig(out_path, dpi=170, bbox_inches='tight')
    plt.close(fig)
    return shared_path, out_path

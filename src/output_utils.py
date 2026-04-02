import json
from datetime import datetime
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from config import COLOR_BOUNDARY, COLOR_OBSTACLE, WORLD_HEIGHT_METERS, WORLD_WIDTH_METERS
from .landmark import Landmark


def ensure_directory(path):
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def grid_to_rgb(grid, free_value, occupied_value):
    rgb = np.ones((grid.shape[0], grid.shape[1], 3), dtype=float)
    rgb[:] = np.array([0.78, 0.78, 0.78])
    rgb[grid == free_value] = np.array([0.98, 0.98, 0.98])
    rgb[grid == occupied_value] = np.array([0.18, 0.18, 0.18])
    return rgb


def _iter_landmark_rows(landmarks):
    if landmarks is None:
        return []
    if isinstance(landmarks, dict):
        return list(landmarks.values())
    return list(landmarks)


def _draw_landmarks(ax, landmarks, alpha=0.95):
    rows = _iter_landmark_rows(landmarks)
    for row in rows:
        lm = Landmark(
            x=float(row.get('x', 0.0)),
            y=float(row.get('y', 0.0)),
            shape=str(row.get('shape', 'circle')),
            color=row.get('color_name', row.get('color', 'yellow')),
            size=float(row.get('size', 0.8)),
        )
        patch = lm.draw(ax)
        if patch is not None:
            patch.set_alpha(alpha)
            patch.set_zorder(6)
    return rows


def draw_map_panel(ax, grid, drones, title, free_value, occupied_value, show_trajectories=False, landmarks=None):
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
        ax.plot([ex], [ey], marker='o', markersize=5.5, color=drone['color'], mec='black', mew=0.8, zorder=7)
    landmark_rows = _draw_landmarks(ax, landmarks)
    if landmark_rows:
        ax.text(
            0.02,
            0.98,
            f"landmarks: {len(landmark_rows)}",
            transform=ax.transAxes,
            ha='left',
            va='top',
            fontsize=8,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#b8c0cc', alpha=0.82),
            zorder=8,
        )


def save_trajectory_png(obstacles, drones, out_dir, ts=None):
    out_dir = ensure_directory(out_dir)
    ts = ts or datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = out_dir / f'trajectories_{ts}.png'

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
    return str(out_path)


def save_map_outputs(shared_known_grid, drones, free_value, occupied_value, out_dir, ts=None, shared_landmarks=None):
    out_dir = ensure_directory(out_dir)
    ts = ts or datetime.now().strftime('%Y%m%d_%H%M%S')
    shared_path = out_dir / f'shared_map_{ts}.png'
    panel_count = len(drones) + 1
    ncols = 2 if panel_count > 1 else 1
    nrows = int(np.ceil(panel_count / ncols))

    fig = plt.figure(figsize=(6.8 * ncols, 6.0 * nrows))
    axes = [fig.add_subplot(nrows, ncols, i + 1) for i in range(nrows * ncols)]
    draw_map_panel(axes[0], shared_known_grid, drones, 'Shared observed map', free_value, occupied_value, show_trajectories=True, landmarks=shared_landmarks)
    fig.tight_layout(pad=0.9)
    fig.savefig(shared_path, dpi=170, bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure(figsize=(6.8 * ncols, 6.0 * nrows))
    axes = [fig.add_subplot(nrows, ncols, i + 1) for i in range(nrows * ncols)]
    draw_map_panel(axes[0], shared_known_grid, drones, 'Shared observed map', free_value, occupied_value, show_trajectories=True, landmarks=shared_landmarks)
    for idx, drone in enumerate(drones, start=1):
        draw_map_panel(
            axes[idx],
            drone['local_known_grid'],
            [drone],
            f"{drone['name']} local map",
            free_value,
            occupied_value,
            show_trajectories=True,
            landmarks=drone.get('known_landmarks', {}),
        )
    for ax in axes[panel_count:]:
        ax.axis('off')
    fig.tight_layout(pad=0.9)
    out_path = out_dir / f'observed_maps_{ts}.png'
    fig.savefig(out_path, dpi=170, bbox_inches='tight')
    plt.close(fig)
    return str(shared_path), str(out_path)


def write_run_metadata(out_dir, metadata):
    out_dir = ensure_directory(out_dir)
    metadata_path = out_dir / 'run_metadata.json'
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return str(metadata_path)

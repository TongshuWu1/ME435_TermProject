import math

import matplotlib.patches as patches
import numpy as np

from config import FOV_ANGLE, VIEW_DISTANCE


def robot_shape_from_pose(x, y, angle_deg, size):
    theta = math.radians(angle_deg)
    nose = (x + size * math.cos(theta), y + size * math.sin(theta))
    left = (x + 0.55 * size * math.cos(theta + 2.45), y + 0.55 * size * math.sin(theta + 2.45))
    right = (x + 0.55 * size * math.cos(theta - 2.45), y + 0.55 * size * math.sin(theta - 2.45))
    return [nose, left, right]


def make_fov_patch(x, y, angle_deg, color, view_distance=VIEW_DISTANCE, fov_angle=FOV_ANGLE):
    return patches.Wedge(
        (x, y),
        float(view_distance),
        float(angle_deg) - float(fov_angle) / 2.0,
        float(angle_deg) + float(fov_angle) / 2.0,
        facecolor=color,
        alpha=0.10,
        edgecolor=color,
        linewidth=0.7,
        zorder=3,
    )


def update_fov_patch(patch, x, y, angle_deg, view_distance=VIEW_DISTANCE, fov_angle=FOV_ANGLE):
    theta1 = float(angle_deg) - float(fov_angle) / 2.0
    theta2 = float(angle_deg) + float(fov_angle) / 2.0

    # Wedge caches its path internally. Direct attribute assignment leaves the
    # cached path stale, which makes the cone appear frozen even though the robot
    # state and vision rays keep updating. Use the patch setters so Matplotlib
    # invalidates and rebuilds the wedge geometry.
    patch.set_center((float(x), float(y)))
    patch.set_radius(float(view_distance))
    patch.set_theta1(theta1)
    patch.set_theta2(theta2)
    patch.stale = True
    return patch


def _ellipse_geometry(mu_xy, cov_xy, n_std=2.5):
    eigvals, eigvecs = np.linalg.eigh(np.asarray(cov_xy, dtype=float))
    eigvals = np.maximum(eigvals, 1e-12)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width, height = 2.0 * float(n_std) * np.sqrt(eigvals)
    return np.asarray(mu_xy, dtype=float), float(width), float(height), float(angle)


def create_uncertainty_ellipse(ax, odometry, n_std=2.5, **kwargs):
    mean_xy, width, height, angle = _ellipse_geometry(odometry.mu[:2], odometry.cov[:2, :2], n_std=n_std)
    ellipse = patches.Ellipse(
        xy=mean_xy,
        width=width,
        height=height,
        angle=angle,
        zorder=5,
        **kwargs,
    )
    ax.add_patch(ellipse)
    return ellipse


def update_uncertainty_ellipse_patch(ellipse_patch, odometry, n_std=2.5):
    mean_xy, width, height, angle = _ellipse_geometry(odometry.mu[:2], odometry.cov[:2, :2], n_std=n_std)
    ellipse_patch.center = (float(mean_xy[0]), float(mean_xy[1]))
    ellipse_patch.width = width
    ellipse_patch.height = height
    ellipse_patch.angle = angle
    ellipse_patch.stale = True
    return ellipse_patch

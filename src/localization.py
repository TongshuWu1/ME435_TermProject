import numpy as np
import matplotlib.patches as patches


class OdometryEstimator:
    """
    EKF localization for a planar robot with state:
        mu = [x (m), y (m), heading (deg)]

    Notes
    -----
    - Heading is stored in DEGREES to stay compatible with the rest of your project.
    - The Jacobians correctly account for the degree/radian conversion.
    - predict() supports either:
        * a 2x2 control-noise covariance for [v, omega], where omega is in rad/s, or
        * a 3x3 local motion covariance for [forward_x, lateral_y, heading_deg]
          expressed in the robot/body frame.
    """

    def __init__(self, init_pos, init_cov):
        self.mu = np.array(init_pos, dtype=float)
        self.cov = np.array(init_cov, dtype=float)

        if self.mu.shape != (3,):
            raise ValueError("init_pos must be length 3: [x, y, heading_deg]")
        if self.cov.shape != (3, 3):
            raise ValueError("init_cov must be a 3x3 covariance matrix")

    @staticmethod
    def _wrap_angle_deg(angle_deg):
        """Wrap an angle to [-180, 180)."""
        return (angle_deg + 180.0) % 360.0 - 180.0

    def predict(self, u, dt, motion_noise_cov):
        """
        EKF prediction step.

        Parameters
        ----------
        u : iterable of length 2
            [v, omega], where v is linear velocity (m/s) and omega is angular
            velocity (rad/s).
        dt : float
            Time step in seconds.
        motion_noise_cov : ndarray
            Either:
              - 2x2 covariance of [v, omega], or
              - 3x3 covariance of local/body-frame motion uncertainty
                [forward_x, lateral_y, heading_deg].
        """
        v, omega = float(u[0]), float(u[1])
        theta_deg = self.mu[2]
        theta_rad = np.deg2rad(theta_deg)

        dx = v * np.cos(theta_rad) * dt
        dy = v * np.sin(theta_rad) * dt
        dtheta_deg = np.rad2deg(omega * dt)

        self.mu = self.mu + np.array([dx, dy, dtheta_deg], dtype=float)
        self.mu[2] = self._wrap_angle_deg(self.mu[2])

        deg_to_rad = np.pi / 180.0
        G = np.array([
            [1.0, 0.0, -v * np.sin(theta_rad) * dt * deg_to_rad],
            [0.0, 1.0,  v * np.cos(theta_rad) * dt * deg_to_rad],
            [0.0, 0.0, 1.0],
        ], dtype=float)

        motion_noise_cov = np.asarray(motion_noise_cov, dtype=float)

        if motion_noise_cov.shape == (2, 2):
            V = np.array([
                [np.cos(theta_rad) * dt, 0.0],
                [np.sin(theta_rad) * dt, 0.0],
                [0.0, np.rad2deg(dt)],
            ], dtype=float)
            Q = V @ motion_noise_cov @ V.T

        elif motion_noise_cov.shape == (3, 3):
            R_theta = np.array([
                [np.cos(theta_rad), -np.sin(theta_rad), 0.0],
                [np.sin(theta_rad),  np.cos(theta_rad), 0.0],
                [0.0,                0.0,               1.0],
            ], dtype=float)
            Q = R_theta @ motion_noise_cov @ R_theta.T

        else:
            raise ValueError(
                "motion_noise_cov must be either 2x2 ([v, omega]) or 3x3 "
                "([forward_x, lateral_y, heading_deg])"
            )

        self.cov = G @ self.cov @ G.T + Q
        self.cov = 0.5 * (self.cov + self.cov.T)

    def correct(self, measurements, measurement_noise_cov, alpha=1.0):
        """
        EKF correction step using landmark range/bearing observations.

        Parameters
        ----------
        measurements : list of tuples
            Each measurement is (range_meas, bearing_meas_deg, landmark_x, landmark_y)
        measurement_noise_cov : 2x2 ndarray
            Covariance for [range (m), bearing (deg)]
        alpha : float
            Optional measurement trust factor in (0, 1].
            alpha = 1.0 -> standard EKF.
            alpha < 1.0 -> soften the update by inflating R.
        """
        R = np.asarray(measurement_noise_cov, dtype=float)
        if R.shape != (2, 2):
            raise ValueError("measurement_noise_cov must be 2x2")

        alpha = float(alpha)
        if alpha <= 0.0:
            return

        R_eff = R / alpha

        I = np.eye(3)
        rad_to_deg = 180.0 / np.pi

        for z_r, z_b, lm_x, lm_y in measurements:
            dx = float(lm_x) - self.mu[0]
            dy = float(lm_y) - self.mu[1]
            q = dx * dx + dy * dy

            if q < 1e-12:
                continue

            sqrt_q = np.sqrt(q)

            predicted_bearing = np.degrees(np.arctan2(dy, dx)) - self.mu[2]
            predicted_bearing = self._wrap_angle_deg(predicted_bearing)
            z_hat = np.array([sqrt_q, predicted_bearing], dtype=float)

            H = np.array([
                [-dx / sqrt_q,            -dy / sqrt_q,            0.0],
                [ rad_to_deg * dy / q, -rad_to_deg * dx / q,     -1.0],
            ], dtype=float)

            S = H @ self.cov @ H.T + R_eff
            K = self.cov @ H.T @ np.linalg.inv(S)

            z = np.array([float(z_r), float(z_b)], dtype=float)
            y = z - z_hat
            y[1] = self._wrap_angle_deg(y[1])

            self.mu = self.mu + K @ y
            self.mu[2] = self._wrap_angle_deg(self.mu[2])

            KH = K @ H
            self.cov = (I - KH) @ self.cov @ (I - KH).T + K @ R_eff @ K.T
            self.cov = 0.5 * (self.cov + self.cov.T)

    def ellipse_params(self, n_std=2.0):
        cov_xy = self.cov[:2, :2]
        mean_xy = self.mu[:2]

        eigvals, eigvecs = np.linalg.eigh(cov_xy)
        eigvals = np.maximum(eigvals, 1e-12)

        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        width, height = 2.0 * n_std * np.sqrt(eigvals)
        return mean_xy, width, height, angle

    def draw_uncertainty_ellipse(self, ax, n_std=2.0, ellipse=None, **kwargs):
        """
        Draw or update and return the 2D uncertainty ellipse from the x-y covariance.
        """
        mean_xy, width, height, angle = self.ellipse_params(n_std=n_std)

        if ellipse is None:
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

        ellipse.center = mean_xy
        ellipse.width = width
        ellipse.height = height
        ellipse.angle = angle
        for key, value in kwargs.items():
            setter = getattr(ellipse, f'set_{key}', None)
            if callable(setter):
                setter(value)
        return ellipse

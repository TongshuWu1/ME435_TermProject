import math
import numpy as np

from config import ROBOT_COLLISION_SUBSTEP, WORLD_HEIGHT_METERS, WORLD_WIDTH_METERS


class Robot:
    def __init__(self, x, y, angle=90, size=0.5, motor_distance=0.5, max_speed=2.0, noise_std=(0.01, 0.01, 0.1), rng=None):
        self.x = x
        self.y = y
        self.angle = angle  # degrees
        self.size = size
        self.motor_distance = motor_distance
        self.max_speed = max_speed

        self.left_motor_speed = 0.0
        self.right_motor_speed = 0.0
        self.noise_std = np.array(noise_std, dtype=float)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.last_visible_landmarks = set()

    def set_motor_speeds(self, left_speed=0.0, right_speed=0.0):
        self.left_motor_speed = np.clip(left_speed, -self.max_speed, self.max_speed)
        self.right_motor_speed = np.clip(right_speed, -self.max_speed, self.max_speed)

    def set_velocity(self, linear_velocity, angular_velocity):
        L = self.motor_distance
        vl = linear_velocity - (angular_velocity * L / 2)
        vr = linear_velocity + (angular_velocity * L / 2)
        self.set_motor_speeds(vl, vr)

    @staticmethod
    def _angle_in_fov(target_angle, robot_angle, fov_angle):
        diff = ((target_angle - robot_angle + 180.0) % 360.0) - 180.0
        return abs(diff) <= (fov_angle / 2.0)

    @staticmethod
    def _ray_box_intersection_distance(x0, y0, dx, dy, xmin, xmax, ymin, ymax, max_distance):
        eps = 1e-9
        tmin = -float('inf')
        tmax = float('inf')

        if abs(dx) < eps:
            if x0 < xmin or x0 > xmax:
                return None
        else:
            tx1 = (xmin - x0) / dx
            tx2 = (xmax - x0) / dx
            tmin = max(tmin, min(tx1, tx2))
            tmax = min(tmax, max(tx1, tx2))

        if abs(dy) < eps:
            if y0 < ymin or y0 > ymax:
                return None
        else:
            ty1 = (ymin - y0) / dy
            ty2 = (ymax - y0) / dy
            tmin = max(tmin, min(ty1, ty2))
            tmax = min(tmax, max(ty1, ty2))

        if tmax < max(tmin, 0.0):
            return None

        hit_t = tmin if tmin >= 0.0 else tmax
        if hit_t < 0.0 or hit_t > max_distance:
            return None
        return hit_t

    @classmethod
    def _segment_box_intersects(cls, x1, y1, x2, y2, xmin, xmax, ymin, ymax):
        dx = x2 - x1
        dy = y2 - y1
        dist = math.hypot(dx, dy)
        if dist < 1e-9:
            return xmin <= x1 <= xmax and ymin <= y1 <= ymax
        ux = dx / dist
        uy = dy / dist
        hit = cls._ray_box_intersection_distance(x1, y1, ux, uy, xmin, xmax, ymin, ymax, dist)
        return hit is not None

    @staticmethod
    def _collides_with_square(x, y, radius, square):
        half = square['size'] / 2.0
        closest_x = min(max(x, square['x'] - half), square['x'] + half)
        closest_y = min(max(y, square['y'] - half), square['y'] + half)
        return math.hypot(x - closest_x, y - closest_y) <= radius

    @staticmethod
    def _collides_with_robot(x, y, radius, other):
        other_radius = getattr(other, 'size', 0.0) / 2.0
        return math.hypot(float(x) - float(other.x), float(y) - float(other.y)) <= (radius + other_radius)

    def collides(self, x, y, obstacles=None, other_robots=None):
        if obstacles is None:
            obstacles = ()
        if other_robots is None:
            other_robots = ()
        radius = self.size / 2.0
        if x - radius < 0 or x + radius > WORLD_WIDTH_METERS:
            return True
        if y - radius < 0 or y + radius > WORLD_HEIGHT_METERS:
            return True
        if any(self._collides_with_square(x, y, radius, obs) for obs in obstacles):
            return True
        return any(self._collides_with_robot(x, y, radius, other) for other in other_robots if other is not self)

    def update(self, dt, obstacles=None, other_robots=None):
        if obstacles is None:
            obstacles = ()
        if other_robots is None:
            other_robots = ()

        vl = self.left_motor_speed
        vr = self.right_motor_speed
        L = self.motor_distance

        velocity = (vl + vr) / 2.0
        angular_velocity = (vr - vl) / L

        v_noisy = velocity + self.rng.normal(0.0, self.noise_std[0])
        lateral_noise = self.rng.normal(0.0, self.noise_std[1])
        omega_noisy = angular_velocity + self.rng.normal(0.0, self.noise_std[2])

        theta_rad = math.radians(self.angle)
        delta_x = (v_noisy * math.cos(theta_rad) - lateral_noise * math.sin(theta_rad)) * dt
        delta_y = (v_noisy * math.sin(theta_rad) + lateral_noise * math.cos(theta_rad)) * dt

        total_dist = math.hypot(delta_x, delta_y)
        max_step = max(1e-6, float(ROBOT_COLLISION_SUBSTEP))
        steps = max(1, int(math.ceil(total_dist / max_step)))
        step_dx = delta_x / steps
        step_dy = delta_y / steps

        for _ in range(steps):
            cand_x = self.x + step_dx
            cand_y = self.y + step_dy
            if not self.collides(cand_x, cand_y, obstacles, other_robots):
                self.x = cand_x
                self.y = cand_y
                continue
            moved = False
            if abs(step_dx) > 1e-9 and not self.collides(self.x + step_dx, self.y, obstacles, other_robots):
                self.x = self.x + step_dx
                moved = True
            if abs(step_dy) > 1e-9 and not self.collides(self.x, self.y + step_dy, obstacles, other_robots):
                self.y = self.y + step_dy
                moved = True
            if not moved:
                break

        self.angle = (self.angle + math.degrees(omega_noisy * dt)) % 360.0

    def scan_obstacles(self, obstacles, fov_angle, view_distance, ray_count=21):
        if ray_count < 2:
            ray_count = 2

        results = []
        for rel in np.linspace(-fov_angle / 2.0, fov_angle / 2.0, ray_count):
            ang_deg = (self.angle + rel) % 360.0
            ang_rad = math.radians(ang_deg)
            dx = math.cos(ang_rad)
            dy = math.sin(ang_rad)

            candidates = []

            # World boundary as an axis-aligned box.
            boundary_hit = self._ray_box_intersection_distance(
                self.x,
                self.y,
                dx,
                dy,
                0.0,
                WORLD_WIDTH_METERS,
                0.0,
                WORLD_HEIGHT_METERS,
                view_distance,
            )
            if boundary_hit is not None:
                candidates.append(boundary_hit)

            # Obstacles as squares.
            for obs in obstacles:
                half = obs['size'] / 2.0
                hit = self._ray_box_intersection_distance(
                    self.x,
                    self.y,
                    dx,
                    dy,
                    obs['x'] - half,
                    obs['x'] + half,
                    obs['y'] - half,
                    obs['y'] + half,
                    view_distance,
                )
                if hit is not None:
                    candidates.append(hit)

            distance = min(candidates) if candidates else view_distance
            results.append({
                'angle_deg': ang_deg,
                'rel_angle_deg': float(rel),
                'distance': distance,
                'hit_x': self.x + distance * dx,
                'hit_y': self.y + distance * dy,
                'is_obstacle_hit': bool(distance < view_distance - 1e-6),
            })
        return results

    def line_of_sight_blocked(self, x2, y2, obstacles):
        for obs in obstacles:
            half = obs['size'] / 2.0
            if self._segment_box_intersects(
                self.x,
                self.y,
                x2,
                y2,
                obs['x'] - half,
                obs['x'] + half,
                obs['y'] - half,
                obs['y'] + half,
            ):
                return True
        return False

    def detect_landmarks(self, landmarks, fov_angle, view_distance, obstacles=None):
        if obstacles is None:
            obstacles = ()

        detected_landmarks = set()
        for landmark in landmarks:
            if self.is_landmark_within_fov(landmark, fov_angle, view_distance, obstacles):
                detected_landmarks.add((landmark.shape, landmark.color_name, landmark.x, landmark.y))

        self.last_visible_landmarks = detected_landmarks
        return [l for l in landmarks if (l.shape, l.color_name, l.x, l.y) in detected_landmarks]

    def is_landmark_within_fov(self, landmark, fov_angle, view_distance, obstacles=None):
        if obstacles is None:
            obstacles = ()

        dx = landmark.x - self.x
        dy = landmark.y - self.y
        distance = math.hypot(dx, dy)
        if distance > view_distance:
            return False

        landmark_angle = math.degrees(math.atan2(dy, dx)) % 360.0
        if not self._angle_in_fov(landmark_angle, self.angle % 360.0, fov_angle):
            return False

        if self.line_of_sight_blocked(landmark.x, landmark.y, obstacles):
            return False

        return True

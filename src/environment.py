import math
import random

from config import (
    BOUNDARY_MARGIN,
    ROBOT_COUNT,
    DRONE_START_POSE,
    HOME_BASE_FORWARD_CLEARANCE,
    HOME_BASE_HEIGHT,
    HOME_BASE_WIDTH,
    LANDMARK_CLEARANCE,
    LANDMARK_COUNT,
    LANDMARK_SIZE,
    OBSTACLE_CLEARANCE,
    OBSTACLE_COUNT,
    OBSTACLE_SIZE_RANGE,
    RANDOM_SEED,
    START_CLEAR_RADIUS,
    WORLD_HEIGHT_METERS,
    WORLD_WIDTH_METERS,
    A_STAR_GRID_RESOLUTION,
)


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def square_contains_point(square, x, y, margin=0.0):
    half = square['size'] / 2.0 + margin
    return (square['x'] - half) <= x <= (square['x'] + half) and (square['y'] - half) <= y <= (square['y'] + half)


def squares_too_close(a, b, margin=0.0):
    half_a = a['size'] / 2.0 + margin
    half_b = b['size'] / 2.0 + margin
    return abs(a['x'] - b['x']) < (half_a + half_b) and abs(a['y'] - b['y']) < (half_a + half_b)


def home_base_region():
    cx, cy, heading_deg = DRONE_START_POSE
    width = float(HOME_BASE_WIDTH)
    height = float(HOME_BASE_HEIGHT)
    heading = math.radians(heading_deg)
    forward = (math.cos(heading), math.sin(heading))
    center_x = float(cx - forward[0] * (height * 0.25))
    center_y = float(cy - forward[1] * (height * 0.25))
    return {
        "cx": center_x,
        "cy": center_y,
        "width": width,
        "height": height,
        "heading_deg": float(heading_deg),
    }


def point_in_home_base(x, y, extra_margin=0.0):
    base = home_base_region()
    hx = base["width"] / 2.0 + float(extra_margin)
    hy = base["height"] / 2.0 + float(extra_margin)
    return (base["cx"] - hx) <= x <= (base["cx"] + hx) and (base["cy"] - hy) <= y <= (base["cy"] + hy)


def home_base_front_clear_radius():
    return max(HOME_BASE_WIDTH / 2.0, HOME_BASE_HEIGHT) + float(HOME_BASE_FORWARD_CLEARANCE)

def _snap_cell_center(v, resolution=A_STAR_GRID_RESOLUTION):
    return (round(float(v) / float(resolution)) + 0.5) * float(resolution)


def _snap_cell_size(v, resolution=A_STAR_GRID_RESOLUTION):
    cells = max(1, int(round(float(v) / float(resolution))))
    return cells * float(resolution)

def generate_obstacles(seed=RANDOM_SEED):
    rng = random.Random(seed)
    obstacles = []
    attempts = 0
    while len(obstacles) < OBSTACLE_COUNT and attempts < 5000:
        attempts += 1
        size = _snap_cell_size(rng.uniform(*OBSTACLE_SIZE_RANGE))
        half = size / 2.0
        x = _snap_cell_center(rng.uniform(BOUNDARY_MARGIN + half, WORLD_WIDTH_METERS - BOUNDARY_MARGIN - half))
        y = _snap_cell_center(rng.uniform(BOUNDARY_MARGIN + half + 2.0, WORLD_HEIGHT_METERS - BOUNDARY_MARGIN - half))
        square = {'x': round(x, 3), 'y': round(y, 3), 'size': round(size, 3)}

        if distance((x, y), DRONE_START_POSE[:2]) < START_CLEAR_RADIUS + half:
            continue
        if point_in_home_base(x, y, extra_margin=half + OBSTACLE_CLEARANCE):
            continue
        if any(squares_too_close(square, other, margin=OBSTACLE_CLEARANCE) for other in obstacles):
            continue
        obstacles.append(square)
    return obstacles


def generate_landmarks(obstacles, seed=RANDOM_SEED + 1):
    rng = random.Random(seed)
    shapes = ['circle', 'square', 'triangle']
    colors = ['yellow', 'orange']
    landmarks = []
    attempts = 0
    while len(landmarks) < LANDMARK_COUNT and attempts < 8000:
        attempts += 1
        x = _snap_cell_center(rng.uniform(BOUNDARY_MARGIN, WORLD_WIDTH_METERS - BOUNDARY_MARGIN))
        y = _snap_cell_center(rng.uniform(BOUNDARY_MARGIN, WORLD_HEIGHT_METERS - BOUNDARY_MARGIN))
        if distance((x, y), DRONE_START_POSE[:2]) < START_CLEAR_RADIUS:
            continue
        if point_in_home_base(x, y, extra_margin=LANDMARK_CLEARANCE):
            continue
        if any(square_contains_point(obs, x, y, margin=LANDMARK_CLEARANCE) for obs in obstacles):
            continue
        if any(distance((x, y), (lm['x'], lm['y'])) < 2.2 for lm in landmarks):
            continue
        landmarks.append({
            'x': round(x, 2),
            'y': round(y, 2),
            'shape': rng.choice(shapes),
            'color': rng.choice(colors),
            'size': LANDMARK_SIZE,
        })
    return landmarks


def generate_environment(seed=RANDOM_SEED):
    obstacles = generate_obstacles(seed)
    landmarks = generate_landmarks(obstacles, seed + 1)
    return obstacles, landmarks


def empty_target_sequences():
    return [[] for _ in range(ROBOT_COUNT)]

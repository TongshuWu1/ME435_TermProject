import math
import random

from .config import (
    BOUNDARY_MARGIN,
    DRONE_NAMES,
    DRONE_START_POSE,
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


def generate_obstacles(seed=RANDOM_SEED):
    rng = random.Random(seed)
    obstacles = []
    attempts = 0
    while len(obstacles) < OBSTACLE_COUNT and attempts < 5000:
        attempts += 1
        size = rng.uniform(*OBSTACLE_SIZE_RANGE)
        half = size / 2.0
        x = rng.uniform(BOUNDARY_MARGIN + half, WORLD_WIDTH_METERS - BOUNDARY_MARGIN - half)
        y = rng.uniform(BOUNDARY_MARGIN + half + 2.0, WORLD_HEIGHT_METERS - BOUNDARY_MARGIN - half)
        square = {'x': round(x, 2), 'y': round(y, 2), 'size': round(size, 2)}

        if distance((x, y), DRONE_START_POSE[:2]) < START_CLEAR_RADIUS + half:
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
        x = rng.uniform(BOUNDARY_MARGIN, WORLD_WIDTH_METERS - BOUNDARY_MARGIN)
        y = rng.uniform(BOUNDARY_MARGIN, WORLD_HEIGHT_METERS - BOUNDARY_MARGIN)
        if distance((x, y), DRONE_START_POSE[:2]) < START_CLEAR_RADIUS:
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
    return [[] for _ in DRONE_NAMES]

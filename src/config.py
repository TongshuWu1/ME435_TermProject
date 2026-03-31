# ------------------------------
# World and timing
# ------------------------------
WORLD_WIDTH_METERS = 40
WORLD_HEIGHT_METERS = 40
TIME_STEP = 0.05
RANDOM_SEED = 123

# ------------------------------
# Robot sensing and control
# ------------------------------
FOV_ANGLE = 90
VIEW_DISTANCE = 10
VISION_RAY_COUNT = 21
SHOW_VISION_RAYS = True
OBSTACLE_AVOID_DISTANCE = 2.2
OBSTACLE_TURN_GAIN = 2.6

# ------------------------------
# Planner and path display
# ------------------------------
A_STAR_GRID_RESOLUTION = 1.0
A_STAR_REPLAN_SECONDS = 1.2
A_STAR_GOAL_TOLERANCE = 0.65
A_STAR_LOOKAHEAD_STEPS = 1
A_STAR_INFLATION_MARGIN = 1.20
KNOWN_MAP_REPLAN_ON_NEW_OBS = True
PATH_LINE_WIDTH = 2.8
TARGET_MARKER_SIZE = 15
SUBGOAL_MARKER_SIZE = 9

# Optional local A* debug overlay.
SHOW_ASTAR_LOCAL_GRID = False
A_STAR_GRID_WINDOW_CELLS = 4
A_STAR_GRID_LINE_ALPHA = 0.14
A_STAR_GRID_OCC_ALPHA = 0.18
A_STAR_GRID_PATH_ALPHA = 0.20
A_STAR_GRID_UPDATE_FRAMES = 4
A_STAR_GRID_MAX_DRONES = 1

# Vision-ray styling.
RAY_LINE_WIDTH = 1.3
RAY_ALPHA = 0.55

# ------------------------------
# State estimation
# ------------------------------
PREDICTION_NOISE = [0.02, 0.07, 0.5]
MEASUREMENT_NOISE = [0.12, 2.5]
MEASUREMENT_ALPHA = 0.35

# ------------------------------
# Recovery / anti-stuck behavior
# ------------------------------
STUCK_WINDOW_SECONDS = 2.0
STUCK_PROGRESS_EPS = 0.8
STUCK_RECOVERY_SECONDS = 1.3
STUCK_REVERSE_SPEED = 0.45
STUCK_TURN_SPEED = 2.8

# ------------------------------
# Multi-robot setup
# ------------------------------
DRONE_NAMES = ["Drone 1", "Drone 2", "Drone 3"]
DRONE_START_POSE = (20.0, 4.0, 90.0)
DRONE_START_SPACING = 1.6
START_SIMULATION_RUNNING = False
OUTPUT_FOLDER_NAME = "outputs"

# ------------------------------
# Random environment generation
# ------------------------------
OBSTACLE_COUNT = 8
OBSTACLE_SIZE_RANGE = (4.0, 7.6)
OBSTACLE_CLEARANCE = 1.2

LANDMARK_COUNT = 14
LANDMARK_SIZE = 0.8
LANDMARK_CLEARANCE = 1.3

START_CLEAR_RADIUS = 4.5
BOUNDARY_MARGIN = 2.5


# Colors shared by multiple modules.
COLOR_UNKNOWN = '#d9dde3'
COLOR_FREE = '#f8fbff'
COLOR_OCCUPIED = '#5b6470'
COLOR_OBSTACLE = '#5A5A5A'
COLOR_BOUNDARY = 'red'


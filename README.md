# ME435 Term Project

Project layout:
- `main.py` — entry point
- `src/` — source code
- `outputs/` — saved PNG results

Main modules inside `src/`:
- `simulation.py` — main simulator loop and robot coordination
- `sim_ui.py` — Matplotlib canvas, buttons, selectors, and shared-map panel
- `auto_explore.py` — frontier detection and paper-inspired Voronoi partition helpers
- `planner.py` — occupancy-grid helpers and A* path planning
- `mapping_utils.py` — shared-map and local-map update helpers
- `environment.py` — obstacle and landmark generation
- `output_utils.py` — saved trajectory and map figures
- `paths.py` — project-root and output-folder helpers
- `robot.py` — robot motion, collision checks, landmark detection, and obstacle scans
- `localization.py` — EKF state estimator
- `landmark.py` — landmark drawing helper
- `config.py` — simulation constants

## Mission modes
The simulator now has two modes:

1. **Manual Click**
   - same as before
   - click to add waypoints for the selected robot
   - use `Set Start` to move a robot's launch position

2. **Auto Explore**
   - robots automatically build the shared map
   - frontiers are detected from the shared known map
   - a Voronoi-style partition is computed from the robots' estimated positions
   - each robot picks a frontier goal mainly inside its own current partition
   - A* is still used to route to that goal

## UI notes
- `Mission Mode` switches between manual waypointing and automatic exploration.
- `Start / Pause` controls the simulation.
- `Hide / Show Partition` toggles the partition overlay.
- The small map on the right is the shared explored map.

## Notes
- The simulator starts paused.
- Saved figures always go to the top-level `outputs/` folder.
- Robot count is still determined by `DRONE_NAMES` in `src/config.py` for this version.

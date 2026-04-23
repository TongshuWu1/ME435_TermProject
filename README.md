# ME435 Term Project

## Project layout
- `main.py` — entry point
- `config.py` — simulation constants
- `src/` — source code

## Source structure
Top-level modules:
- `src/simulation.py` — main simulator loop and high-level coordination
- `src/auto_explore.py` — frontier selection logic
- `src/planner.py` — occupancy-grid helpers and A* planning
- `src/mapping_utils.py` — map update helpers
- `src/environment.py` — obstacle and landmark generation
- `src/robot.py` — robot motion, collision checks, landmark detection, and obstacle scans
- `src/localization.py` — EKF state estimator
- `src/landmark.py` — landmark model

Refactored subpackages:
- `src/sim_ui.py` — Matplotlib canvas, buttons, selectors, and shared-map panel
- `src/sim/rendering.py` — reusable robot / FOV / covariance ellipse rendering helpers
- `src/sim/drone_factory.py` — drone creation and artist setup
- `src/sim/partition_state.py` — cached partition / density overlay generation
- `src/controllers/frontier_controller.py` — frontier-goal controller wrapper
- `src/controllers/coverage_controller.py` — weighted Voronoi / centroid coverage controller

Compatibility shim:
- `src/sim_ui.py` — imports `SimulatorUI` from `src/sim_ui.py`

## Mission modes
1. **Manual Click**
   - click to add waypoints for the selected robot
   - use `Set Start` to move a robot's launch position

2. **Auto Explore**
   - robots automatically build the shared map
   - frontiers are detected from the shared known map
   - a Voronoi-style partition is computed from the robots' estimated positions
   - you can switch between **Frontier** and **Weighted Coverage** auto policies
   - the weighted-coverage policy now favors reachable, high-information pockets in each robot's region, penalizes revisiting already-worked cells, and uses the partition centroid only as a soft fallback
   - A* is still used to route to that goal

## UI notes
- `Mission Mode` switches between manual waypointing and automatic exploration.
- `Auto Policy` switches the exploration controller between classic frontier goals and weighted Voronoi coverage.
- `Start / Pause` controls the simulation.
- `Show / Hide Region` toggles the partition overlay.
- `Show / Hide Density` toggles the density overlay.
- The top-right status block is now clipped inside its own panel so it no longer covers the main canvas.
- The small map on the right is the shared explored map.

## Notes
- The simulator starts paused.
- The current baseline focuses on live simulation, local mapping, and monitoring windows rather than report-folder export.
- Robot count is determined by `DRONE_NAMES` in `config.py`.

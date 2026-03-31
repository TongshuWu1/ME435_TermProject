# ME435 Term Project

## Project layout
- `main.py` — entry point
- `config.py` — simulation constants
- `src/` — source code
- `outputs/` — saved run folders with PNG, CSV, and JSON summaries

## Source structure
Top-level modules:
- `src/simulation.py` — main simulator loop and high-level coordination
- `src/auto_explore.py` — frontier selection logic
- `src/planner.py` — occupancy-grid helpers and A* planning
- `src/mapping_utils.py` — map update helpers
- `src/environment.py` — obstacle and landmark generation
- `src/output_utils.py` — saved trajectory and map figures
- `src/reporting.py` — run summaries, event logs, coverage-history CSVs, and coverage progress plots
- `src/paths.py` — project-root and output-folder helpers
- `src/robot.py` — robot motion, collision checks, landmark detection, and obstacle scans
- `src/localization.py` — EKF state estimator
- `src/landmark.py` — landmark model

Refactored subpackages:
- `src/ui/simulator_ui.py` — Matplotlib canvas, buttons, selectors, and shared-map panel
- `src/sim/rendering.py` — reusable robot / FOV / covariance ellipse rendering helpers
- `src/sim/drone_factory.py` — drone creation and artist setup
- `src/sim/partition_state.py` — cached partition / density overlay generation
- `src/controllers/frontier_controller.py` — frontier-goal controller wrapper
- `src/controllers/coverage_controller.py` — weighted Voronoi / centroid coverage controller

Compatibility shim:
- `src/sim_ui.py` — imports `SimulatorUI` from `src/ui/simulator_ui.py`

## Mission modes
1. **Manual Click**
   - click to add waypoints for the selected robot
   - use `Set Start` to move a robot's launch position

2. **Auto Explore**
   - robots automatically build the shared map
   - frontiers are detected from the shared known map
   - a Voronoi-style partition is computed from the robots' estimated positions
   - you can switch between **Frontier** and **Weighted Coverage** auto policies
   - the weighted-coverage policy drives each robot toward a reachable, high-density point near its weighted partition centroid
   - A* is still used to route to that goal

## UI notes
- `Mission Mode` switches between manual waypointing and automatic exploration.
- `Auto Policy` switches the exploration controller between classic frontier goals and weighted Voronoi coverage.
- `Start / Pause` controls the simulation.
- `Show / Hide Region` toggles the partition overlay.
- `Show / Hide Density` toggles the density overlay.
- The small map on the right is the shared explored map.

## Notes
- The simulator starts paused.
- Each run saves a folder under `outputs/` containing trajectory/map PNGs plus `run_summary.json`, `robot_stats.csv`, `coverage_history.csv`, and `event_log.csv`.
- Robot count is determined by `DRONE_NAMES` in `config.py`.

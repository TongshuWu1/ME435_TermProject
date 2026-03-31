# ME435 Term Project

Project layout:
- `main.py` — entry point
- `src/` — source code
- `outputs/` — saved PNG results

Main modules inside `src/`:
- `simulation.py` — main simulator loop and robot coordination
- `sim_ui.py` — Matplotlib canvas, buttons, selectors, and shared-map panel
- `planner.py` — occupancy-grid helpers and A* path planning
- `mapping_utils.py` — shared-map and local-map update helpers
- `environment.py` — obstacle and landmark generation
- `output_utils.py` — saved trajectory and map figures
- `paths.py` — project-root and output-folder helpers
- `robot.py` — robot motion, collision checks, landmark detection, and obstacle scans
- `localization.py` — EKF state estimator
- `landmark.py` — landmark drawing helper
- `config.py` — simulation constants

Notes:
- The simulator starts paused with no automatic waypoints.
- Add starts and waypoints manually in the UI.
- Saved figures always go to the top-level `outputs/` folder.

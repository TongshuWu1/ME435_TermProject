import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


def ensure_directory(path):
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def polyline_length(points):
    if not points or len(points) < 2:
        return 0.0
    total = 0.0
    for p0, p1 in zip(points[:-1], points[1:]):
        total += math.hypot(float(p1[0]) - float(p0[0]), float(p1[1]) - float(p0[1]))
    return total


def compute_coverage_metrics(shared_known_grid, truth_occupancy, *, unknown_value=0, free_value=1, occupied_value=2):
    total_cells = int(shared_known_grid.size)
    known_mask = shared_known_grid != unknown_value
    known_cells = int(known_mask.sum())

    truth_occ = truth_occupancy.astype(bool)
    truth_free = ~truth_occ
    total_free_cells = int(truth_free.sum())
    total_occupied_cells = int(truth_occ.sum())

    covered_free_cells = int(((shared_known_grid == free_value) & truth_free).sum())
    mapped_occupied_cells = int(((shared_known_grid == occupied_value) & truth_occ).sum())

    unknown_cells = total_cells - known_cells
    metrics = {
        'total_cells': total_cells,
        'known_cells': known_cells,
        'unknown_cells': unknown_cells,
        'known_ratio': (known_cells / total_cells) if total_cells else 0.0,
        'total_free_cells_truth': total_free_cells,
        'covered_free_cells': covered_free_cells,
        'free_coverage_ratio': (covered_free_cells / total_free_cells) if total_free_cells else 0.0,
        'total_occupied_cells_truth': total_occupied_cells,
        'mapped_occupied_cells': mapped_occupied_cells,
        'occupied_recall_ratio': (mapped_occupied_cells / total_occupied_cells) if total_occupied_cells else 0.0,
    }
    return metrics


def write_csv(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in fieldnames})
    return str(path)


def save_coverage_progress_plot(coverage_history, out_dir, ts=None):
    if not coverage_history:
        return ''
    out_dir = ensure_directory(out_dir)
    suffix = '' if ts is None else f'_{ts}'
    out_path = out_dir / f'coverage_progress{suffix}.png'

    times = [float(row.get('time_seconds', 0.0)) for row in coverage_history]
    known_pct = [100.0 * float(row.get('known_ratio', 0.0)) for row in coverage_history]
    free_pct = [100.0 * float(row.get('free_coverage_ratio', 0.0)) for row in coverage_history]
    frontier_counts = [float(row.get('frontier_count', 0.0)) for row in coverage_history]

    fig, ax1 = plt.subplots(figsize=(8.4, 4.8))
    ax1.plot(times, known_pct, linewidth=2.2, label='Map known (%)')
    ax1.plot(times, free_pct, linewidth=2.2, linestyle='--', label='Free space covered (%)')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Coverage (%)')
    ax1.set_ylim(0.0, 100.0)
    ax1.grid(True, alpha=0.2)

    ax2 = ax1.twinx()
    ax2.plot(times, frontier_counts, linewidth=1.6, linestyle=':', label='Frontier groups')
    ax2.set_ylabel('Frontier groups')
    ax2.set_ylim(bottom=0.0)

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='lower right')
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches='tight')
    plt.close(fig)
    return str(out_path)


def save_run_reports(*, out_dir, metadata, coverage_history, event_log, robot_rows, summary, ts=None):
    out_dir = ensure_directory(out_dir)
    suffix = '' if ts is None else f'_{ts}'

    summary_path = out_dir / 'run_summary.json'
    summary_payload = {
        'metadata': metadata,
        'summary': summary,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2))

    coverage_path = write_csv(
        out_dir / f'coverage_history{suffix}.csv',
        [
            'time_seconds', 'known_ratio', 'free_coverage_ratio', 'occupied_recall_ratio',
            'known_cells', 'covered_free_cells', 'mapped_occupied_cells', 'frontier_count',
            'active_goals', 'running', 'auto_finished', 'selected_robot', 'auto_policy', 'mission_mode'
        ],
        coverage_history,
    )

    event_path = write_csv(
        out_dir / f'event_log{suffix}.csv',
        ['seq', 'time_seconds', 'event_type', 'robot_name', 'message', 'data_json'],
        event_log,
    )

    robot_path = write_csv(
        out_dir / f'robot_stats{suffix}.csv',
        [
            'name', 'phase', 'goal_type', 'goal_assignments', 'goals_reached', 'replans', 'replans_blocked',
            'stuck_events', 'measurement_updates', 'last_landmark_update_time', 'distance_travelled_m',
            'idle_time_s', 'visited_cell_count', 'path_waypoint_count', 'remaining_path_points',
            'remaining_path_length_m', 'final_true_x', 'final_true_y', 'final_est_x', 'final_est_y',
        ],
        robot_rows,
    )

    coverage_plot_path = save_coverage_progress_plot(coverage_history, out_dir, ts=ts)
    return {
        'summary_json': str(summary_path),
        'coverage_history_csv': coverage_path,
        'event_log_csv': event_path,
        'robot_stats_csv': robot_path,
        'coverage_plot_png': coverage_plot_path,
    }

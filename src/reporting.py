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


def save_coverage_progress_plot(coverage_history, out_dir, ts=None, *, mission_mode='manual_click', auto_policy='frontier'):
    if not coverage_history:
        return ''
    out_dir = ensure_directory(out_dir)
    suffix = '' if ts is None else f'_{ts}'
    out_path = out_dir / f'coverage_progress{suffix}.png'

    times = [float(row.get('time_seconds', 0.0)) for row in coverage_history]
    known_pct = [100.0 * float(row.get('known_ratio', 0.0)) for row in coverage_history]
    free_pct = [100.0 * float(row.get('free_coverage_ratio', 0.0)) for row in coverage_history]
    occ_pct = [100.0 * float(row.get('occupied_recall_ratio', 0.0)) for row in coverage_history]
    frontier_counts = [float(row.get('frontier_count', 0.0)) for row in coverage_history]

    fig, ax1 = plt.subplots(figsize=(8.8, 5.0))
    ax1.plot(times, known_pct, linewidth=2.2, label='Map known (%)')
    ax1.plot(times, free_pct, linewidth=2.2, linestyle='--', label='Free space covered (%)')
    ax1.plot(times, occ_pct, linewidth=1.8, linestyle='-.', label='Obstacle cells found (%)')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Coverage (%)')
    ax1.set_ylim(0.0, 100.0)
    ax1.grid(True, alpha=0.2)
    title_mode = 'Auto Explore' if mission_mode == 'auto_explore' else 'Manual Click'
    title_policy = 'Weighted Coverage' if auto_policy == 'weighted_coverage' else 'Frontier'
    ax1.set_title(f'Coverage progress — {title_mode} / {title_policy}')

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


def _pct(value):
    return 100.0 * float(value)


def _fmt_pct(value):
    return f'{_pct(value):.1f}%'


def _build_summary_payload(metadata, summary, robot_rows, generated_files):
    return {
        'run': {
            'label': metadata.get('run_label', ''),
            'saved_at': metadata.get('saved_at', ''),
            'seed': metadata.get('seed', ''),
            'mission_mode': metadata.get('mission_mode', ''),
            'auto_policy': metadata.get('auto_policy', ''),
            'robot_count': metadata.get('robot_count', 0),
            'sim_time_seconds': metadata.get('sim_time_seconds', 0.0),
            'auto_mode_enabled_at_save': metadata.get('auto_mode_enabled_at_save', False),
        },
        'coverage': {
            'map_known_ratio': summary.get('known_ratio', 0.0),
            'map_known_percent': round(_pct(summary.get('known_ratio', 0.0)), 3),
            'free_space_covered_ratio': summary.get('free_coverage_ratio', 0.0),
            'free_space_covered_percent': round(_pct(summary.get('free_coverage_ratio', 0.0)), 3),
            'obstacle_cells_found_ratio': summary.get('occupied_recall_ratio', 0.0),
            'obstacle_cells_found_percent': round(_pct(summary.get('occupied_recall_ratio', 0.0)), 3),
            'frontier_groups_at_end': summary.get('frontier_count', 0),
            'active_goals_at_end': summary.get('active_goals', 0),
            'auto_finished': summary.get('auto_finished', False),
        },
        'logging': {
            'event_count': summary.get('event_count', 0),
            'coverage_samples': summary.get('coverage_samples', 0),
        },
        'robots': robot_rows,
        'files': generated_files,
    }


def _build_summary_text(metadata, summary, robot_rows, generated_files):
    mission_name = 'Auto Explore' if metadata.get('mission_mode') == 'auto_explore' else 'Manual Click'
    policy_name = 'Weighted Coverage' if metadata.get('auto_policy') == 'weighted_coverage' else 'Frontier'
    lines = [
        'ME435 Simulator Run Summary',
        '=' * 32,
        f"Run label: {metadata.get('run_label', '')}",
        f"Saved at:  {metadata.get('saved_at', '')}",
        f"Seed:      {metadata.get('seed', '')}",
        f"Mission:   {mission_name}",
        f"Policy:    {policy_name}",
        f"Robots:    {metadata.get('robot_count', 0)}",
        f"Sim time:  {float(metadata.get('sim_time_seconds', 0.0)):.2f} s",
        '',
        'Coverage at save time',
        '-' * 22,
        f"Map known:            {_fmt_pct(summary.get('known_ratio', 0.0))}",
        f"Free space covered:   {_fmt_pct(summary.get('free_coverage_ratio', 0.0))}",
        f"Obstacle cells found: {_fmt_pct(summary.get('occupied_recall_ratio', 0.0))}",
        f"Frontier groups:      {int(summary.get('frontier_count', 0))}",
        f"Active robot goals:   {int(summary.get('active_goals', 0))}",
        f"Run finished:         {'yes' if summary.get('auto_finished', False) else 'no'}",
        '',
        'Per-robot snapshot',
        '-' * 18,
    ]
    for row in robot_rows:
        lines.append(
            f"- {row.get('robot', row.get('name', 'Robot'))}: phase={row.get('phase', '')}, "
            f"goal_type={row.get('last_goal_type', row.get('goal_type', ''))}, "
            f"goals={row.get('goals_reached', 0)}/{row.get('goal_assignments', 0)}, "
            f"replans={row.get('replans', 0)}, distance={float(row.get('distance_travelled_m', 0.0)):.2f} m"
        )
    lines.extend([
        '',
        'Saved files',
        '-' * 11,
        '- run_summary.txt: human-readable summary',
        '- run_summary.json: structured summary for scripts',
        '- robot_summary.csv: one row per robot',
        '- coverage_timeline.csv: map progress over time',
        '- event_timeline.csv: timestamped events',
        f"- coverage_progress{'_' + generated_files.get('timestamp', '') if generated_files.get('timestamp') else ''}.png: coverage chart",
        f"- trajectories{'_' + generated_files.get('timestamp', '') if generated_files.get('timestamp') else ''}.png: world trajectories",
        f"- observed_maps{'_' + generated_files.get('timestamp', '') if generated_files.get('timestamp') else ''}.png: shared + per-robot maps",
    ])
    return '\n'.join(lines) + '\n'


def save_run_reports(*, out_dir, metadata, coverage_history, event_log, robot_rows, summary, ts=None):
    out_dir = ensure_directory(out_dir)
    suffix = '' if ts is None else f'_{ts}'

    coverage_rows_pretty = []
    for row in coverage_history:
        coverage_rows_pretty.append({
            'time_s': row.get('time_seconds', ''),
            'map_known_pct': round(_pct(row.get('known_ratio', 0.0)), 3),
            'free_space_covered_pct': round(_pct(row.get('free_coverage_ratio', 0.0)), 3),
            'obstacle_cells_found_pct': round(_pct(row.get('occupied_recall_ratio', 0.0)), 3),
            'known_cells': row.get('known_cells', ''),
            'covered_free_cells': row.get('covered_free_cells', ''),
            'mapped_obstacle_cells': row.get('mapped_occupied_cells', ''),
            'frontier_groups': row.get('frontier_count', ''),
            'active_goals': row.get('active_goals', ''),
            'running': row.get('running', ''),
            'auto_finished': row.get('auto_finished', ''),
            'selected_robot': row.get('selected_robot', ''),
            'auto_policy': row.get('auto_policy', ''),
            'mission_mode': row.get('mission_mode', ''),
        })

    event_rows_pretty = []
    for row in event_log:
        event_rows_pretty.append({
            'event_index': row.get('seq', ''),
            'time_s': row.get('time_seconds', ''),
            'event': row.get('event_type', ''),
            'robot': row.get('robot_name', ''),
            'message': row.get('message', ''),
            'details_json': row.get('data_json', ''),
        })

    robot_rows_pretty = []
    for row in robot_rows:
        robot_rows_pretty.append({
            'robot': row.get('name', ''),
            'phase': row.get('phase', ''),
            'last_goal_type': row.get('goal_type', ''),
            'goal_assignments': row.get('goal_assignments', ''),
            'goals_reached': row.get('goals_reached', ''),
            'replans': row.get('replans', ''),
            'blocked_replans': row.get('replans_blocked', ''),
            'stuck_events': row.get('stuck_events', ''),
            'landmark_updates': row.get('measurement_updates', ''),
            'last_landmark_update_time_s': row.get('last_landmark_update_time', ''),
            'distance_travelled_m': row.get('distance_travelled_m', ''),
            'idle_time_s': row.get('idle_time_s', ''),
            'visited_cell_count': row.get('visited_cell_count', ''),
            'manual_waypoint_count': row.get('path_waypoint_count', ''),
            'remaining_path_points': row.get('remaining_path_points', ''),
            'remaining_path_length_m': row.get('remaining_path_length_m', ''),
            'final_true_x': row.get('final_true_x', ''),
            'final_true_y': row.get('final_true_y', ''),
            'final_est_x': row.get('final_est_x', ''),
            'final_est_y': row.get('final_est_y', ''),
        })

    coverage_path = write_csv(
        out_dir / 'coverage_timeline.csv',
        [
            'time_s', 'map_known_pct', 'free_space_covered_pct', 'obstacle_cells_found_pct',
            'known_cells', 'covered_free_cells', 'mapped_obstacle_cells', 'frontier_groups',
            'active_goals', 'running', 'auto_finished', 'selected_robot', 'auto_policy', 'mission_mode'
        ],
        coverage_rows_pretty,
    )

    event_path = write_csv(
        out_dir / 'event_timeline.csv',
        ['event_index', 'time_s', 'event', 'robot', 'message', 'details_json'],
        event_rows_pretty,
    )

    robot_path = write_csv(
        out_dir / 'robot_summary.csv',
        [
            'robot', 'phase', 'last_goal_type', 'goal_assignments', 'goals_reached', 'replans', 'blocked_replans',
            'stuck_events', 'landmark_updates', 'last_landmark_update_time_s', 'distance_travelled_m',
            'idle_time_s', 'visited_cell_count', 'manual_waypoint_count', 'remaining_path_points',
            'remaining_path_length_m', 'final_true_x', 'final_true_y', 'final_est_x', 'final_est_y',
        ],
        robot_rows_pretty,
    )

    coverage_plot_path = save_coverage_progress_plot(
        coverage_history,
        out_dir,
        ts=ts,
        mission_mode=metadata.get('mission_mode', 'manual_click'),
        auto_policy=metadata.get('auto_policy', 'frontier'),
    )

    generated_files = {
        'timestamp': ts or '',
        'coverage_timeline_csv': coverage_path,
        'event_timeline_csv': event_path,
        'robot_summary_csv': robot_path,
        'coverage_progress_png': coverage_plot_path,
        'trajectory_png': summary.get('last_saved_plot_path', ''),
        'observed_maps_png': summary.get('last_saved_map_path', ''),
        'run_metadata_json': str(out_dir / 'run_metadata.json'),
    }

    summary_path = out_dir / 'run_summary.json'
    summary_payload = _build_summary_payload(metadata, summary, robot_rows_pretty, generated_files)
    summary_path.write_text(json.dumps(summary_payload, indent=2))

    summary_text_path = out_dir / 'run_summary.txt'
    summary_text_path.write_text(_build_summary_text(metadata, summary, robot_rows_pretty, generated_files))

    return {
        'summary_json': str(summary_path),
        'summary_txt': str(summary_text_path),
        'coverage_timeline_csv': coverage_path,
        'event_timeline_csv': event_path,
        'robot_summary_csv': robot_path,
        'coverage_plot_png': coverage_plot_path,
    }

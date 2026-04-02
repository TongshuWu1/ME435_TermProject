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


def _fmt_float(value, digits=2):
    return f'{float(value):.{digits}f}'


def _safe_div(num, den):
    den = float(den)
    return float(num) / den if abs(den) > 1e-12 else 0.0


def _mission_name(metadata):
    return 'Auto Explore' if metadata.get('mission_mode') == 'auto_explore' else 'Manual Click'


def _policy_name(metadata):
    return 'Weighted Coverage' if metadata.get('auto_policy') == 'weighted_coverage' else 'Frontier'


def _stop_assessment(metadata, summary):
    mission_mode = metadata.get('mission_mode', 'manual_click')
    auto_running = bool(metadata.get('auto_mode_enabled_at_save', False))
    free_cov = float(summary.get('free_coverage_ratio', 0.0))
    occ_recall = float(summary.get('occupied_recall_ratio', 0.0))
    frontier_count = int(summary.get('frontier_count', 0))
    active_goals = int(summary.get('active_goals', 0))
    pending_manual_goals = int(summary.get('pending_manual_goals', 0))

    if bool(summary.get('auto_finished', False)):
        return {
            'outcome': 'Success',
            'code': 'success',
            'reason': 'Auto-finish condition was satisfied before the run was saved.',
        }

    if mission_mode == 'auto_explore':
        if frontier_count == 0 and active_goals > 0:
            return {
                'outcome': 'Partial Success',
                'code': 'no_frontiers_pending_goals',
                'reason': 'No frontier groups remained, but some active robot goals were still pending when outputs were saved.',
            }
        if frontier_count == 0:
            return {
                'outcome': 'Partial Success',
                'code': 'no_frontiers_remaining',
                'reason': 'No frontier groups remained, but the formal finish condition had not latched yet when outputs were saved.',
            }
        if free_cov >= 0.95:
            return {
                'outcome': 'Partial Success',
                'code': 'high_coverage_unfinished',
                'reason': 'Free-space coverage was already very high, but the mission had not officially finished at save time.',
            }
        if auto_running:
            return {
                'outcome': 'In Progress',
                'code': 'saved_while_running',
                'reason': 'Outputs were saved while the simulator was still running.',
            }
        if free_cov >= 0.60 or occ_recall >= 0.50:
            return {
                'outcome': 'Stopped Early',
                'code': 'stopped_mid_exploration',
                'reason': 'The run ended before the exploration objectives were completed.',
            }
        return {
            'outcome': 'Needs Attention',
            'code': 'low_progress',
            'reason': 'Coverage progress remained low at the time of save.',
        }

    if pending_manual_goals == 0:
        return {
            'outcome': 'Success',
            'code': 'manual_goals_complete',
            'reason': 'All manual waypoints were cleared before the run was saved.',
        }
    if auto_running:
        return {
            'outcome': 'In Progress',
            'code': 'manual_saved_while_running',
            'reason': 'Outputs were saved while robots were still executing manual waypoints.',
        }
    return {
        'outcome': 'Stopped Early',
        'code': 'manual_pending_goals',
        'reason': 'Some manual waypoints were still pending when outputs were saved.',
    }


def _robot_highlights(robot_rows):
    if not robot_rows:
        return {}

    def _pick_max(key):
        best = max(robot_rows, key=lambda row: float(row.get(key, 0.0)))
        return {
            'robot': best.get('robot', ''),
            'value': best.get(key, 0.0),
        }

    def _pick_min(key):
        best = min(robot_rows, key=lambda row: float(row.get(key, 0.0)))
        return {
            'robot': best.get('robot', ''),
            'value': best.get(key, 0.0),
        }

    return {
        'most_distance': _pick_max('distance_travelled_m'),
        'most_goals_reached': _pick_max('goals_reached'),
        'most_replan_attempts': _pick_max('replan_attempts'),
        'worst_final_pose_error': _pick_max('final_position_error_m'),
        'highest_mean_pose_error': _pick_max('mean_position_error_m'),
        'most_idle_fraction': _pick_max('idle_fraction'),
        'lowest_mean_pose_error': _pick_min('mean_position_error_m'),
    }


def _build_interpretation(summary):
    notes = []
    free_cov = float(summary.get('free_coverage_ratio', 0.0))
    occ_recall = float(summary.get('occupied_recall_ratio', 0.0))
    goals_assigned = int(summary.get('total_goal_assignments', 0))
    goals_reached = int(summary.get('total_goals_reached', 0))
    replan_attempts = int(summary.get('total_replan_attempts', 0))
    replan_failures = int(summary.get('total_replan_failures', 0))
    mean_final_err = float(summary.get('mean_final_position_error_m', 0.0))
    landmark_ratio = float(summary.get('landmark_discovery_ratio', 0.0))
    landmarks_total = int(summary.get('total_landmarks', 0))

    if free_cov >= 0.95 and occ_recall < 0.80:
        notes.append('Free space was covered more completely than obstacles were confirmed, so obstacle mapping is the main remaining gap.')
    if goals_assigned >= max(6, 2 * max(1, goals_reached)):
        notes.append('Goal assignment churn was high relative to goals actually reached, which suggests frequent retargeting or replan churn.')
    if replan_failures > 0:
        notes.append('Some replans failed, so blocked or stale goals likely contributed to wasted motion.')
    elif replan_attempts >= 20:
        notes.append('Replanning was active throughout the run; this is healthy only if it also translated into steady coverage gains.')
    if landmarks_total > 0 and landmark_ratio < 0.70:
        notes.append('Landmark discovery lagged behind spatial coverage, so robots explored area faster than they built a persistent landmark memory.')
    elif landmarks_total > 0 and landmark_ratio >= 0.90:
        notes.append('Most landmarks were discovered and remembered, so the semantic map stayed aligned with the explored area.')
    if mean_final_err >= 1.0:
        notes.append('Final localization error remained noticeable, so estimator drift should be checked alongside exploration performance.')
    if not notes:
        notes.append('Coverage, planning, and localization signals were broadly consistent at save time.')
    return notes


def _mission_kpis_row(metadata, summary):
    assessment = _stop_assessment(metadata, summary)
    return {
        'run_label': metadata.get('run_label', ''),
        'saved_at': metadata.get('saved_at', ''),
        'seed': metadata.get('seed', ''),
        'mission_mode': metadata.get('mission_mode', ''),
        'auto_policy': metadata.get('auto_policy', ''),
        'robot_count': metadata.get('robot_count', 0),
        'sim_time_s': round(float(metadata.get('sim_time_seconds', 0.0)), 3),
        'outcome': assessment.get('outcome', ''),
        'stop_reason_code': assessment.get('code', ''),
        'stop_reason': assessment.get('reason', ''),
        'map_known_pct': round(_pct(summary.get('known_ratio', 0.0)), 3),
        'free_space_covered_pct': round(_pct(summary.get('free_coverage_ratio', 0.0)), 3),
        'obstacle_cells_found_pct': round(_pct(summary.get('occupied_recall_ratio', 0.0)), 3),
        'landmarks_discovered_pct': round(_pct(summary.get('landmark_discovery_ratio', 0.0)), 3),
        'shared_landmarks_discovered': int(summary.get('shared_landmarks_discovered', 0)),
        'total_landmarks': int(summary.get('total_landmarks', 0)),
        'frontier_groups_at_end': int(summary.get('frontier_count', 0)),
        'active_goals_at_end': int(summary.get('active_goals', 0)),
        'pending_manual_goals': int(summary.get('pending_manual_goals', 0)),
        'total_distance_m': round(float(summary.get('total_distance_m', 0.0)), 3),
        'total_idle_time_s': round(float(summary.get('total_idle_time_s', 0.0)), 3),
        'coverage_rate_pct_per_s': round(float(summary.get('coverage_rate_pct_per_s', 0.0)), 3),
        'coverage_per_meter_pct_per_m': round(float(summary.get('coverage_per_meter_pct_per_m', 0.0)), 3),
        'obstacle_recall_rate_pct_per_s': round(float(summary.get('obstacle_recall_rate_pct_per_s', 0.0)), 3),
        'goal_completion_rate': round(float(summary.get('goal_completion_rate', 0.0)), 6),
        'replan_failure_rate': round(float(summary.get('replan_failure_rate', 0.0)), 6),
        'mean_position_error_m': round(float(summary.get('mean_position_error_m', 0.0)), 3),
        'rms_position_error_m': round(float(summary.get('rms_position_error_m', 0.0)), 3),
        'max_position_error_m': round(float(summary.get('max_position_error_m', 0.0)), 3),
        'mean_final_position_error_m': round(float(summary.get('mean_final_position_error_m', 0.0)), 3),
        'max_final_position_error_m': round(float(summary.get('max_final_position_error_m', 0.0)), 3),
        'total_goal_assignments': int(summary.get('total_goal_assignments', 0)),
        'total_goals_reached': int(summary.get('total_goals_reached', 0)),
        'goal_reassignments': int(summary.get('goal_reassignments', 0)),
        'total_replan_attempts': int(summary.get('total_replan_attempts', 0)),
        'total_replan_failures': int(summary.get('total_replan_failures', 0)),
        'total_stuck_events': int(summary.get('total_stuck_events', 0)),
        'total_landmark_updates': int(summary.get('total_landmark_updates', 0)),
        'auto_finished': bool(summary.get('auto_finished', False)),
    }


def _build_summary_payload(metadata, summary, robot_rows, landmark_rows, generated_files):
    assessment = _stop_assessment(metadata, summary)
    highlights = _robot_highlights(robot_rows)
    interpretation = _build_interpretation(summary)
    return {
        'run': {
            'label': metadata.get('run_label', ''),
            'saved_at': metadata.get('saved_at', ''),
            'seed': metadata.get('seed', ''),
            'mission_mode': metadata.get('mission_mode', ''),
            'mission_name': _mission_name(metadata),
            'auto_policy': metadata.get('auto_policy', ''),
            'policy_name': _policy_name(metadata),
            'robot_count': metadata.get('robot_count', 0),
            'sim_time_seconds': metadata.get('sim_time_seconds', 0.0),
            'auto_mode_enabled_at_save': metadata.get('auto_mode_enabled_at_save', False),
        },
        'outcome': assessment,
        'coverage': {
            'map_known_ratio': summary.get('known_ratio', 0.0),
            'map_known_percent': round(_pct(summary.get('known_ratio', 0.0)), 3),
            'free_space_covered_ratio': summary.get('free_coverage_ratio', 0.0),
            'free_space_covered_percent': round(_pct(summary.get('free_coverage_ratio', 0.0)), 3),
            'obstacle_cells_found_ratio': summary.get('occupied_recall_ratio', 0.0),
            'obstacle_cells_found_percent': round(_pct(summary.get('occupied_recall_ratio', 0.0)), 3),
            'landmark_discovery_ratio': summary.get('landmark_discovery_ratio', 0.0),
            'landmark_discovery_percent': round(_pct(summary.get('landmark_discovery_ratio', 0.0)), 3),
            'shared_landmarks_discovered': summary.get('shared_landmarks_discovered', 0),
            'total_landmarks': summary.get('total_landmarks', 0),
            'frontier_groups_at_end': summary.get('frontier_count', 0),
            'active_goals_at_end': summary.get('active_goals', 0),
            'pending_manual_goals': summary.get('pending_manual_goals', 0),
            'auto_finished': summary.get('auto_finished', False),
        },
        'efficiency': {
            'total_distance_m': summary.get('total_distance_m', 0.0),
            'total_idle_time_s': summary.get('total_idle_time_s', 0.0),
            'team_idle_fraction': summary.get('team_idle_fraction', 0.0),
            'coverage_rate_pct_per_s': summary.get('coverage_rate_pct_per_s', 0.0),
            'coverage_per_meter_pct_per_m': summary.get('coverage_per_meter_pct_per_m', 0.0),
            'obstacle_recall_rate_pct_per_s': summary.get('obstacle_recall_rate_pct_per_s', 0.0),
            'total_goal_assignments': summary.get('total_goal_assignments', 0),
            'total_goals_reached': summary.get('total_goals_reached', 0),
            'goal_reassignments': summary.get('goal_reassignments', 0),
            'goal_completion_rate': summary.get('goal_completion_rate', 0.0),
            'total_replan_attempts': summary.get('total_replan_attempts', 0),
            'total_replan_failures': summary.get('total_replan_failures', 0),
            'replan_failure_rate': summary.get('replan_failure_rate', 0.0),
            'total_stuck_events': summary.get('total_stuck_events', 0),
            'total_landmark_updates': summary.get('total_landmark_updates', 0),
        },
        'localization': {
            'mean_position_error_m': summary.get('mean_position_error_m', 0.0),
            'rms_position_error_m': summary.get('rms_position_error_m', 0.0),
            'max_position_error_m': summary.get('max_position_error_m', 0.0),
            'mean_final_position_error_m': summary.get('mean_final_position_error_m', 0.0),
            'max_final_position_error_m': summary.get('max_final_position_error_m', 0.0),
        },
        'robot_highlights': highlights,
        'interpretation': interpretation,
        'logging': {
            'event_count': summary.get('event_count', 0),
            'coverage_samples': summary.get('coverage_samples', 0),
        },
        'robots': robot_rows,
        'landmarks': landmark_rows,
        'files': generated_files,
    }


def _build_summary_text(metadata, summary, robot_rows, landmark_rows, generated_files):
    assessment = _stop_assessment(metadata, summary)
    highlights = _robot_highlights(robot_rows)
    interpretation = _build_interpretation(summary)

    lines = [
        'ME435 Simulator Run Summary',
        '=' * 32,
        '',
        f"Outcome : {assessment.get('outcome', '')}",
        f"Reason  : {assessment.get('reason', '')}",
        f"Mode    : {_mission_name(metadata)} / {_policy_name(metadata)}",
        f"Run     : {metadata.get('run_label', '')}",
        f"Saved   : {metadata.get('saved_at', '')}",
        f"Seed    : {metadata.get('seed', '')}",
        f"Robots  : {metadata.get('robot_count', 0)}",
        f"Time    : {_fmt_float(metadata.get('sim_time_seconds', 0.0), 2)} s",
        '',
        'Mission KPIs',
        '-' * 12,
        f"Map known              : {_fmt_pct(summary.get('known_ratio', 0.0))}",
        f"Free-space coverage    : {_fmt_pct(summary.get('free_coverage_ratio', 0.0))}",
        f"Obstacle recall        : {_fmt_pct(summary.get('occupied_recall_ratio', 0.0))}",
        f"Landmarks discovered   : {int(summary.get('shared_landmarks_discovered', 0))}/{int(summary.get('total_landmarks', 0))} ({_fmt_pct(summary.get('landmark_discovery_ratio', 0.0))})",
        f"Frontier groups left   : {int(summary.get('frontier_count', 0))}",
        f"Active goals left      : {int(summary.get('active_goals', 0))}",
        f"Pending manual goals   : {int(summary.get('pending_manual_goals', 0))}",
        f"Total travel distance  : {_fmt_float(summary.get('total_distance_m', 0.0), 2)} m",
        f"Total idle time        : {_fmt_float(summary.get('total_idle_time_s', 0.0), 2)} s",
        f"Coverage rate          : {_fmt_float(summary.get('coverage_rate_pct_per_s', 0.0), 2)} %/s",
        f"Coverage per meter     : {_fmt_float(summary.get('coverage_per_meter_pct_per_m', 0.0), 3)} %/m",
        f"Obstacle recall rate   : {_fmt_float(summary.get('obstacle_recall_rate_pct_per_s', 0.0), 2)} %/s",
        f"Goal assignments       : {int(summary.get('total_goal_assignments', 0))}",
        f"Goals reached          : {int(summary.get('total_goals_reached', 0))}",
        f"Goal reassignments     : {int(summary.get('goal_reassignments', 0))}",
        f"Goal completion rate   : {_fmt_pct(summary.get('goal_completion_rate', 0.0))}",
        f"Replan attempts        : {int(summary.get('total_replan_attempts', 0))}",
        f"Replan failures        : {int(summary.get('total_replan_failures', 0))}",
        f"Replan failure rate    : {_fmt_pct(summary.get('replan_failure_rate', 0.0))}",
        f"Stuck recoveries       : {int(summary.get('total_stuck_events', 0))}",
        '',
        'Localization',
        '-' * 12,
        f"Mean pose error        : {_fmt_float(summary.get('mean_position_error_m', 0.0), 3)} m",
        f"RMS pose error         : {_fmt_float(summary.get('rms_position_error_m', 0.0), 3)} m",
        f"Max pose error         : {_fmt_float(summary.get('max_position_error_m', 0.0), 3)} m",
        f"Mean final pose error  : {_fmt_float(summary.get('mean_final_position_error_m', 0.0), 3)} m",
        f"Worst final pose error : {_fmt_float(summary.get('max_final_position_error_m', 0.0), 3)} m",
        '',
        'Robot Highlights',
        '-' * 16,
    ]

    if highlights:
        lines.extend([
            f"Most distance          : {highlights['most_distance']['robot']} ({_fmt_float(highlights['most_distance']['value'], 2)} m)",
            f"Most goals reached     : {highlights['most_goals_reached']['robot']} ({int(float(highlights['most_goals_reached']['value']))})",
            f"Most replans           : {highlights['most_replan_attempts']['robot']} ({int(float(highlights['most_replan_attempts']['value']))})",
            f"Worst final pose error : {highlights['worst_final_pose_error']['robot']} ({_fmt_float(highlights['worst_final_pose_error']['value'], 3)} m)",
            f"Most idle              : {highlights['most_idle_fraction']['robot']} ({_fmt_pct(highlights['most_idle_fraction']['value'])})",
        ])
    else:
        lines.append('No robot statistics were available.')

    lines.extend([
        '',
        'Per-robot summary',
        '-' * 17,
    ])
    for row in robot_rows:
        lines.append(
            f"- {row.get('robot', 'Robot')}: phase={row.get('phase', '')}, goal_type={row.get('last_goal_type', '')}, "
            f"goals={row.get('goals_reached', 0)}/{row.get('goal_assignments', 0)}, "
            f"replans={row.get('replan_attempts', 0)} ({row.get('replan_failures', 0)} failed), "
            f"landmarks={row.get('landmarks_discovered', 0)} known, "
            f"distance={_fmt_float(row.get('distance_travelled_m', 0.0), 2)} m, "
            f"mean_err={_fmt_float(row.get('mean_position_error_m', 0.0), 3)} m, "
            f"final_err={_fmt_float(row.get('final_position_error_m', 0.0), 3)} m"
        )

    lines.extend([
        '',
        'Landmark memory',
        '-' * 15,
    ])
    if landmark_rows:
        for row in landmark_rows:
            lines.append(
                f"- L{int(row.get('landmark_id', 0)):02d}: {row.get('color_name', '')} {row.get('shape', '')} at "
                f"({_fmt_float(row.get('x_m', 0.0), 2)}, {_fmt_float(row.get('y_m', 0.0), 2)}), "
                f"seen {int(row.get('seen_count', 0))}x by {row.get('seen_by_robots', '') or row.get('first_seen_by', '')}"
            )
    else:
        lines.append('No landmarks were discovered during this run.')

    lines.extend([
        '',
        'Interpretation',
        '-' * 14,
    ])
    for note in interpretation:
        lines.append(f'- {note}')

    lines.extend([
        '',
        'Saved files',
        '-' * 11,
        '- run_summary.txt: human-readable run verdict and KPIs',
        '- run_summary.json: structured summary for scripts',
        '- mission_kpis.csv: one-row summary for comparing runs',
        '- robot_summary.csv: one row per robot',
        '- landmark_summary.csv: one row per discovered landmark',
        '- coverage_timeline.csv: coverage versus time',
        '- event_timeline.csv: timestamped events',
        f"- coverage_progress{'_' + generated_files.get('timestamp', '') if generated_files.get('timestamp') else ''}.png: coverage chart",
        f"- trajectories{'_' + generated_files.get('timestamp', '') if generated_files.get('timestamp') else ''}.png: world trajectories",
        f"- observed_maps{'_' + generated_files.get('timestamp', '') if generated_files.get('timestamp') else ''}.png: shared + per-robot maps",
    ])
    return '\n'.join(lines) + '\n'


def save_run_reports(*, out_dir, metadata, coverage_history, event_log, robot_rows, landmark_rows=None, summary, ts=None):
    out_dir = ensure_directory(out_dir)

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
            'replan_attempts': row.get('replan_attempts', row.get('replans', '')),
            'replan_failures': row.get('replan_failures', row.get('replans_blocked', '')),
            'replan_successes': row.get('replan_successes', ''),
            'replan_failure_rate': row.get('replan_failure_rate', ''),
            'stuck_events': row.get('stuck_events', ''),
            'landmark_updates': row.get('measurement_updates', ''),
            'landmarks_discovered': row.get('landmarks_discovered', ''),
            'currently_visible_landmarks': row.get('currently_visible_landmarks', ''),
            'last_landmark_update_time_s': row.get('last_landmark_update_time', ''),
            'distance_travelled_m': row.get('distance_travelled_m', ''),
            'idle_time_s': row.get('idle_time_s', ''),
            'idle_fraction': row.get('idle_fraction', ''),
            'visited_cell_count': row.get('visited_cell_count', ''),
            'manual_waypoint_count': row.get('path_waypoint_count', ''),
            'remaining_path_points': row.get('remaining_path_points', ''),
            'remaining_path_length_m': row.get('remaining_path_length_m', ''),
            'mean_position_error_m': row.get('mean_position_error_m', ''),
            'rms_position_error_m': row.get('rms_position_error_m', ''),
            'max_position_error_m': row.get('max_position_error_m', ''),
            'final_position_error_m': row.get('final_position_error_m', ''),
            'final_true_x': row.get('final_true_x', ''),
            'final_true_y': row.get('final_true_y', ''),
            'final_est_x': row.get('final_est_x', ''),
            'final_est_y': row.get('final_est_y', ''),
        })

    landmark_rows_pretty = []
    for row in (landmark_rows or []):
        landmark_rows_pretty.append({
            'landmark_id': row.get('landmark_id', ''),
            'shape': row.get('shape', ''),
            'color_name': row.get('color_name', ''),
            'x_m': row.get('x_m', ''),
            'y_m': row.get('y_m', ''),
            'size_m': row.get('size_m', ''),
            'seen_count': row.get('seen_count', ''),
            'first_seen_time_s': row.get('first_seen_time_s', ''),
            'last_seen_time_s': row.get('last_seen_time_s', ''),
            'first_seen_by': row.get('first_seen_by', ''),
            'last_seen_by': row.get('last_seen_by', ''),
            'seen_by_count': row.get('seen_by_count', ''),
            'seen_by_robots': row.get('seen_by_robots', ''),
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
            'robot', 'phase', 'last_goal_type', 'goal_assignments', 'goals_reached', 'replan_attempts',
            'replan_failures', 'replan_successes', 'replan_failure_rate', 'stuck_events', 'landmark_updates',
            'landmarks_discovered', 'currently_visible_landmarks', 'last_landmark_update_time_s', 'distance_travelled_m', 'idle_time_s', 'idle_fraction',
            'visited_cell_count', 'manual_waypoint_count', 'remaining_path_points', 'remaining_path_length_m',
            'mean_position_error_m', 'rms_position_error_m', 'max_position_error_m', 'final_position_error_m',
            'final_true_x', 'final_true_y', 'final_est_x', 'final_est_y',
        ],
        robot_rows_pretty,
    )

    landmark_path = write_csv(
        out_dir / 'landmark_summary.csv',
        [
            'landmark_id', 'shape', 'color_name', 'x_m', 'y_m', 'size_m', 'seen_count',
            'first_seen_time_s', 'last_seen_time_s', 'first_seen_by', 'last_seen_by', 'seen_by_count', 'seen_by_robots',
        ],
        landmark_rows_pretty,
    )

    mission_kpis_path = write_csv(
        out_dir / 'mission_kpis.csv',
        [
            'run_label', 'saved_at', 'seed', 'mission_mode', 'auto_policy', 'robot_count', 'sim_time_s',
            'outcome', 'stop_reason_code', 'stop_reason', 'map_known_pct', 'free_space_covered_pct',
            'obstacle_cells_found_pct', 'landmarks_discovered_pct', 'shared_landmarks_discovered', 'total_landmarks', 'frontier_groups_at_end', 'active_goals_at_end', 'pending_manual_goals',
            'total_distance_m', 'total_idle_time_s', 'coverage_rate_pct_per_s', 'coverage_per_meter_pct_per_m',
            'obstacle_recall_rate_pct_per_s', 'goal_completion_rate', 'replan_failure_rate',
            'mean_position_error_m', 'rms_position_error_m', 'max_position_error_m',
            'mean_final_position_error_m', 'max_final_position_error_m', 'total_goal_assignments',
            'total_goals_reached', 'goal_reassignments', 'total_replan_attempts', 'total_replan_failures',
            'total_stuck_events', 'total_landmark_updates', 'auto_finished',
        ],
        [_mission_kpis_row(metadata, summary)],
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
        'landmark_summary_csv': landmark_path,
        'mission_kpis_csv': mission_kpis_path,
        'coverage_progress_png': coverage_plot_path,
        'trajectory_png': summary.get('last_saved_plot_path', ''),
        'observed_maps_png': summary.get('last_saved_map_path', ''),
        'run_metadata_json': str(out_dir / 'run_metadata.json'),
    }

    summary_path = out_dir / 'run_summary.json'
    summary_payload = _build_summary_payload(metadata, summary, robot_rows_pretty, landmark_rows_pretty, generated_files)
    summary_path.write_text(json.dumps(summary_payload, indent=2))

    summary_text_path = out_dir / 'run_summary.txt'
    summary_text_path.write_text(_build_summary_text(metadata, summary, robot_rows_pretty, landmark_rows_pretty, generated_files))

    return {
        'summary_json': str(summary_path),
        'summary_txt': str(summary_text_path),
        'coverage_timeline_csv': coverage_path,
        'event_timeline_csv': event_path,
        'robot_summary_csv': robot_path,
        'landmark_summary_csv': landmark_path,
        'mission_kpis_csv': mission_kpis_path,
        'coverage_plot_png': coverage_plot_path,
    }

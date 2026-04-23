import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.widgets import Button, RadioButtons, TextBox
from matplotlib import patches

from config import DRONE_NAMES, WORLD_HEIGHT_METERS, WORLD_WIDTH_METERS

EDIT_MODE_LABELS = ['Add Waypoint', 'Set Start']
MISSION_MODE_LABELS = ['Manual Click', 'Auto Explore']
AUTO_POLICY_LABELS = ['Frontier', 'Weighted Coverage']


class SimulatorUI:
    """Builds and manages the Matplotlib canvas / widgets for the simulator."""

    def __init__(self, sim):
        self.sim = sim
        self.fig = None
        self.ax = None
        self.status_ax = None
        self.status_text = None
        self.toggle_button = None
        self.reset_button = None
        self.apply_seed_button = None
        self.clear_button = None
        self.partition_button = None
        self.density_button = None
        self.uncertainty_button = None
        self.seed_box = None
        self.robot_selector = None
        self.mode_selector = None
        self.mission_selector = None
        self.auto_policy_selector = None
        self.shared_map_ax = None
        self.shared_map_image = None
        self.shared_uncertainty_image = None
        self.shared_robot_scatter = None
        self.shared_landmark_scatter = None
        self.shared_los_collection = None
        self.shared_base_patch = None
        self.partition_image = None
        self.partition_generator_scatter = None
        self.partition_centroid_scatter = None
        self.density_image = None
        self.uncertainty_image = None
        self.robot_monitor_fig = None
        self.robot_monitor_entries = []
        self._monitor_cmap = ListedColormap(['#d9dde3', '#f8fbff', '#5b6470'])
        self._monitor_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], self._monitor_cmap.N)

    def _make_panel(self, bounds, *, title=None, facecolor='#f5f5f5', title_fontsize=9):
        ax = self.fig.add_axes(bounds)
        ax.set_facecolor(facecolor)
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
            spine.set_edgecolor('#b8c0cc')
        if title:
            ax.set_title(title, fontsize=title_fontsize, pad=3)
        return ax

    def build(self):
        self.fig = plt.figure(figsize=(14.8, 8.9))
        self.ax = self.fig.add_axes([0.05, 0.08, 0.60, 0.86])
        self.ax.set_xlim(-0.35, WORLD_WIDTH_METERS + 0.35)
        self.ax.set_ylim(-0.35, WORLD_HEIGHT_METERS + 0.35)
        self.ax.set_aspect('equal')
        self.ax.set_title('Multi-Robot Search / Mapping', pad=10)
        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Y (meters)')
        self.ax.grid(True, alpha=0.14)

        self.density_image = self.ax.imshow(
            self.sim.density_rgba,
            origin='lower',
            interpolation='nearest',
            extent=[0.0, WORLD_WIDTH_METERS, 0.0, WORLD_HEIGHT_METERS],
            alpha=1.0,
            zorder=0.55,
            visible=self.sim.show_density_overlay,
        )

        self.uncertainty_image = self.ax.imshow(
            self.sim.uncertainty_rgba,
            origin='lower',
            interpolation='nearest',
            extent=[0.0, WORLD_WIDTH_METERS, 0.0, WORLD_HEIGHT_METERS],
            alpha=1.0,
            zorder=0.575,
            visible=self.sim.show_uncertainty_overlay,
        )

        self.partition_image = self.ax.imshow(
            self.sim.partition_rgba,
            origin='lower',
            interpolation='nearest',
            extent=[0.0, WORLD_WIDTH_METERS, 0.0, WORLD_HEIGHT_METERS],
            alpha=1.0,
            zorder=0.6,
            visible=self.sim.show_partition_overlay,
        )
        self.partition_generator_scatter = self.ax.scatter(
            [], [], s=55, marker='x', c=[], linewidths=1.7, zorder=7,
            visible=self.sim.show_partition_overlay,
        )
        self.partition_centroid_scatter = self.ax.scatter(
            [], [], s=38, marker='+', c=[], linewidths=1.6, zorder=7,
            visible=self.sim.show_partition_overlay,
        )

        self.status_ax = self._make_panel([0.68, 0.74, 0.28, 0.22], title='Run Status', facecolor='#fbfcfd')
        self.status_ax.set_xticks([])
        self.status_ax.set_yticks([])
        self.status_ax.set_xlim(0.0, 1.0)
        self.status_ax.set_ylim(0.0, 1.0)
        self.status_text = self.status_ax.text(
            0.03,
            0.97,
            '',
            transform=self.status_ax.transAxes,
            fontsize=8.5,
            verticalalignment='top',
            horizontalalignment='left',
            family='monospace',
            clip_on=True,
        )

        mission_ax = self._make_panel([0.68, 0.62, 0.135, 0.10], title='Mission Mode')
        self.mission_selector = RadioButtons(mission_ax, MISSION_MODE_LABELS, active=0 if self.sim.mission_mode == 'manual_click' else 1)
        self.mission_selector.on_clicked(self.sim.on_select_mission_mode)

        auto_policy_ax = self._make_panel([0.825, 0.62, 0.135, 0.10], title='Auto Policy')
        self.auto_policy_selector = RadioButtons(auto_policy_ax, AUTO_POLICY_LABELS, active=0 if getattr(self.sim, 'auto_policy', 'frontier') == 'frontier' else 1)
        self.auto_policy_selector.on_clicked(self.sim.on_select_auto_policy)

        robot_ax = self._make_panel([0.68, 0.44, 0.135, 0.15], title='Robot')
        self.robot_selector = RadioButtons(robot_ax, DRONE_NAMES, active=0)
        self.robot_selector.on_clicked(self.sim.on_select_robot)

        mode_ax = self._make_panel([0.825, 0.44, 0.135, 0.15], title='Click Action')
        self.mode_selector = RadioButtons(mode_ax, EDIT_MODE_LABELS, active=0)
        self.mode_selector.on_clicked(self.sim.on_select_edit_mode)

        toggle_ax = self.fig.add_axes([0.68, 0.365, 0.062, 0.052])
        self.toggle_button = Button(toggle_ax, '')
        self.toggle_button.on_clicked(self.sim.toggle_auto_mode)

        self.partition_button = Button(self.fig.add_axes([0.749, 0.365, 0.062, 0.052]), '')
        self.density_button = Button(self.fig.add_axes([0.818, 0.365, 0.062, 0.052]), '')
        self.uncertainty_button = Button(self.fig.add_axes([0.887, 0.365, 0.073, 0.052]), '')
        self.partition_button.on_clicked(self.sim.toggle_partition_overlay)
        self.density_button.on_clicked(self.sim.toggle_density_overlay)
        self.uncertainty_button.on_clicked(self.sim.toggle_uncertainty_overlay)

        clear_ax = self.fig.add_axes([0.68, 0.295, 0.28, 0.052])
        self.clear_button = Button(clear_ax, 'Clear Path / Goal')
        self.clear_button.on_clicked(self.sim.clear_selected_path)

        self.shared_map_ax = self._make_panel([0.68, 0.10, 0.18, 0.17], title='Shared Map', facecolor='#fbfcfd')
        self.shared_map_ax.set_xlim(0.0, WORLD_WIDTH_METERS)
        self.shared_map_ax.set_ylim(0.0, WORLD_HEIGHT_METERS)
        self.shared_map_ax.set_aspect('equal')
        self.shared_map_ax.set_xticks([])
        self.shared_map_ax.set_yticks([])
        base = self.sim.home_base
        self.shared_base_patch = patches.Rectangle(
            (base["cx"] - base["width"] / 2.0, base["cy"] - base["height"] / 2.0),
            base["width"],
            base["height"],
            facecolor=(0.18, 0.60, 0.32, 0.16),
            edgecolor=(0.12, 0.45, 0.22, 0.95),
            linewidth=1.0,
            linestyle='--',
            zorder=0.9,
        )
        self.shared_map_ax.add_patch(self.shared_base_patch)
        cmap = ListedColormap(['#d9dde3', '#f8fbff', '#5b6470'])
        norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
        self.shared_map_image = self.shared_map_ax.imshow(
            self.sim.shared_known_grid,
            origin='lower',
            interpolation='nearest',
            cmap=cmap,
            norm=norm,
            extent=[0.0, WORLD_WIDTH_METERS, 0.0, WORLD_HEIGHT_METERS],
        )
        self.shared_uncertainty_image = self.shared_map_ax.imshow(
            self.sim.uncertainty_rgba,
            origin='lower',
            interpolation='nearest',
            extent=[0.0, WORLD_WIDTH_METERS, 0.0, WORLD_HEIGHT_METERS],
            alpha=1.0,
            visible=self.sim.show_uncertainty_overlay,
        )
        self.shared_robot_scatter = self.shared_map_ax.scatter([], [], s=28, c=[], edgecolors='black', linewidths=0.5)
        self.shared_landmark_scatter = self.shared_map_ax.scatter([], [], s=52, marker='*', c='#FFCC00', edgecolors='black', linewidths=0.6)
        self.shared_los_collection = LineCollection([], colors=[(0.10, 0.55, 0.95, 0.55)], linewidths=1.6, zorder=4)
        self.shared_map_ax.add_collection(self.shared_los_collection)

        seed_box_ax = self.fig.add_axes([0.88, 0.19, 0.08, 0.05])
        self.seed_box = TextBox(seed_box_ax, 'Seed ', initial=str(self.sim.current_seed))

        apply_seed_ax = self.fig.add_axes([0.88, 0.13, 0.08, 0.05])
        self.apply_seed_button = Button(apply_seed_ax, 'Apply')
        self.apply_seed_button.on_clicked(self.sim.apply_seed_from_box)

        reset_ax = self.fig.add_axes([0.88, 0.07, 0.08, 0.05])
        self.reset_button = Button(reset_ax, 'Reset')
        self.reset_button.on_clicked(self.sim.reset_simulation)

        self.fig.canvas.mpl_connect('button_press_event', self.sim.on_map_click)

        self.refresh_all()
        return self.fig, self.ax

    def build_robot_monitor(self, drones):
        if self.robot_monitor_fig is not None:
            try:
                plt.close(self.robot_monitor_fig)
            except Exception:
                pass
        count = max(1, len(drones))
        self.robot_monitor_fig = plt.figure(figsize=(10.5, max(4.0, 2.6 * count)))
        self.robot_monitor_fig.suptitle('Robot Monitor', fontsize=12)
        gs = self.robot_monitor_fig.add_gridspec(count, 2, width_ratios=[1.2, 1.0], hspace=0.28, wspace=0.12)
        self.robot_monitor_entries = []
        for idx, drone in enumerate(drones):
            map_ax = self.robot_monitor_fig.add_subplot(gs[idx, 0])
            text_ax = self.robot_monitor_fig.add_subplot(gs[idx, 1])
            map_ax.set_xlim(0.0, WORLD_WIDTH_METERS)
            map_ax.set_ylim(0.0, WORLD_HEIGHT_METERS)
            map_ax.set_aspect('equal')
            map_ax.set_xticks([])
            map_ax.set_yticks([])
            map_ax.set_title(f"{drone['name']} local map", fontsize=9, pad=3)
            base = self.sim.home_base
            map_ax.add_patch(patches.Rectangle(
                (base["cx"] - base["width"] / 2.0, base["cy"] - base["height"] / 2.0),
                base["width"],
                base["height"],
                facecolor=(0.18, 0.60, 0.32, 0.14),
                edgecolor=(0.12, 0.45, 0.22, 0.90),
                linewidth=0.9,
                linestyle='--',
                zorder=0.9,
            ))
            image = map_ax.imshow(
                drone['local_known_grid'],
                origin='lower',
                interpolation='nearest',
                cmap=self._monitor_cmap,
                norm=self._monitor_norm,
                extent=[0.0, WORLD_WIDTH_METERS, 0.0, WORLD_HEIGHT_METERS],
            )
            uncertainty_image = map_ax.imshow(
                self.sim._local_uncertainty_rgba(drone),
                origin='lower',
                interpolation='nearest',
                extent=[0.0, WORLD_WIDTH_METERS, 0.0, WORLD_HEIGHT_METERS],
                alpha=1.0,
                visible=self.sim.show_uncertainty_overlay,
                zorder=3,
            )
            est_marker = map_ax.scatter([], [], s=42, c=[drone['color']], edgecolors='black', linewidths=0.6, zorder=5)
            goal_marker = map_ax.scatter([], [], s=85, marker='*', c=[drone['color']], edgecolors='black', linewidths=0.6, zorder=6)
            landmark_scatter = map_ax.scatter([], [], s=38, marker='^', c='#FFCC00', edgecolors='black', linewidths=0.5, zorder=5)
            plan_line, = map_ax.plot([], [], color=drone['color'], linewidth=1.5, alpha=0.95, zorder=4)
            los_collection = LineCollection([], colors=[(*drone['color'][:3], 0.65)], linewidths=1.4, zorder=4.5)
            map_ax.add_collection(los_collection)

            text_ax.set_xticks([])
            text_ax.set_yticks([])
            text_ax.set_facecolor('#fbfcfd')
            for spine in text_ax.spines.values():
                spine.set_linewidth(1.0)
                spine.set_edgecolor('#b8c0cc')
            text_artist = text_ax.text(
                0.03, 0.97, '', transform=text_ax.transAxes, fontsize=8.5,
                verticalalignment='top', horizontalalignment='left', family='monospace', clip_on=True
            )
            self.robot_monitor_entries.append({
                'map_ax': map_ax,
                'text_ax': text_ax,
                'image': image,
                'uncertainty_image': uncertainty_image,
                'est_marker': est_marker,
                'goal_marker': goal_marker,
                'landmark_scatter': landmark_scatter,
                'plan_line': plan_line,
                'los_collection': los_collection,
                'text_artist': text_artist,
            })
        self.refresh_robot_monitor()

    def refresh_robot_monitor(self):
        if self.robot_monitor_fig is None:
            return
        drones = getattr(self.sim, 'drones', [])
        for idx, entry in enumerate(self.robot_monitor_entries):
            if idx >= len(drones):
                continue
            drone = drones[idx]
            entry['image'].set_data(drone['local_known_grid'])
            if self.sim.show_uncertainty_overlay:
                entry['uncertainty_image'].set_data(self.sim._local_uncertainty_rgba(drone))
            entry['uncertainty_image'].set_visible(bool(self.sim.show_uncertainty_overlay))
            est_x, est_y, est_theta = drone['odometry'].mu
            entry['est_marker'].set_offsets(np.array([[float(est_x), float(est_y)]]))

            goal_xy = drone.get('auto_goal_xy')
            if goal_xy is None and self.sim.mission_mode == 'manual_click':
                path = drone.get('path', [])
                goal_xy = tuple(path[-1]) if path else None
            if goal_xy is not None:
                entry['goal_marker'].set_offsets(np.array([[float(goal_xy[0]), float(goal_xy[1])]]))
                entry['goal_marker'].set_sizes([85.0])
            else:
                entry['goal_marker'].set_offsets(np.array([[0.0, 0.0]]))
                entry['goal_marker'].set_sizes([0.0])

            plan_pts = list(drone.get('planned_path', []))
            if plan_pts:
                entry['plan_line'].set_data([p[0] for p in plan_pts], [p[1] for p in plan_pts])
            else:
                entry['plan_line'].set_data([], [])

            entry['los_collection'].set_segments(list(drone.get('visible_segments_est', [])))

            rows = list(drone.get('known_landmarks', {}).values())
            if rows:
                offsets = np.array([[float(r.get('x', 0.0)), float(r.get('y', 0.0))] for r in rows])
                entry['landmark_scatter'].set_offsets(offsets)
                entry['landmark_scatter'].set_sizes([38.0] * len(rows))
            else:
                entry['landmark_scatter'].set_offsets(np.array([[0.0, 0.0]]))
                entry['landmark_scatter'].set_sizes([0.0])

            true_x, true_y = drone['robot'].x, drone['robot'].y
            pos_err = ((float(true_x) - float(est_x)) ** 2 + (float(true_y) - float(est_y)) ** 2) ** 0.5
            remaining_length = self.sim._remaining_path_length(drone)
            mode = 'AUTO' if self.sim.mission_mode == 'auto_explore' else 'MANUAL'
            role = drone.get('auto_phase', 'manual')
            goal_type = self.sim._goal_type_for_drone(drone)
            local_known_ratio = float((drone['local_known_grid'] != 0).sum()) / float(drone['local_known_grid'].size)
            local_conf_mean = float(np.mean(drone.get('local_confidence_grid', np.zeros_like(drone['local_known_grid'], dtype=float))))
            local_pose_unc = np.asarray(drone.get('local_pose_uncertainty_grid', np.full_like(drone['local_known_grid'], np.inf, dtype=float)), dtype=float)
            finite_unc = local_pose_unc[np.isfinite(local_pose_unc)]
            mean_pose_unc = float(np.mean(finite_unc)) if finite_unc.size else float('nan')
            pose_trace = float(np.trace(np.asarray(drone['odometry'].cov)[:2, :2]))
            info_lines = [
                f"Mode     : {mode}",
                f"Role     : {role}",
                f"GoalType : {goal_type}",
                f"Est Pose : ({float(est_x):4.1f}, {float(est_y):4.1f}, {float(est_theta):5.1f})",
                f"Pose Err : {pos_err:4.2f} m",
                f"Pose Tr  : {pose_trace:5.3f}",
                f"Path Rem : {remaining_length:4.1f} m",
                f"Map Known: {100.0 * local_known_ratio:5.1f}%",
                f"Map Conf : {local_conf_mean:5.2f}",
                f"Map Unc  : {mean_pose_unc:5.2f} m",
                f"Goals    : {drone.get('goal_reached_count', 0)} / {drone.get('goal_assignment_count', 0)}",
                f"Replans  : {drone.get('replan_count', 0)}   Stuck: {drone.get('stuck_events', 0)}",
                f"Landmarks: {len(drone.get('known_landmarks', {}))} known, {len(drone.get('last_detected_landmarks', []))} visible",
                f"LOS Peers : {len(drone.get('visible_teammates', []))} -> {', '.join(drone.get('visible_teammates', [])[:3]) if drone.get('visible_teammates', []) else '-'}",
            ]
            if goal_xy is not None:
                info_lines.insert(3, f"Goal     : ({float(goal_xy[0]):4.1f}, {float(goal_xy[1]):4.1f})")
            else:
                info_lines.insert(3, "Goal     : none")
            entry['text_artist'].set_text('\n'.join(info_lines))
        try:
            self.robot_monitor_fig.canvas.draw_idle()
        except Exception:
            pass

    def refresh_all(self):
        self.refresh_toggle_button()
        self.refresh_partition_button()
        self.refresh_density_button()
        self.refresh_uncertainty_button()
        self.refresh_status_text()
        self.refresh_shared_map()
        self.refresh_partition_overlay()
        self.refresh_robot_monitor()

    def refresh_toggle_button(self):
        if self.toggle_button is not None:
            self.toggle_button.label.set_text('Pause' if self.sim.auto_mode else 'Start')

    def refresh_partition_button(self):
        if self.partition_button is not None:
            self.partition_button.label.set_text('Hide Region' if self.sim.show_partition_overlay else 'Show Region')

    def refresh_density_button(self):
        if self.density_button is not None:
            self.density_button.label.set_text('Hide Density' if self.sim.show_density_overlay else 'Show Density')

    def refresh_uncertainty_button(self):
        if self.uncertainty_button is not None:
            self.uncertainty_button.label.set_text('Hide Unc' if self.sim.show_uncertainty_overlay else 'Show Unc')

    def refresh_status_text(self):
        if self.status_text is None:
            return
        self.status_text.set_text(self.sim.build_status_text())

    def sync_seed_box(self):
        if self.seed_box is not None:
            self.seed_box.set_val(str(self.sim.current_seed))

    def refresh_shared_map(self):
        if self.shared_map_image is not None:
            self.shared_map_image.set_data(self.sim.shared_known_grid)
        if self.shared_uncertainty_image is not None:
            if self.sim.show_uncertainty_overlay:
                self.shared_uncertainty_image.set_data(self.sim._shared_uncertainty_rgba())
            self.shared_uncertainty_image.set_visible(bool(self.sim.show_uncertainty_overlay))
        if self.shared_robot_scatter is not None and getattr(self.sim, 'drones', None):
            offsets = [[float(d['odometry'].mu[0]), float(d['odometry'].mu[1])] for d in self.sim.drones]
            colors = [d['color'] for d in self.sim.drones]
            self.shared_robot_scatter.set_offsets(offsets)
            self.shared_robot_scatter.set_facecolors(colors)
        if self.shared_los_collection is not None:
            self.shared_los_collection.set_segments(list(getattr(self.sim, 'shared_los_segments', [])))
        if self.shared_landmark_scatter is not None:
            rows = list(getattr(self.sim, 'shared_known_landmarks', {}).values())
            if rows:
                offsets = [[float(r.get('x', 0.0)), float(r.get('y', 0.0))] for r in rows]
                colors = ['#FFCC00' if str(r.get('color_name', r.get('color', 'yellow'))) == 'yellow' else '#FF8C00' for r in rows]
                self.shared_landmark_scatter.set_offsets(offsets)
                self.shared_landmark_scatter.set_facecolors(colors)
                self.shared_landmark_scatter.set_sizes([52.0] * len(rows))
            else:
                self.shared_landmark_scatter.set_offsets([[0.0, 0.0]])
                self.shared_landmark_scatter.set_sizes([0.0])

    def refresh_partition_overlay(self):
        visible = bool(self.sim.show_partition_overlay)
        if self.density_image is not None:
            self.density_image.set_data(self.sim.density_rgba)
            self.density_image.set_visible(bool(self.sim.show_density_overlay))
        if self.uncertainty_image is not None:
            if self.sim.show_uncertainty_overlay:
                self.uncertainty_image.set_data(self.sim._shared_uncertainty_rgba())
            self.uncertainty_image.set_visible(bool(self.sim.show_uncertainty_overlay))
        if self.partition_image is not None:
            self.partition_image.set_data(self.sim.partition_rgba)
            self.partition_image.set_visible(visible)
        if self.partition_generator_scatter is not None:
            if len(self.sim.partition_generators_xy) > 0:
                self.partition_generator_scatter.set_offsets(self.sim.partition_generators_xy)
                self.partition_generator_scatter.set_sizes([55] * len(self.sim.partition_generators_xy))
                self.partition_generator_scatter.set_color(self.sim.partition_generator_colors)
            else:
                self.partition_generator_scatter.set_offsets([[0.0, 0.0]])
                self.partition_generator_scatter.set_sizes([0.0])
            self.partition_generator_scatter.set_visible(visible)
        if self.partition_centroid_scatter is not None:
            if len(self.sim.partition_centroids_xy) > 0:
                self.partition_centroid_scatter.set_offsets(self.sim.partition_centroids_xy)
                self.partition_centroid_scatter.set_sizes([38] * len(self.sim.partition_centroids_xy))
                self.partition_centroid_scatter.set_color(self.sim.partition_generator_colors)
            else:
                self.partition_centroid_scatter.set_offsets([[0.0, 0.0]])
                self.partition_centroid_scatter.set_sizes([0.0])
            self.partition_centroid_scatter.set_visible(visible)

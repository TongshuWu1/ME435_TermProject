import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.widgets import Button, RadioButtons, TextBox

from .config import DRONE_NAMES, WORLD_HEIGHT_METERS, WORLD_WIDTH_METERS

EDIT_MODE_LABELS = ['Add Waypoint', 'Set Start']
MISSION_MODE_LABELS = ['Manual Click', 'Auto Explore']


class SimulatorUI:
    """Builds and manages the Matplotlib canvas / widgets for the simulator."""

    def __init__(self, sim):
        self.sim = sim
        self.fig = None
        self.ax = None
        self.status_text = None
        self.toggle_button = None
        self.reset_button = None
        self.apply_seed_button = None
        self.clear_button = None
        self.partition_button = None
        self.seed_box = None
        self.robot_selector = None
        self.mode_selector = None
        self.mission_selector = None
        self.shared_map_ax = None
        self.shared_map_image = None
        self.shared_robot_scatter = None
        self.partition_image = None
        self.partition_generator_scatter = None
        self.partition_centroid_scatter = None

    def build(self):
        self.fig = plt.figure(figsize=(13.6, 8.9))
        self.ax = self.fig.add_axes([0.05, 0.08, 0.64, 0.86])
        self.ax.set_xlim(-0.35, WORLD_WIDTH_METERS + 0.35)
        self.ax.set_ylim(-0.35, WORLD_HEIGHT_METERS + 0.35)
        self.ax.set_aspect('equal')
        self.ax.set_title('Multi-Robot Search / Mapping', pad=10)
        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Y (meters)')
        self.ax.grid(True, alpha=0.14)

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

        self.status_text = self.fig.text(
            0.76,
            0.92,
            '',
            fontsize=8.8,
            verticalalignment='top',
            family='monospace',
        )

        mission_ax = self.fig.add_axes([0.77, 0.77, 0.18, 0.12])
        self.mission_selector = RadioButtons(mission_ax, MISSION_MODE_LABELS, active=0 if self.sim.mission_mode == 'manual_click' else 1)
        self.mission_selector.on_clicked(self.sim.on_select_mission_mode)
        mission_ax.set_title('Mission Mode', fontsize=9, pad=2)
        mission_ax.set_facecolor('#f5f5f5')

        robot_ax = self.fig.add_axes([0.77, 0.63, 0.18, 0.11])
        self.robot_selector = RadioButtons(robot_ax, DRONE_NAMES, active=0)
        self.robot_selector.on_clicked(self.sim.on_select_robot)
        robot_ax.set_title('Robot', fontsize=9, pad=2)
        robot_ax.set_facecolor('#f5f5f5')

        mode_ax = self.fig.add_axes([0.77, 0.50, 0.18, 0.10])
        self.mode_selector = RadioButtons(mode_ax, EDIT_MODE_LABELS, active=0)
        self.mode_selector.on_clicked(self.sim.on_select_edit_mode)
        mode_ax.set_title('Click Action', fontsize=9, pad=2)
        mode_ax.set_facecolor('#f5f5f5')

        toggle_ax = self.fig.add_axes([0.77, 0.43, 0.085, 0.052])
        self.toggle_button = Button(toggle_ax, '')
        self.toggle_button.on_clicked(self.sim.toggle_auto_mode)

        self.partition_button = Button(self.fig.add_axes([0.865, 0.43, 0.085, 0.052]), '')
        self.partition_button.on_clicked(self.sim.toggle_partition_overlay)

        clear_ax = self.fig.add_axes([0.77, 0.36, 0.18, 0.052])
        self.clear_button = Button(clear_ax, 'Clear Path/Goal')
        self.clear_button.on_clicked(self.sim.clear_selected_path)

        self.shared_map_ax = self.fig.add_axes([0.77, 0.16, 0.18, 0.16])
        self.shared_map_ax.set_title('Shared Map', fontsize=9, pad=3)
        self.shared_map_ax.set_xlim(0.0, WORLD_WIDTH_METERS)
        self.shared_map_ax.set_ylim(0.0, WORLD_HEIGHT_METERS)
        self.shared_map_ax.set_aspect('equal')
        self.shared_map_ax.set_xticks([])
        self.shared_map_ax.set_yticks([])
        for spine in self.shared_map_ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_edgecolor('#444444')
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
        self.shared_robot_scatter = self.shared_map_ax.scatter([], [], s=28, c=[], edgecolors='black', linewidths=0.5)

        seed_box_ax = self.fig.add_axes([0.77, 0.08, 0.11, 0.05])
        self.seed_box = TextBox(seed_box_ax, 'Seed ', initial=str(self.sim.current_seed))

        apply_seed_ax = self.fig.add_axes([0.89, 0.08, 0.06, 0.052])
        self.apply_seed_button = Button(apply_seed_ax, 'Apply')
        self.apply_seed_button.on_clicked(self.sim.apply_seed_from_box)

        reset_ax = self.fig.add_axes([0.77, 0.02, 0.18, 0.052])
        self.reset_button = Button(reset_ax, 'Reset')
        self.reset_button.on_clicked(self.sim.reset_simulation)

        self.fig.canvas.mpl_connect('button_press_event', self.sim.on_map_click)

        self.refresh_all()
        return self.fig, self.ax

    def refresh_all(self):
        self.refresh_toggle_button()
        self.refresh_partition_button()
        self.refresh_status_text()
        self.refresh_shared_map()
        self.refresh_partition_overlay()

    def refresh_toggle_button(self):
        if self.toggle_button is None:
            return
        self.toggle_button.label.set_text('Pause' if self.sim.auto_mode else 'Start')

    def refresh_partition_button(self):
        if self.partition_button is None:
            return
        self.partition_button.label.set_text('Hide Partition' if self.sim.show_partition_overlay else 'Show Partition')

    def refresh_status_text(self):
        if self.status_text is None:
            return
        drone_name = 'None'
        point_count = 0
        extra = ''
        if getattr(self.sim, 'drones', None):
            drone = self.sim.drones[self.sim.selected_drone_index]
            drone_name = drone['name']
            point_count = len(drone['path'])
            if self.sim.mission_mode == 'auto_explore':
                extra = f"\nGoal:  {'set' if drone.get('auto_goal_xy') is not None else 'none'}"
        sim_state = 'Running' if self.sim.auto_mode else 'Paused'
        mode_name = 'Add Waypoint' if getattr(self.sim, 'edit_mode', 'add_waypoint') == 'add_waypoint' else 'Set Start'
        mission_name = 'Manual Click' if self.sim.mission_mode == 'manual_click' else 'Auto Explore'
        text = (
            f'State: {sim_state}\n'
            f'Mission:{mission_name}\n'
            f'Robot: {drone_name}\n'
            f'Click: {mode_name}\n'
            f'Path:  {point_count}\n'
            f'Seed:  {self.sim.current_seed}'
            f'{extra}'
        )
        self.status_text.set_text(text)

    def sync_seed_box(self):
        if self.seed_box is not None:
            self.seed_box.set_val(str(self.sim.current_seed))

    def refresh_shared_map(self):
        if self.shared_map_image is not None:
            self.shared_map_image.set_data(self.sim.shared_known_grid)
        if self.shared_robot_scatter is not None and getattr(self.sim, 'drones', None):
            offsets = [[d['robot'].x, d['robot'].y] for d in self.sim.drones]
            colors = [d['color'] for d in self.sim.drones]
            self.shared_robot_scatter.set_offsets(offsets)
            self.shared_robot_scatter.set_facecolors(colors)

    def refresh_partition_overlay(self):
        visible = bool(self.sim.show_partition_overlay)
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

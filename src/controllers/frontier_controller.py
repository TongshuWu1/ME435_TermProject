from dataclasses import dataclass

from config import OWN_PATH_AVOID_GAIN, OWN_PATH_AVOID_RADIUS, TEAMMATE_PATH_AVOID_GAIN, TEAMMATE_PATH_AVOID_RADIUS
from ..auto_explore import choose_frontier_goal_for_robot


@dataclass
class FrontierController:
    fallback_global: bool
    top_k_candidates: int
    info_gain: float
    partition_penalty: float
    teammate_radius: float
    teammate_penalty: float
    progress_weight: float
    min_goal_distance: float
    centroid_weight: float
    density_value_weight: float

    def choose_goal(
        self,
        *,
        robot_index,
        frontier_components,
        partition_labels,
        grid_to_world_fn,
        robot_xy,
        known_grid,
        planner,
        occupied_value,
        unknown_value,
        density_map,
        centroid_xy,
        teammate_positions,
        teammate_goal_positions,
        own_recent_path=None,
        teammate_path_histories=None,
        partition_penalty_scale=1.0,
    ):
        return choose_frontier_goal_for_robot(
            robot_index,
            frontier_components,
            partition_labels,
            grid_to_world_fn,
            robot_xy,
            known_grid,
            planner,
            occupied_value,
            unknown_value,
            fallback_global=self.fallback_global,
            teammate_positions=teammate_positions,
            teammate_goal_positions=teammate_goal_positions,
            top_k_candidates=self.top_k_candidates,
            info_gain=self.info_gain,
            partition_penalty=self.partition_penalty,
            teammate_radius=self.teammate_radius,
            teammate_penalty=self.teammate_penalty,
            progress_weight=self.progress_weight,
            min_goal_distance=self.min_goal_distance,
            density_map=density_map,
            centroid_xy=centroid_xy,
            centroid_weight=self.centroid_weight,
            density_value_weight=self.density_value_weight,
            own_recent_path=own_recent_path,
            teammate_path_histories=teammate_path_histories,
            partition_penalty_scale=partition_penalty_scale,
            own_path_radius=OWN_PATH_AVOID_RADIUS,
            own_path_gain=OWN_PATH_AVOID_GAIN,
            teammate_path_radius=TEAMMATE_PATH_AVOID_RADIUS,
            teammate_path_gain=TEAMMATE_PATH_AVOID_GAIN,
        )

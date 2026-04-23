import math


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
    return {
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

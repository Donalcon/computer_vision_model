# Croke Park. Stadium should be included in initial config and we can pick custom dst_points as needed.
import mlflow
import numpy as np


def get_dst_points():
    dst = {
        '0A': [72.5, 39],
        '0B': [72.5, 49],
        '1A': [0, 0],
        '1B': [0, 34.5],
        '1C': [0, 37],
        '1D': [0, 51],
        '1E': [0, 53.5],
        '1F': [0, 88],
        '1G': [4.5, 37],
        '1H': [4.5, 51],
        '1I': [13, 0],
        '1J': [13, 34.5],
        '1K': [13, 53.5],
        '1L': [13, 88],
        '1M': [21, 0],
        '1N': [21, 34.5],
        '1O': [21, 53.5],
        '1P': [21, 88],
        '1Q': [45, 0],
        '1R': [45, 88],
        '1S': [65, 0],
        '1T': [65, 88],
        '2A': [145, 0],
        '2B': [145, 34.5],
        '2C': [145, 37],
        '2D': [145, 51],
        '2E': [145, 53.5],
        '2F': [145, 88],
        '2G': [140.5, 37],
        '2H': [140.5, 51],
        '2I': [132, 0, ],
        '2J': [132, 34.5],
        '2K': [132, 53.5],
        '2L': [132, 88],
        '2M': [124, 0],
        '2N': [124, 34.5],
        '2O': [124, 53.5],
        '2P': [124, 88],
        '2Q': [100, 0],
        '2R': [100, 88],
        '2S': [80, 0],
        '2T': [80, 88],
        '1GPA': [0, 40.75],
        '1GPB': [0, 47.25],
        '2GPA': [145, 40.75],
        '2GPB': [145, 47.25],
    }
    return dst


# Initialize counters
insufficient_points_count = 0
minimum_points_count = 0
additional_points_count = 0
collinear_points_count = 0
pitch_boundary_fail_count = 0

def collinear(p0, p1, p2, epsilon=0.001):
    x1, y1 = p1[0] - p0[0], p1[1] - p0[1]
    x2, y2 = p2[0] - p0[0], p2[1] - p0[1]
    return abs(x1 * y2 - x2 * y1) < epsilon


def check_num_points(src_points, dst_points):
    global insufficient_points_count, minimum_points_count, additional_points_count, collinear_points_count

    num_points = len(src_points)
    test_colinear = False
    if num_points == 4:
        if (collinear(dst_points[0], dst_points[1], dst_points[2]) or
                collinear(dst_points[0], dst_points[1], dst_points[3]) or
                collinear(dst_points[1], dst_points[2], dst_points[3])):
            test_colinear = True
            collinear_points_count += 1  # Increment collinearity counter
            return "collinear"

    if num_points < 4:
        print("Bad homography: Insufficient points")
        insufficient_points_count += 1
        return "insufficient"
    elif num_points == 4:
        print("Good homography: Minimum points")
        minimum_points_count += 1
        return "minimum"
    elif num_points > 4:
        print("Great homography: Additional points for robustness")
        additional_points_count += 1
        return "additional"

    return None


def verify_distance_between_players(transformed_coords):
    global distance_verification_fail_count
    for i, coord1 in enumerate(transformed_coords):
        for j, coord2 in enumerate(transformed_coords[i + 1:]):
            distance = np.linalg.norm(np.array(coord1) - np.array(coord2))
            if distance > 150:
                print("Bad homography: Players computed to be more than 150m apart.")
                distance_verification_fail_count += 1
                return False
    return True


def verify_players_within_pitch(transformed_coords):
    global pitch_boundary_fail_count
    for coord in transformed_coords:
        x, y = coord
        if x < 0 or x > 155 or y < 0 or y > 98:
            print("Bad homography: Player outside pitch parameters.")
            pitch_boundary_fail_count += 1
            return False
    return True


def log_fail_counts_to_mlflow():
    mlflow.log_metric("H - Distance Verification Failed", distance_verification_fail_count)
    mlflow.log_metric("H - Pitch Boundary Verification Failed", pitch_boundary_fail_count)
    mlflow.log_metric("H - Insufficient Points", insufficient_points_count)
    mlflow.log_metric("H - Minimum Points", minimum_points_count)
    mlflow.log_metric("H - Additional Points", additional_points_count)
    mlflow.log_metric("H - Collinear Points", collinear_points_count)

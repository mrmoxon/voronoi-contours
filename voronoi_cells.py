# Reloading the necessary dependencies and redefining the functions

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Point, LineString

# Redefining the previously defined functions
def generate_random_points(n, lower_bound, upper_bound):
    x_coords = np.random.uniform(lower_bound, upper_bound, n)
    y_coords = np.random.uniform(lower_bound, upper_bound, n)
    return np.column_stack((x_coords, y_coords))

def bounded_voronoi(points, lower_bound, upper_bound):
    bounding_points = [
        [lower_bound - 1, lower_bound - 1],
        [upper_bound + 1, lower_bound - 1],
        [upper_bound + 1, upper_bound + 1],
        [lower_bound - 1, upper_bound + 1]
    ]
    all_points = np.vstack([points, bounding_points])
    vor = Voronoi(all_points)
    return vor

def compute_closest_centroid(midpoint, points):
    distances = np.linalg.norm(points - midpoint, axis=1)
    return np.argmin(distances)

def calculate_intersection_angles(centroid, intersection):
    if intersection.geom_type == 'MultiLineString' or intersection.geom_type == 'MultiPoint':
        coords = [list(line.coords) for line in intersection]
        coords = [item for sublist in coords for item in sublist]
    else:
        coords = list(intersection.coords)
    
    angles = []
    for coord in coords:
        angle = np.arctan2(coord[1] - centroid[1], coord[0] - centroid[0]) % (2 * np.pi)
        angles.append(angle)
    
    return sorted(angles)

def extract_and_compute_arcs_alternate(vor, points, d, ring_counts):
    centroids_data = []

    for idx, point in enumerate(points):
        region = vor.regions[vor.point_region[vor.points.tolist().index(point.tolist())]]
        if not -1 in region:
            polygon = [vor.vertices[i] for i in region]
            cell = LineString(polygon + [polygon[0]])
            
            ring_details = []
            for i in range(1, ring_counts[idx] + 1):
                radius = i * d
                circle = Point(point).buffer(radius).boundary
                
                intersection = circle.intersection(cell)
                if intersection.geom_type in ["MultiLineString", "MultiPoint"]:
                    coords = [list(geom.coords) for geom in intersection.geoms]
                    coords = [item for sublist in coords for item in sublist]
                else:
                    coords = list(intersection.coords)
                
                angles = []
                for coord in coords:
                    angle = np.arctan2(coord[1] - point[1], coord[0] - point[0]) % (2 * np.pi)
                    angles.append(angle)
                
                arcs = []
                if len(angles) == 0:
                    arcs.append({
                        'start_angle': 0,
                        'end_angle': 2*np.pi,
                        'orientation': compute_orientation(point, cell, np.pi, radius)
                    })
                else:
                    angles_loop = angles + [angles[0] + 2*np.pi]
                    first_arc_orientation = compute_orientation(point, cell, angles[0] - 0.01, radius)
                    for j in range(0, len(angles)):
                        start_angle = angles_loop[j]
                        end_angle = angles_loop[j+1]
                        if j == 0:
                            orientation = first_arc_orientation
                        else:
                            orientation = 1 - orientation  # Alternate the orientation
                        arcs.append({
                            'start_angle': start_angle,
                            'end_angle': end_angle,
                            'orientation': orientation
                        })
                
                ring_details.append({
                    f"ring_{i}": {
                        "arcs": arcs
                    }
                })
            
            centroids_data.append({
                'centroid_number': idx + 1,
                'ring_details': ring_details
            })

    return centroids_data

def compute_orientation(centroid, cell, angle, radius):
    """
    Determine the orientation of the arc (inside or outside the Voronoi cell).
    If the ray from the centroid in the direction of the angle intersects with the Voronoi cell, the arc is outside.
    Otherwise, it's inside.
    """
    # Compute a point along the direction of the angle at a distance = radius
    end_x = centroid[0] + radius * np.cos(angle)
    end_y = centroid[1] + radius * np.sin(angle)
    
    # Create a line from the centroid to the computed point
    line = LineString([centroid, (end_x, end_y)])
    
    # If the line intersects with the Voronoi cell, the arc is outside
    if line.intersects(cell):
        return 0  # Outside
    else:
        return 1  # Inside

def plot_inner_arcs_gradient_no_voronoi_modified(vor, points, d, ring_counts, arc_data):
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Slightly more orange gradient and excluding the two innermost colors
    colors = plt.cm.autumn(np.linspace(0, 1, max(ring_counts) - 2))
    
    for idx, point in enumerate(points):
        for i, details in enumerate(arc_data[idx]['ring_details']):
            if i < 2:  # Skip the two innermost rings
                continue
                
            radius = (i + 1) * d
            for arc in details[f'ring_{i+1}']['arcs']:
                start_angle = arc['start_angle']
                end_angle = arc['end_angle']
                arc_midpoint_angle = (start_angle + end_angle) / 2
                mid_x = point[0] + radius * np.cos(arc_midpoint_angle)
                mid_y = point[1] + radius * np.sin(arc_midpoint_angle)
                
                # Determine closest centroid
                closest_idx = compute_closest_centroid([mid_x, mid_y], points)
                if closest_idx == idx:
                    color = colors[i - 2]  # Offset by 2 to account for the skipped rings
                    theta = np.linspace(start_angle, end_angle, 100)
                    x = point[0] + radius * np.cos(theta)
                    y = point[1] + radius * np.sin(theta)
                    ax.plot(x, y, color=color, linewidth=2)

# Adjusting parameters for closer spacing and more rings
d = 0.15
ring_count_fixed = 10

# New list for the number of centroids
num_centroids_list_new = [10, 15, 20, 25]

# Reinitializing the missing parameters
lower_bound = 0
upper_bound = 10

# Generating the visualizations again with the new parameters
for num_centroids in num_centroids_list_new:
    new_points = generate_random_points(num_centroids, lower_bound, upper_bound)
    vor_new = bounded_voronoi(new_points, lower_bound, upper_bound)
    
    # Using a fixed ring count for all centroids
    local_ring_counts = [ring_count_fixed] * len(new_points)
    
    arc_data_new = extract_and_compute_arcs_alternate(vor_new, new_points, d, local_ring_counts)
    plot_inner_arcs_gradient_no_voronoi_modified(vor_new, new_points, d, local_ring_counts, arc_data_new)

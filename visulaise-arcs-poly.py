import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
import math
from matplotlib.patches import Arc
from matplotlib.colors import ListedColormap

# Points that form two finite cells and four infinite cells around them
points = np.array([[2, 2], [8, 8], [2, 8], [8, 2], [5, 5], [5, 1]])

# Generating the Voronoi diagram
vor_symmetrical = Voronoi(points)

# Function to get finite centroids and polygons from the Voronoi diagram
def get_finite_centroids_and_polygons(vor):
    finite_centroids = []
    finite_polygons = []
    for region_index in vor.point_region:
        region = vor.regions[region_index]
        if -1 not in region and len(region) > 0:
            polygon = [vor.vertices[i] for i in region]
            centroid = np.mean(polygon, axis=0)
            finite_centroids.append(centroid)
            finite_polygons.append(polygon)
    return finite_centroids, finite_polygons

# Getting finite centroids and polygons for the specific Voronoi diagram
finite_centroids_symmetrical, finite_polygons_symmetrical = get_finite_centroids_and_polygons(vor_symmetrical)

# Function to draw arcs with colors for a given finite centroid
def draw_all_rings_with_arcs_for_finite_centroid(point, polygon, ax, num_rings, outer_radius):
    colors = ListedColormap(plt.cm.tab20.colors)
    for ring_number in range(num_rings + 1):
        automatic_arcs_for_specific_ring = find_automatic_arcs_for_specific_ring(
            point, polygon, ring_number, outer_radius, num_rings)
        for i, arc in enumerate(automatic_arcs_for_specific_ring):
            radius, start_angle, end_angle, midpoint_angle = arc
            ax.add_patch(Arc(point, 2 * radius, 2 * radius, angle=0,
                             theta1=start_angle, theta2=end_angle, color=colors(i)))

# Function to find arcs for a specific ring with automatic identification of midpoints
def find_automatic_arcs_for_specific_ring(point, polygon, ring_number, outer_radius, num_rings):
    # Radius of the specific ring
    circle_radius = ring_number * (outer_radius / num_rings)
    # Identifying intersection points and sorting them by angle
    intersection_angles = sorted(set([
        math.degrees(math.atan2(intersection[1] - point[1], intersection[0] - point[0]))
        for j in range(len(polygon))
        for edge_segment in [(polygon[j], polygon[(j + 1) % len(polygon)])]
        for intersection in circle_line_intersection(point, circle_radius, edge_segment)
    ]))
    # Forming arcs with automatic identification of midpoints
    arcs_vector = []
    if intersection_angles:  # Intersecting ring
        if len(intersection_angles) == 2:  # Two intersection points
            start_angle, end_angle = intersection_angles
            # Adjusting angles to ensure continuity
            if start_angle > end_angle:
                start_angle, end_angle = end_angle, start_angle
            # Arc 1 with automatic midpoint
            midpoint_angle_1 = (start_angle + end_angle) / 2
            arcs_vector.append((circle_radius, start_angle, end_angle, midpoint_angle_1))
            # Arc 2 with automatic midpoint
            midpoint_angle_2 = (end_angle + start_angle + 360) / 2
            arcs_vector.append((circle_radius, end_angle, start_angle + 360, midpoint_angle_2))
        else:  # More complex case
            # TODO: Handle more complex cases with more than two intersection points
            pass
    else:  # Complete circle without intersections
        arcs_vector.append((circle_radius, 0, 360, 180))
    return arcs_vector

# Function to find circle-line intersection points
def circle_line_intersection(circle_center, radius, line_segment):
    x1, y1 = line_segment[0]
    x2, y2 = line_segment[1]
    cx, cy = circle_center
    dx = x2 - x1
    dy = y2 - y1
    dr = np.sqrt(dx**2 + dy**2)
    D = x1 * y2 - x2 * y1
    discriminant = radius**2 * dr**2 - D**2
    if discriminant < 0:
        return []
    else:
        intersections = [
            ((D * dy + sgn(dy) * dx * np.sqrt(discriminant)) / (dr**2),
             (-D * dx + abs(dy) * np.sqrt(discriminant)) / (dr**2))
            for sgn in [lambda x: 1 if x >= 0 else -1]
        ]
        return [p for p in intersections if is_point_on_segment(p, line_segment)]

# Function to check if a point is on a line segment
def is_point_on_segment(point, segment):
    return min(segment[0][0], segment[1][0]) <= point[0] <= max(segment[0][0], segment[1][0]) and \
           min(segment[0][1], segment[1][1]) <= point[1] <= max(segment[0][1], segment[1][1])

# Points that form four finite cells and two infinite cells around them
points_with_four_finite_cells = np.array([[2, 2], [8, 8], [2, 8], [8, 2], [3, 3], [7, 7]])

# Generating the Voronoi diagram
vor_with_four_finite_cells = Voronoi(points_with_four_finite_cells)

# Getting finite centroids and polygons for the new Voronoi diagram
finite_centroids_four_finite, finite_polygons_four_finite = get_finite_centroids_and_polygons(vor_with_four_finite_cells)

# Plotting the specific Voronoi diagram with six concentric rings for the selected finite centroids (first two)
fig, ax = plt.subplots(figsize=(8, 8))
voronoi_plot_2d(vor_with_four_finite_cells, ax=ax, show_vertices=False, line_colors='black', line_alpha=1)
for i in range(2):  # Using the first two finite centroids
    draw_all_rings_with_arcs_for_finite_centroid(finite_centroids_four_finite[i], finite_polygons_four_finite[i], ax, num_rings=6, outer_radius=4)
plt.xlim(-1, 11)
plt.ylim(-1, 11)
plt.title(f"Voronoi Diagram with Six Concentric Rings for Selected Finite Centroids")
plt.show()

# # Plotting the specific Voronoi diagram with six concentric rings for the selected finite centroids
# fig, ax = plt.subplots(figsize=(8, 8))
# voronoi_plot_2d(vor_symmetrical, ax=ax, show_vertices=False, line_colors='black', line_alpha=1)
# for i in range(2):
#     draw_all_rings_with_arcs_for_finite_centroid(finite_centroids_symmetrical[i], finite_polygons_symmetrical[i], ax, num_rings=6, outer_radius=4)
# plt.xlim(-1, 11)
# plt.ylim(-1, 11)
# plt.title(f"Voronoi Diagram with All Arcs for Selected Finite Centroids")
# plt.show()

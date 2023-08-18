import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
import math
from matplotlib.patches import Arc
import cmath

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


# Function to find the intersection points between a circle and a line segment
def circle_line_intersection(circle_center, circle_radius, line_segment):
    x1, y1 = line_segment[0]
    x2, y2 = line_segment[1]
    cx, cy = circle_center
    dx, dy = x2 - x1, y2 - y1
    A = dx**2 + dy**2
    B = 2 * (dx * (x1 - cx) + dy * (y1 - cy))
    C = (x1 - cx)**2 + (y1 - cy)**2 - circle_radius**2
    discriminant = B**2 - 4 * A * C
    if discriminant < 0:
        return []  # No intersection
    t1 = (-B + cmath.sqrt(discriminant)) / (2 * A)
    t2 = (-B - cmath.sqrt(discriminant)) / (2 * A)
    intersections = []
    for t in [t1, t2]:
        if 0 <= t <= 1:
            ix = x1 + t * dx
            iy = y1 + t * dy
            intersections.append((ix, iy))
    return intersections

# Function to draw arcs with different colors for a given centroid
def draw_arcs_with_colors_for_finite_centroid(point, arcs_vector, ax):
    colors = plt.cm.jet(np.linspace(0, 1, len(arcs_vector)))  # Color map
    for i, (radius, start_angle, end_angle, midpoint_angle) in enumerate(arcs_vector):
        arc = Arc(point, 2 * radius, 2 * radius, theta1=start_angle, theta2=end_angle, color=colors[i])
        ax.add_patch(arc)

# Function to find arcs for any ring (including full circles) for a given centroid
def find_all_arcs_for_ring(point, polygon, ring_radius):
    intersection_angles = sorted(set([
        math.degrees(math.atan2(intersection[1] - point[1], intersection[0] - point[0]))
        for j in range(len(polygon))
        for edge_segment in [(polygon[j], polygon[(j + 1) % len(polygon)])]
        for intersection in circle_line_intersection(point, ring_radius, edge_segment)
    ]))
    arcs_vector = []
    if len(intersection_angles) == 0:
        arcs_vector.append((ring_radius, 0, 360, 180))
    elif len(intersection_angles) == 2:
        start_angle, end_angle = intersection_angles
        if start_angle > end_angle:
            start_angle, end_angle = end_angle, start_angle
        midpoint_angle_1 = (start_angle + end_angle) / 2
        arcs_vector.append((ring_radius, start_angle, end_angle, midpoint_angle_1))
        midpoint_angle_2 = (end_angle + start_angle + 360) / 2
        arcs_vector.append((ring_radius, end_angle, start_angle + 360, midpoint_angle_2))
    else:
        for i in range(len(intersection_angles)):
            start_angle = intersection_angles[i]
            end_angle = intersection_angles[(i + 1) % len(intersection_angles)]
            if end_angle < start_angle:
                end_angle += 360
            midpoint_angle = (start_angle + end_angle) / 2
            arcs_vector.append((ring_radius, start_angle, end_angle, midpoint_angle))
    return arcs_vector

# Function to draw all rings (arcs and full circles) for a given centroid
def draw_all_rings_with_arcs_for_finite_centroid(point, polygon, ax, num_rings, outer_radius):
    for ring_number in range(num_rings + 1):
        circle_radius = ring_number * (outer_radius / num_rings)
        arcs_vector = find_all_arcs_for_ring(point, polygon, circle_radius)
        draw_arcs_with_colors_for_finite_centroid(point, arcs_vector, ax)

# Points that form two identically sized finite cells (triangular) and four infinite cells around it
symmetrical_points = np.array([[2, 2], [8, 8], [2, 8], [8, 2], [5, 5], [5, 1]])
vor_symmetrical = Voronoi(symmetrical_points)
finite_centroids_symmetrical, finite_polygons_symmetrical = get_finite_centroids_and_polygons(vor_symmetrical)
selected_finite_centroid_point_symmetrical, selected_finite_centroid_polygon_symmetrical = finite_centroids_symmetrical[0], finite_polygons_symmetrical[0]

# Plotting
fig, ax = plt.subplots(figsize=(8, 8))
voronoi_plot_2d(vor_symmetrical, ax=ax, show_vertices=False, line_colors='black', line_alpha=1)
draw_all_rings_with_arcs_for_finite_centroid(selected_finite_centroid_point_symmetrical, selected_finite_centroid_polygon_symmetrical, ax, num_rings=6, outer_radius=4)
plt.xlim(-1, 11)
plt.ylim(-1, 11)
plt.title(f"Voronoi Diagram with All Arcs for Selected Finite Centroid")
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
import math
from matplotlib.patches import Arc
import cmath

# Function to find the intersection points between a circle and a line segment
def circle_line_intersection(circle_center, circle_radius, line_segment):
    x1, y1 = line_segment[0]
    x2, y2 = line_segment[1]
    cx, cy = circle_center
    dx, dy = x2 - x1, y2 - y1
    A = dx**2 + dy**2
    B = 2 * (dx * (x1 - cx) + dy * (y1 - cy))
    C = (x1 - cx)**2 + (y1 - cy)**2 - circle_radius**2
    discriminant = B**2 - 4 * A * C
    if discriminant < 0:
        return []  # No intersection
    t1 = (-B + cmath.sqrt(discriminant)) / (2 * A)
    t2 = (-B - cmath.sqrt(discriminant)) / (2 * A)
    intersections = []
    for t in [t1, t2]:
        if 0 <= t <= 1:
            ix = x1 + t * dx
            iy = y1 + t * dy
            intersections.append((ix, iy))
    return intersections

# Function to draw arcs with different colors for a given centroid
def draw_arcs_with_colors_for_finite_centroid(point, arcs_vector, ax):
    colors = plt.cm.jet(np.linspace(0, 1, len(arcs_vector)))  # Color map
    for i, (radius, start_angle, end_angle, midpoint_angle) in enumerate(arcs_vector):
        arc = Arc(point, 2 * radius, 2 * radius, theta1=start_angle, theta2=end_angle, color=colors[i])
        ax.add_patch(arc)

# Function to find arcs for any ring (including full circles) for a given centroid
def find_all_arcs_for_ring(point, polygon, ring_radius):
    intersection_angles = sorted(set([
        math.degrees(math.atan2(intersection[1] - point[1], intersection[0] - point[0]))
        for j in range(len(polygon))
        for edge_segment in [(polygon[j], polygon[(j + 1) % len(polygon)])]
        for intersection in circle_line_intersection(point, ring_radius, edge_segment)
    ]))
    arcs_vector = []
    if len(intersection_angles) == 0:
        arcs_vector.append((ring_radius, 0, 360, 180))
    elif len(intersection_angles) == 2:
        start_angle, end_angle = intersection_angles
        if start_angle > end_angle:
            start_angle, end_angle = end_angle, start_angle
        midpoint_angle_1 = (start_angle + end_angle) / 2
        arcs_vector.append((ring_radius, start_angle, end_angle, midpoint_angle_1))
        midpoint_angle_2 = (end_angle + start_angle + 360) / 2
        arcs_vector.append((ring_radius, end_angle, start_angle + 360, midpoint_angle_2))
    else:
        for i in range(len(intersection_angles)):
            start_angle = intersection_angles[i]
            end_angle = intersection_angles[(i + 1) % len(intersection_angles)]
            if end_angle < start_angle:
                end_angle += 360
            midpoint_angle = (start_angle + end_angle) / 2
            arcs_vector.append((ring_radius, start_angle, end_angle, midpoint_angle))
    return arcs_vector

# Function to draw all rings (arcs and full circles) for a given centroid
def draw_all_rings_with_arcs_for_finite_centroid(point, polygon, ax, num_rings, outer_radius):
    for ring_number in range(num_rings + 1):
        circle_radius = ring_number * (outer_radius / num_rings)
        arcs_vector = find_all_arcs_for_ring(point, polygon, circle_radius)
        draw_arcs_with_colors_for_finite_centroid(point, arcs_vector, ax)

# Points that form two identically sized finite cells (triangular) and four infinite cells around it
symmetrical_points = np.array([[2, 2], [8, 8], [2, 8], [8, 2], [5, 5], [5, 1]])
vor_symmetrical = Voronoi(symmetrical_points)
finite_centroids_symmetrical, finite_polygons_symmetrical = get_finite_centroids_and_polygons(vor_symmetrical)
selected_finite_centroid_point_symmetrical, selected_finite_centroid_polygon_symmetrical = finite_centroids_symmetrical[0], finite_polygons_symmetrical[0]

# Plotting
fig, ax = plt.subplots(figsize=(8, 8))
voronoi_plot_2d(vor_symmetrical, ax=ax, show_vertices=False, line_colors='black', line_alpha=1)
draw_all_rings_with_arcs_for_finite_centroid(selected_finite_centroid_point_symmetrical, selected_finite_centroid_polygon_symmetrical, ax, num_rings=6, outer_radius=4)
plt.xlim(-1, 11)
plt.ylim(-1, 11)
plt.title(f"Voronoi Diagram with All Arcs for Selected Finite Centroid")
plt.show()

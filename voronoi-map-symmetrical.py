import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, PathPatch
from matplotlib.path import Path

# Points that form two identically sized finite cells (triangular) and four infinite cells around it
symmetrical_points = np.array([[2, 2], [8, 8], [2, 8], [8, 2], [5, 5], [5, 1]])

# Generating the specific Voronoi diagram
vor_symmetrical = Voronoi(symmetrical_points)

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

# Function to draw concentric rings for a given point with specified outer radius and number of rings
def draw_concentric_rings_for_finite_centroid(point, polygon, ax, num_rings, outer_radius):
    path = Path(polygon)
    for ring_number in range(1, num_rings + 1):
        circle_radius = ring_number * (outer_radius / num_rings)
        circle = plt.Circle(point, circle_radius, color='blue', fill=False)
        circle.set_clip_path(PathPatch(path, transform=ax.transData))
        ax.add_artist(circle)

# Getting finite centroids and polygons for the specific symmetrical Voronoi diagram
finite_centroids_symmetrical, finite_polygons_symmetrical = get_finite_centroids_and_polygons(vor_symmetrical)

# Selecting the only available finite centroid for visualization
selected_finite_centroid_point_symmetrical, selected_finite_centroid_polygon_symmetrical = finite_centroids_symmetrical[0], finite_polygons_symmetrical[0]

# Plotting the specific Voronoi diagram with six concentric rings for the selected finite centroid
fig, ax = plt.subplots(figsize=(8, 8))
voronoi_plot_2d(vor_symmetrical, ax=ax, show_vertices=False, line_colors='black', line_alpha=1)
draw_concentric_rings_for_finite_centroid(selected_finite_centroid_point_symmetrical, selected_finite_centroid_polygon_symmetrical, ax, num_rings=6, outer_radius=4)
plt.xlim(-1, 11)
plt.ylim(-1, 11)
plt.title(f"Voronoi Diagram with Six Concentric Rings for Selected Finite Centroid (Symmetrical Map)")
plt.show()

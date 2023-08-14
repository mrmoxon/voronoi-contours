# Importing necessary libraries
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
import numpy as np
import math

# Function to draw concentric rings for a given point
def draw_concentric_rings(point, outer_radius, num_rings, ax):
    for ring_number in range(1, num_rings + 1):
        circle_radius = ring_number * (outer_radius / num_rings)
        circle = plt.Circle(point, circle_radius, color='blue', fill=False)
        ax.add_artist(circle)

# Generating six seed points
# generate a random seed number each time the script is run
np.random.seed(1234)
points = np.random.rand(6, 2) * 10

# Generating Voronoi diagram
vor = Voronoi(points)

# Plotting Voronoi diagram with concentric rings
fig, ax = plt.subplots(figsize=(8, 8))
voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='black', line_alpha=1)
for point in points:
    draw_concentric_rings(point, outer_radius=4, num_rings=5, ax=ax)
plt.xlim(-1, 11)
plt.ylim(-1, 11)
plt.title("Voronoi Diagram with Concentric Rings")
plt.show()

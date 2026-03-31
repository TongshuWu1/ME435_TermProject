# landmark.py

import matplotlib.patches as patches
import numpy as np

class Landmark:
    def __init__(self, x, y, shape="circle", color="yellow", size=1.1):
        """
        Parameters:
          x, y: Position in meters.
          shape: "circle", "square", or "triangle".
          color: "yellow" or "orange".
          size: Landmark size in meters.
        """
        self.x = x
        self.y = y
        self.shape = shape
        self.color_name = color  # Ensure this attribute is present
        if color == "yellow":
            self.color = '#FFCC00'
        else:
            self.color = '#FF8C00'
        self.size = size

    def draw(self, ax):
        """Draw the landmark on the given axes."""
        if self.shape == "circle":
            patch = patches.Circle((self.x, self.y), self.size/2,
                                   facecolor=self.color, edgecolor='black')
        elif self.shape == "square":
            patch = patches.Rectangle((self.x - self.size/2, self.y - self.size/2),
                                      self.size, self.size,
                                      facecolor=self.color, edgecolor='black')
        elif self.shape == "triangle":
            half = self.size / 2
            points = [(self.x, self.y + half),
                      (self.x - half, self.y - half),
                      (self.x + half, self.y - half)]
            patch = patches.Polygon(points, closed=True,
                                    facecolor=self.color, edgecolor='black')
        else:
            patch = None
        if patch is not None:
            ax.add_patch(patch)
        return patch

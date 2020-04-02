"""
Created on Wed Sep 25 20:52:45 2019
@author: kyleguan

"""

import numpy as np
import matplotlib.pyplot as plt

def draw_box(pyplot_axis, vertices, axes=[0, 1, 2], color='red'):
    """
    Draws a bounding 3D box in a pyplot axis.

    Parameters
    ----------
    pyplot_axis : Pyplot axis to draw in.
    vertices    : Array 8 box vertices containing x, y, z coordinates.
    axes        : Axes to use. Defaults to `[0, 1, 2]`, e.g. x, y and z axes.
    color       : Drawing color. Defaults to `black`.
    """
    vertices = vertices[axes, :]
    connections = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Lower plane parallel to Z=0 plane
        [4, 5], [5, 6], [6, 7], [7, 4],  # Upper plane parallel to Z=0 plane
        [0, 4], [1, 5], [2, 6], [3, 7]  # Connections between upper and lower planes
    ]
    for connection in connections:
        pyplot_axis.plot(*vertices[:, connection], c=color, lw=0.5)




def draw_point_cloud(cloud, ax, title, axes_str , axes=[0, 1, 2]):

        cloud = np.array(cloud) # Covert point cloud to numpy array
        no_points = np.shape(cloud)[0]
        point_size = 10**(3- int(np.log10(no_points))) # Adjust the point size based on the point cloud size
        if np.shape(cloud)[1] == 4: # If point cloud is XYZI format (e.g., I stands for intensity)
            ax.scatter(*np.transpose(cloud[:, axes]), s = point_size, c=cloud[:, 3], cmap='gray')
        elif np.shape(cloud)[1] == 3:   # If point cloud is XYZ format
            ax.scatter(*np.transpose(cloud[:, axes]), s = point_size, c='b', alpha = 0.7)
        ax.set_xlabel('{} axis'.format(axes_str[axes[0]]))
        ax.set_ylabel('{} axis'.format(axes_str[axes[1]]))
        # if len(axes) > 2: # 3-D plot
        #     ax.set_xlim3d(axes_limits[axes[0]])
        #     ax.set_ylim3d(axes_limits[axes[1]])
        #     ax.set_zlim3d(axes_limits[axes[2]])
        #     ax.set_zlabel('{} axis'.format(axes_str[axes[2]]))
        # else: # 2-D plot
        #     ax.set_xlim(*axes_limits[axes[0]])
        #     ax.set_ylim(*axes_limits[axes[1]])
#        # User specified limits
#        if xlim3d!=None:
#            ax.set_xlim3d(xlim3d)
#        if ylim3d!=None:
#            ax.set_ylim3d(ylim3d)
#        if zlim3d!=None:
#            ax.set_zlim3d(zlim3d)
        ax.set_title(title)

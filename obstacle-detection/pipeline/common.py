"""
Our implementation of obstacle detection pipeline steps
@authors:
"""
import numpy as np
import pandas as pd
from datetime import datetime

from scipy.spatial import ConvexHull
from scipy.ndimage.interpolation import rotate


def roi_filter_rounded(pcloud, verbose=True, **params):
    """ Region of Interest filter """

    a = (- params['max_x'] - params['max_x']) / 2
    b = (params['min_y'] - params['max_y']) / 2

    if verbose:
        print('Input pcloud size: {}'.format(len(pcloud)))
    pcloud['equation'] = (pcloud['x'] ** 2) / (a ** 2) + \
        (pcloud['y'] ** 2) / (b ** 2)

    pcloud['camera'] = ((pcloud['z'] > params['min_z']) & (pcloud['z'] < params['max_z']) &
                        (pcloud['x'] > params['min_x']) &
                        (pcloud['equation'] <= 1.0))

    pcloud = pcloud[pcloud['camera'] == True]

    if verbose:
        print('Output ROI pcloud size: {}'.format(len(pcloud)))
    return pcloud


def roi_filter(pcloud, verbose=True, **params):
    """
    Region Of Interest function, which filter required area
    that relative to LIDAR scanner (point (0, 0, 0) is a center)
    """
    if verbose:
        print('Input pcloud size: {}'.format(len(pcloud)))
    pcloud['camera'] = ((pcloud['x'] > params['min_x']) & (pcloud['x'] < params['max_x']) &
                        (pcloud['y'] > params['min_y']) & (pcloud['y'] < params['max_y']) &
                        (pcloud['z'] > params['min_z']) & (pcloud['z'] < params['max_z']))
    pcloud = pcloud[pcloud['camera'] == True]
    if verbose:
        print('Output ROI pcloud size: {}'.format(len(pcloud)))
    return pcloud


def obstacle_filter(pcloud, obstacle_lst, proc_labels=True, verbose=True):
    """
    Obstacle filtering function
    pcloud: pandas.DataFrame,
    Point cloud DataFrame that have columns=['x', 'y', 'z', 'seg_id']
    obstacle_lst: list,
    A list of segments id you want to be remain after filtering
    """
    # sanity check
    assert isinstance(pcloud, pd.DataFrame)

    origin_point_size = len(pcloud)

    if proc_labels:
        pcloud.seg_id = pcloud.seg_id.astype('uint32')
        pcloud.seg_id = pcloud.seg_id.apply(lambda x: x & 0xFFFF)
        pcloud = pcloud[pcloud['seg_id'].isin(list(obstacle_lst.keys()))]
    else:
        pcloud = pcloud[pcloud['seg_id'].isin(obstacle_lst)]
    if verbose:
        print('Filter required segments')
        print(
            'Point size before: {} and after filtering: {}'.format(
                origin_point_size,
                len(pcloud)))

    return pcloud


def outlier_filter(tcluster, verbose=True):
    """Outlier filter with 3-sigmas rule"""

    # tcluster['norm'] = np.sqrt(np.square(tcluster).sum(axis=1))
    start_time = datetime.now()
    try:
        _mean, _std = tcluster['norm'].mean(), tcluster['norm'].std()
        lower, higher = _mean - 3 * _std, _mean + 3 * _std
    except BaseException:
        tcluster['norm'] = np.sqrt(
            np.square(tcluster[['x', 'y', 'z']]).sum(axis=1))
        _mean, _std = tcluster['norm'].mean(), tcluster['norm'].std()
        lower, higher = _mean - 3 * _std, _mean + 3 * _std
    end_time = (datetime.now() - start_time).total_seconds()
    if verbose:
        print('Computing lower-higher bounds {}'.format(end_time))

    start_time = datetime.now()
    tcluster = tcluster[(tcluster['norm'] > lower) &
                        (tcluster['norm'] < higher)]
    end_time = (datetime.now() - start_time).total_seconds()
    if verbose:
        print('Applying  bounds {}'.format(end_time))
    return tcluster


def get_bounding_boxes(clusters):
    box_coord_list = []
    for i in range(len(clusters)):
        x_min, x_max, y_min, y_max, z_min, z_max = list(clusters.iloc[i])
        box = np.zeros([8, 3])
        box[0, :] = [x_min, y_min, z_min]
        box[1, :] = [x_max, y_min, z_min]
        box[2, :] = [x_max, y_max, z_min]
        box[3, :] = [x_min, y_max, z_min]
        box[4, :] = [x_min, y_min, z_max]
        box[5, :] = [x_max, y_min, z_max]
        box[6, :] = [x_max, y_max, z_max]
        box[7, :] = [x_min, y_max, z_max]
        box = np.transpose(box)
        box_coord_list.append(box)
    return box_coord_list


def get_OBB(cluster):
    """compute Oriented Bounding Boxes for cluster"""

    # sanity check
    assert isinstance(cluster, pd.DataFrame)

    # get min max values for Z axis
    z_min, z_max = cluster['z'].min(), cluster['z'].max()

    # get minimum bounding box on XoY surfuce
    xy_minimum_bb = minimum_bounding_box(cluster[['x', 'y']].values)

    # make array [z_min, z_min , z_min , z_min , z_max,  z_max,  z_max,  z_max]
    z_array = np.array([z_min] * 4 + [z_max] * 4)

    # double xy bbox z_array
    xy_minimum_bb = np.concatenate((xy_minimum_bb, xy_minimum_bb), axis=0)

    # concatenate xy with z values and get array of 8x3 shape
    obb = np.hstack((xy_minimum_bb, z_array.reshape(8, 1)))

    return obb


def minimum_bounding_box(points):
    """compute minimum bounding box in XoY"""

    pi2 = np.pi / 2.

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = np.zeros((len(hull_points) - 1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles - pi2),
        np.cos(angles + pi2),
        np.cos(angles)]).T
    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    #             closest leftmost
    rval[0] = np.dot([x2, y2], r)
    #             closest rightmost
    rval[1] = np.dot([x2, y1], r)
    #             farthest leftmost
    rval[2] = np.dot([x1, y2], r)
    #             farthest rightmost
    rval[3] = np.dot([x1, y1], r)

    return rval

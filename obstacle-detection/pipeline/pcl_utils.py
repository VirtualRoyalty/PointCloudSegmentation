#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Sep 25 20:52:45 2019
@author: kyleguan
"""

import numpy as np
import pcl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


def voxel_filter(cloud, leaf_sizes):
    """
    Input parameters:
    cloud: input point cloud to be filtered
    leaf_sizes: a list of leaf_size for X, Y, Z
    Output:
    cloud_voxel_filtered: voxel-filtered cloud
    """
    sor = cloud.make_voxel_grid_filter()
    size_x, size_y, size_z = leaf_sizes
    sor.set_leaf_size(size_x, size_y, size_z)
    cloud_voxel_filtered = sor.filter()

    return cloud_voxel_filtered


def roi_filter(cloud, x_roi, y_roi, z_roi):
    """
    Input Parameters:
        cloud: input point cloud
        x_roi: ROI range in X
        y_roi: ROI range in Y
        z_roi: ROI range in Z

    Output:
        ROI region filtered point cloud
    """
    clipper = cloud.make_cropbox()
    cloud_roi_filtered = pcl.PointCloud()
    xc_min, xc_max = x_roi
    yc_min, yc_max = y_roi
    zc_min, zc_max = z_roi
    clipper.set_MinMax(xc_min, yc_min, zc_min, 0, xc_max, yc_max, zc_max, 0)
    cloud_roi_filtered = clipper.filter()
    return cloud_roi_filtered


def plane_segmentation(cloud, dist_thold, max_iter):
    """
    Input parameters:
        cloud: Input cloud
        dist_thold: distance threshold
        max_iter: maximal number of iteration
    Output:
        indices: list of indices of the PCL points
                 that belongs to the plane
        coefficient: the coefficients of the plane-fitting
                     (e.g., [a, b, c, d] for ax + by +cz + d =0)
    """
    seg = cloud.make_segmenter_normals(ksearch=50)  # For simplicity,hard coded
    seg.set_optimize_coefficients(True)
    seg.set_model_type(pcl.SACMODEL_NORMAL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_distance_threshold(dist_thold)
    seg.set_max_iterations(max_iter)
    indices, coefficients = seg.segment()
    return indices, coefficients


def clustering(cloud, tol, min_size, max_size):
    """
    Input parameters:
        cloud: Input cloud
        tol: tolerance
        min_size: minimal number of points to form a cluster
        max_size: maximal number of points that a cluster allows
    Output:
        cluster_indices: a list of list. Each element list contains
                         the indices of the points that belongs to
                         the same cluster
    """
    tree = cloud.make_kdtree()
    ec = cloud.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(tol)
    ec.set_MinClusterSize(min_size)
    ec.set_MaxClusterSize(max_size)
    ec.set_SearchMethod(tree)
    cluster_indices = ec.Extract()
    return cluster_indices


def get_cluster_box_list(
    cluster_indices, cloud_obsts, radius_search=0.8, min_neighbors_in_radius=2
):
    """
    Input parameters:
        cluster_indices: a list of list. Each element list contains
                         the indices of the points that belongs to
                         the same cluster
        colud_obsts: PCL for the obstacles
    Output:
        cloud_cluster_list: a list for the PCL clusters each element
                            is a point cloud of a cluster
        box_coord_list: a list of corrdinates for bounding boxes
    """
    cloud_cluster_list = []
    box_coord_list = []
    box_min_max_list = np.zeros((len(cluster_indices), 8, 3))

    for j, indices in enumerate(cluster_indices):
        points = np.zeros((len(indices), 3), dtype=np.float32)
        for i, indice in enumerate(indices):
            points[i][0] = cloud_obsts[indice][0]
            points[i][1] = cloud_obsts[indice][1]
            points[i][2] = cloud_obsts[indice][2]
        cloud_cluster = pcl.PointCloud()
        cloud_cluster.from_array(points)

        # http://pointclouds.org/documentation/tutorials/remove_outliers.php

        # radius remove-outliers

        outrem = cloud_cluster.make_RadiusOutlierRemoval()
        outrem.set_radius_search(radius_search)
        outrem.set_MinNeighborsInRadius(min_neighbors_in_radius)
        cloud_filtered = outrem.filter()

        # condition remove-outliers
        # range_cond = cloud_cluster.make_ConditionAnd()

        # range_cond.add_Comparison2('z', pcl.CythonCompareOp_Type.GT, 0.0)
        # range_cond.add_Comparison2('z', pcl.CythonCompareOp_Type.LT, 0.8)

        # build the filter
        # condrem = cloud_cluster.make_ConditionalRemoval(range_cond)
        # condrem.set_KeepOrganized(True)

        # cloud_filtered = condrem.filter()

        # other filter
        # cloud_filtered = cloud_cluster.make_statistical_outlier_filter()
        # cloud_filtered.set_mean_k(50)
        # cloud_filtered.set_std_dev_mul_thresh(1.0)
        # cloud_filtered = cloud_cluster

        cloud_cluster_list.append(cloud_filtered)
        x_max, x_min = np.max(points[:, 0]), np.min(points[:, 0])
        y_max, y_min = np.max(points[:, 1]), np.min(points[:, 1])
        z_max, z_min = np.max(points[:, 2]), np.min(points[:, 2])
        box = np.zeros([8, 3])
        box[0, :] = [x_min, y_min, z_min]
        box[1, :] = [x_min, y_max, z_min]
        box[2, :] = [x_max, y_min, z_min]
        box[3, :] = [x_max, y_max, z_min]
        box[4, :] = [x_min, y_min, z_max]
        box[5, :] = [x_min, y_max, z_max]
        box[6, :] = [x_max, y_min, z_max]
        box[7, :] = [x_max, y_max, z_max]
        # box = np.transpose(box)
        box_coord_list.append(box)
        box_min_max_list[j] = box
    return box_min_max_list, box_coord_list


def box_center(box):
    """
    Calculate the centroid of a 3D bounding box
    Input: box, a 3-by-8 matrix, each coloum represents the xyz coordinate of a corner of the box
           (e.g.
           array([[42.62581635, 46.09998703, 46.09998703, 42.62581635, 42.62581635, 46.09998703, 46.09998703, 42.62581635],
                  [2.64766479,  2.64766479,  4.64661026,  4.64661026,  2.64766479, 2.64766479,  4.64661026,  4.64661026],
                  [0.10515476,  0.10515476,  0.10515476,  0.10515476,  1.98793995, 1.98793995,  1.98793995,  1.98793995]])
           )
    Output: the centroid of the box in 3D [x_cent, y_cent, z_cent]
    """
    x_min, x_max = min(box[0]), max(box[0])
    y_min, y_max = min(box[1]), max(box[1])
    z_min, z_max = min(box[2]), max(box[2])

    return ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0, (z_min + z_max) / 2.0)


def get_min_max_box(box):
    """
    Get x_min, x_max, y_min, y_max, z_min, z_max from boxes
    Input: box, a 3-by-8 matrix, each coloum represents the xyz coordinate of a corner of the box
           (e.g.
           array([[42.62581635, 46.09998703, 46.09998703, 42.62581635, 42.62581635, 46.09998703, 46.09998703, 42.62581635],
                  [2.64766479,  2.64766479,  4.64661026,  4.64661026,  2.64766479, 2.64766479,  4.64661026,  4.64661026],
                  [0.10515476,  0.10515476,  0.10515476,  0.10515476,  1.98793995, 1.98793995,  1.98793995,  1.98793995]])
           )
    Output: min, max of the box in 3D [x_min, x_max, y_min, y_max, z_min, z_max]
    """
    x_min, x_max = min(box[0]), max(box[0])
    y_min, y_max = min(box[1]), max(box[1])
    z_min, z_max = min(box[2]), max(box[2])

    return [x_min, x_max, y_min, y_max, z_min, z_max]

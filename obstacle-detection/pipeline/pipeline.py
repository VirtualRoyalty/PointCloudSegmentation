import numpy as np
import pandas as pd
import importlib as imp

from datetime import datetime
from sklearn.cluster import DBSCAN
from pyobb.obb import OBB
from pipeline import common

common = imp.reload(common)

common = imp.reload(common)


def pipeline(
    scan, label, obstacle_lst, verbose=False, OBBoxes=False, exec_time=False, **params
):
    """ ROI filtering """
    ##########################################################################
    start_time = datetime.now()
    pcloud = pd.DataFrame(
        np.concatenate((scan, label.reshape(len(label), 1)), axis=1),
        columns=["x", "y", "z", "seg_id"],
    )
    pcloud = common.roi_filter(
        pcloud,
        min_x=params["roi_x_min"],
        max_x=params["roi_x_max"],
        min_y=params["roi_y_min"],
        max_y=params["roi_y_max"],
        min_z=params["roi_z_min"],
        max_z=params["roi_z_max"],
        verbose=False,
    )
    roi_time = (datetime.now() - start_time).total_seconds()
    ##########################################################################
    """ Obstacles filtering """
    ##########################################################################
    start_time = datetime.now()
    pcloud = common.obstacle_filter(
        pcloud, obstacle_lst, proc_labels=True, verbose=False
    )
    obstacle_time = (datetime.now() - start_time).total_seconds()
    ##########################################################################

    if len(pcloud) > 200:

        # Getting voxel grid
        start_time = datetime.now()
        voxel_time = (datetime.now() - start_time).total_seconds()
        """ Сlustering obstacles """
        #######################################################################
        start_time = datetime.now()
        clusterer = DBSCAN(
            eps=params["eps"],
            min_samples=params["min_samples"],
            algorithm="auto",
            leaf_size=params["leaf_size"],
            n_jobs=-1,
        )
        clusterer.fit(pcloud[["x", "y", "z"]])
        pcloud["cluster_id"] = clusterer.labels_
        cluster_time = (datetime.now() - start_time).total_seconds()
        #######################################################################
        """ Getting bounding boxes coord """
        #######################################################################
        start_time = datetime.now()
        pcloud["norm"] = np.sqrt(np.square(pcloud[["x", "y", "z"]]).sum(axis=1))
        cluster_data = pd.DataFrame.from_dict(
            {"x": [], "y": [], "z": [], "cluster_id": []}
        )
        clusters = []
        for _id in sorted(pcloud["cluster_id"].unique()):
            if _id == -1 or not 50 < len(pcloud[pcloud["cluster_id"] == _id]) < 5000:
                continue
            tcluster = pcloud[pcloud["cluster_id"] == _id]
            tcluster = common.outlier_filter(tcluster, verbose=False)
            cluster_data = cluster_data.append(tcluster)
            if OBBoxes:
                obb = common.get_OBB(tcluster[["x", "y", "z"]])
                clusters.append(obb)
        if not OBBoxes:
            clusters = (
                cluster_data.groupby(["cluster_id"])
                .agg({"x": ["min", "max"], "y": ["min", "max"], "z": ["min", "max"]})
                .values
            )
        bb_time = (datetime.now() - start_time).total_seconds()
        #######################################################################
    else:
        clusters, cluster_data = np.empty((0, 0)), np.empty((0, 0))
        voxel_time, cluster_time, bb_time = 0, 0, 0

    if verbose:
        print("Execution time:")
        print("\n - ROI filtering: {:.5f}s".format(roi_time))
        print("\n - Filtering obstacles: {:.5f}s".format(obstacle_time))
        print("\n - Voxel grid: {:.5f}s".format(voxel_time))
        print("\n - Clustering: {:.5f}s".format(cluster_time))
        print("\n - Min-max cluster points: {:.5f}s \n".format(bb_time))

    if exec_time:
        return (
            clusters,
            cluster_data,
            {
                "roi_time": roi_time,
                "filter_obstacle_time": obstacle_time,
                "voxel_grid_time": voxel_time,
                "clustering_time": cluster_time,
                "outlier_filter_bbox_time": bb_time,
            },
        )
    else:
        return clusters, cluster_data

import re
import time
import json
import numpy as np

from datetime import datetime
from tqdm import tqdm, tqdm_notebook

from sklearn.model_selection import ParameterGrid
from sklearn.metrics import silhouette_score


def grid_search_optimization(scan, label, obstacle_lst, pipeline, params,
                             score=False, verbose=True):
    """
    Grid Search of hyperparametrs for optimization of executional time of pipeline

    scan: numpy.array,
    An N X 3 array of point cloud from LIDAR

    label: numpy.array,
    A list of segmentation labels respectively

    obstacle_lst: list,
    A list of obstacles id

    pipeline: function,
    An obstacle-detection pipeline function

    params: dict,
    A dictionary of params range that is reqired to search

    verbose: bool, optional, defualt True
    Whether or not print info during execution.

    """
    time_exec_dct = {}
    try:
        for param in tqdm_notebook(ParameterGrid(params), total=len(ParameterGrid(params)), desc='Scan processed'):
            clusters, cls_data, exec_time =  pipeline(scan, label, obstacle_lst, exec_time=True,
                                                      verbose=False, **param)
            end_time = sum(exec_time.values())
            if verbose:
                print('Total time {} ms. Created {} clusters'.format(end_time, len(clusters)))
                print('*' * 40)
                print(json.dumps(param, indent=3))
                print(json.dumps(exec_time, indent=3))
                print('*' * 40)
                print()
            if score:
                if len(cls_data) > 0 and cls_data['cluster_id'].nunique() > 1 :
                    silh_score = silhouette_score(cls_data[['x', 'y', 'z']], cls_data['cluster_id'])
                else:
                    silh_score = 0
                time_exec_dct[json.dumps(param, indent=3)] = (end_time, clusters.shape[0], silh_score)
            else:
                time_exec_dct[json.dumps(param, indent=3)] = (end_time, clusters.shape[0])
        return time_exec_dct
    except KeyboardInterrupt:
        print('User`s KeyboardInterruption...')
        return time_exec_dct


def get_scan_id(scan):
    return re.findall(r'\d\d\d\d\d.', scan)[0]


def get_bbox_and_stat(scan_lst, labels_lst, obstacle_lst, pipeline,
                      write_path=None, write_rotated=False, detailed=False, **pipeline_params):
    """
    Gettitng bounding boxes for reqired sequence of scans and labels
    Also ability to grep time execution statistic.

    scan_lst: list,
    A list of LIDAR scans

    labels_lst: list,
    A list of labels respectively

    obstacle_lst: list,
    A list of obstacles id

    pipeline: function,
    An obstacle-detection pipeline function with required args

    write_path: string, optional, default None
    A path where to write labels. If None labels will not be recorded

    detailed: bool, optional, default False
    If True there will be time execution statistic returned

    """

    # sanity check
    assert len(scan_lst) == len(labels_lst)

    exec_time_dct = {}
    clusters_minmax_dct = {}
    stats = []

    try:
        for scan, label in tqdm_notebook(zip(sorted(scan_lst), sorted(labels_lst)),
                            total=len(scan_lst), desc='Scan processed'):
            # sanity check
            scan_id = get_scan_id(scan)
            assert scan_id == get_scan_id(label)

            # read scan
            scan = np.fromfile(scan, dtype=np.float32)
            scan = scan.reshape((-1, 4))[:, :3]

            # read label
            label = np.fromfile(label, dtype=np.uint32)
            label = label.reshape((-1))

            start_time = datetime.now()
            # start pipeline
            if detailed:
                clusters, cluster_data, stat = pipeline(scan, label, obstacle_lst, exec_time=True, **pipeline_params)
                stats.append(stat)
            else:
                clusters, _ = pipeline(scan, label, obstacle_lst, **pipeline_params)

            end_time =  datetime.now() - start_time
            exec_time_dct[str(scan_id)[-3:]] = end_time.total_seconds()
            clusters_minmax_dct[str(scan_id)[-3:]] = clusters

            if write_path:

                if write_rotated:
                    clusters_rotated = np.empty((0,18))
                    for cluster_id in cluster_data['cluster_id'].unique():
                        tcluster = cluster_data[cluster_data['cluster_id'] == cluster_id][['x', 'y', 'z']]
                        min_poitns = [tcluster[tcluster.index == indx].values for indx in list(tcluster.idxmin())]
                        max_points = [tcluster[tcluster.index == indx].values for indx in list(tcluster.idxmax())]
                        vertices_lst = min_poitns + max_points
                        varray = vertices_lst[0]
                        for v in vertices_lst[1:]:
                            varray = np.concatenate((varray, v), axis=1)
                        clusters_rotated = np.concatenate((clusters_rotated, varray), axis=0)
                    clusters = clusters_rotated

                # write cluster in format x_min, x_max, y_min, y_max, z_min, z_max
                assert isinstance(clusters, np.ndarray)
                np.savetxt(write_path + str(scan_id) + '.bbox', clusters)
    except KeyboardInterrupt:
        print('User`s KeyboardInterruption...')
        return clusters_minmax_dct, exec_time_dct, stats

    return clusters_minmax_dct, exec_time_dct, stats

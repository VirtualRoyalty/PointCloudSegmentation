import re
import time
import json
import numpy as np
import pandas as pd

from datetime import datetime
from tqdm import tqdm, tqdm_notebook

from sklearn.model_selection import ParameterGrid
from sklearn.metrics import silhouette_score
from pipeline import common


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
                      write_path=None, OBB=False, write_seg_id=False, detailed=False,
                      seg_model=None, **pipeline_params):
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
            scan = scan.reshape((-1, 4))

            start_time = datetime.now()
            if seg_model:
                seg_time = datetime.now()
                scan = common.roi_filter(pd.DataFrame(scan, columns=['x', 'y', 'z', 'remission']),
                                        min_x=pipeline_params['roi_x_min'], max_x=pipeline_params['roi_x_max'],
                                        min_y=pipeline_params['roi_y_min'], max_y=pipeline_params['roi_y_max'],
                                        min_z=pipeline_params['roi_z_min'], max_z=pipeline_params['roi_z_max'],
                                        verbose=False)[['x', 'y', 'z', 'remission']].values
                label = seg_model.infer(scan)
                seg_time = (datetime.now() - seg_time).total_seconds()
            else:
                # read label
                label = np.fromfile(label, dtype=np.uint32)
                label = label.reshape((-1))

            # start pipeline
            if detailed:
                clusters, cluster_data, stat = pipeline(scan[:, :3], label, obstacle_lst, exec_time=True, **pipeline_params)
                if seg_model:
                    stat['segmentation_time'] = seg_time
                stats.append(stat)
            else:
                clusters, _ = pipeline(scan, label, obstacle_lst, **pipeline_params)

            end_time =  datetime.now() - start_time
            exec_time_dct[str(scan_id)[-3:]] = end_time.total_seconds()
            clusters_minmax_dct[str(scan_id)[-3:]] = clusters

            if write_path:
                if len(clusters) == 0:
                    np.savetxt(write_path + str(scan_id) + '.bbox', np.empty((0, 0)))
                    if OBB:
                        np.savetxt(write_path + str(scan_id) + '.segs', np.empty((0, 0)))
                    continue
                # Oriented Bounding Boxes
                if OBB:
                    np_clusters = np.empty((0, 24))
                    for cluster in clusters:
                        _obb = []
                        for v in cluster:
                            _obb = _obb + v.tolist()
                        _obb = np.asarray(_obb).reshape(1, 24)
                        np_clusters = np.concatenate((np_clusters, _obb), axis=0)
                    clusters = np_clusters
                # Seg id for additional info e.g. for visualization
                if write_seg_id:
                    seg_lst = []
                    for cl_id in sorted(cluster_data['cluster_id'].unique()):
                        seg = cluster_data[cluster_data['cluster_id'] == cl_id].agg({'seg_id': 'mode'}).values
                        seg_lst.append(seg)
                    seg_arr = np.array(seg_lst, dtype='int64').reshape(1, len(seg_lst))
                    np.savetxt(write_path + str(scan_id) + '.segs', seg_arr)
                # sanity check
                assert isinstance(clusters, np.ndarray)
                # if OBB=False write bounding boxes in format x_min, x_max, y_min, y_max, z_min, z_max
                # else write oriented bounding boxes in format 8 vertixes x1, y1, z1 ... x8, y8, z8
                np.savetxt(write_path + str(scan_id) + '.bbox', clusters)
    except KeyboardInterrupt:
        print('User`s KeyboardInterruption...')
        return clusters_minmax_dct, exec_time_dct, stats

    return clusters_minmax_dct, exec_time_dct, stats

import re
import time
import json
import numpy as np

from datetime import datetime
from tqdm import tqdm, tqdm_notebook

from sklearn.model_selection import ParameterGrid


def grid_search_optimization(scan, label, obstacle_lst, pipeline, params, verbose=True):
    time_exec_lst = {}
    for param in ParameterGrid(params):
        start_time = datetime.now()
        clusters, _, exec_time =  pipeline(scan, label, obstacle_lst,exec_time=True, verbose=False, **param)
        end_time = sum(exec_time.values())
        print('Total time {} ms. Created {} clusters'.format(end_time, len(clusters)))
        if verbose:
            print('*' * 40)
            print(json.dumps(param, indent=3))
            print(json.dumps(exec_time, indent=3))
            print('*' * 40)
            print()
        time_exec_lst[json.dumps(param, indent=3)] = (end_time, len(clusters))
    return time_exec_lst


def get_scan_id(scan):
    return re.findall(r'\d\d\d\d\d.', scan)[0]


def get_bbox_and_stat(scan_lst, labels_lst, obstacle_lst, pipeline,
                      write_path=None, detailed=False, **pipeline_params):
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
                clusters, _, stat = pipeline(scan, label, obstacle_lst, exec_time=True, **pipeline_params)
                stats.append(stat)
            else:
                clusters, _ = pipeline(scan, label, obstacle_lst, **pipeline_params)

            end_time =  datetime.now() - start_time
            exec_time_dct[str(scan_id)[-3:]] = end_time.microseconds / 1000
            clusters_minmax_dct[str(scan_id)[-3:]] = clusters

            if write_path:
                # write cluster in format x_min, x_max, y_min, y_max, z_min, z_max
                clusters_lst = []
                for segment in clusters:
                    for cluster in segment:
                        clusters_lst.append(cluster)
                np.savetxt(write_path + str(scan_id) + '.bbox', clusters_lst)
    except KeyboardInterrupt:
        print('User`s KeyboardInterruption...')
        return clusters_minmax_dct, exec_time_dct, stats

    return clusters_minmax_dct, exec_time_dct, stats

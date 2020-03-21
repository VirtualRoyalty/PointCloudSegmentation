import re
import time

from datetime import datetime
from tqdm import tqdm, tqdm_notebook

import numpy as np

def get_scan_id(scan):
    return re.findall(r'\d\d\d\d\d.', scan)[0]


def get_bbox_and_stat(scan_lst, labels_lst, obstacle_lst,
                      pipeline,  write_path=None, **pipeline_params):

    # sanity check
    assert len(scan_lst) == len(labels_lst)
    exec_time_dct = {}
    clusters_minmax_dct = {}

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


        # start pipeline
        start_time = datetime.now()
        clusters, cluster_data = pipeline(scan, label, obstacle_lst, **pipeline_params)
        end_time =  datetime.now() - start_time
        exec_time_dct[str(scan_id)[-3:]] = end_time
        clusters_minmax_dct[str(scan_id)[-3:]] = clusters

        if write_path:
            # write cluster in format x_min, x_max, y_min, y_max, z_min, z_max
            np.savetxt(write_path + str(scan_id) + '.bbox', clusters)
    return clusters_minmax_dct, exec_time_dct

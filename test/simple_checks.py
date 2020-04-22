import sys
import yaml
import glob
import pytest

import numpy as np
import pandas as pd

sys.path.append('./obstacle-detection/')
from pipeline import common

scan_lst = sorted(glob.glob("./test/data/*.bin"))
label_lst = sorted(glob.glob("./test/data/*.label"))

with open('./test/config.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
obstacle_lst = config['obstacles']


def get_pcloud(scan, label, proc_labels=True):
    scan = np.fromfile(scan, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    scan = scan[:,  :3]
    label = np.fromfile(label, dtype=np.uint32)
    label = label.reshape((-1))
    values = np.concatenate((scan, label.reshape(len(label), 1)), axis=1)
    pcloud = pd.DataFrame(values, columns=['x', 'y', 'z', 'seg_id'])
    if proc_labels:
        pcloud.seg_id = pcloud.seg_id.astype("uint32")
        pcloud.seg_id = pcloud.seg_id.apply(lambda x: x & 0xFFFF)
    return pcloud


class TestClass:
    def test_obstacle_filter_1(self):
        pcloud = get_pcloud(scan_lst[0], label_lst[0])
        cloud = common.obstacle_filter(pcloud, obstacle_lst, proc_labels=False)
        seg_lst = list(cloud['seg_id'].unique())
        for seg in seq_lst:
            assert seg in list(obstacle_lst.keys())

    def test_obstacle_filter_2(self):
        pcloud = get_pcloud(scan_lst[1], label_lst[1])
        cloud = common.obstacle_filter(pcloud, obstacle_lst, proc_labels=False)
        seg_lst = list(cloud['seg_id'].unique())
        for seg in seq_lst:
            assert seg in list(obstacle_lst.keys())

    def test_obstacle_filter_3(self):
        pcloud = get_pcloud(scan_lst[2], label_lst[2])
        cloud = common.obstacle_filter(pcloud, obstacle_lst, proc_labels=False)
        seg_lst = list(cloud['seg_id'].unique())
        for seg in seq_lst:
            assert seg in list(obstacle_lst.keys())

    def test_roi_filter_1(self):
        params = {'roi_x_min': -10, 'roi_x_max': 10, 'roi_y_min': -14, 'roi_y_max': 14, 'roi_z_min': -2, 'roi_z_max': 1}
        pcloud = get_pcloud(scan_lst[0], label_lst[0])
        cloud = common.roi_filter(pcloud,
                                  min_x=params["roi_x_min"],
                                  max_x=params["roi_x_max"],
                                  min_y=params["roi_y_min"],
                                  max_y=params["roi_y_max"],
                                  min_z=params["roi_z_min"],
                                  max_z=params["roi_z_max"],
                                  verbose=False)
        assert cloud['x'].min() >= params['roi_x_min']
        assert cloud['y'].min() >= params['roi_y_min']
        assert cloud['z'].min() >= params['roi_z_min']
        assert cloud['x'].max() <= params['roi_x_max']
        assert cloud['y'].max() <= params['roi_y_max']
        assert cloud['z'].max() <= params['roi_z_max']

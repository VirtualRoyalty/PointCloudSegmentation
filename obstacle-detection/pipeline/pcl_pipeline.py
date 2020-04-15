from pipeline import common
from datetime import datetime
from importlib import reload  
from pipeline import pcl_utils
import time
import pandas as pd
import numpy as np
import pcl

pcl_utils = reload(pcl_utils)

def pipeline_optimized_pcl(scan, label, obstacle_lst, verbose=False, exec_time=False, **params):
    # get segment id
    start_time = datetime.now()
    pcloud = pd.DataFrame(np.concatenate((scan,label.reshape(len(label), 1)), axis=1), 
                          columns=['x', 'y', 'z', 'seg_id'])
    
    pcloud = common.roi_filter(pcloud,  min_x=params['roi_x_min'], max_x=params['roi_x_max'], 
                                                min_y=params['roi_y_min'], max_y=params['roi_y_max'],
                                                min_z=params['roi_z_min'], max_z=params['roi_z_max'], 
                                                verbose=False)
    
    pcloud = common.obstacle_filter(pcloud, obstacle_lst, proc_labels=True, verbose=False)
    pcloud = pcloud.drop(['seg_id'], axis=1)
    pcloud = pcloud.drop(['camera'], axis=1)
    obstacle_time = datetime.now() - start_time
    if (len(pcloud.index) > 0):
        start_time = datetime.now()
        pcloud_pcl = pcl.PointCloud()
        pcloud_pcl.from_array(pcloud.to_numpy(dtype=np.float32))
        convert_time = datetime.now() - start_time

        # get voxel grid
        start_time = datetime.now()
        voxelgrid_id = pcl_utils.voxel_filter(pcloud_pcl, [params['x_voxels'],
                                                              params['y_voxels'],
                                                              params['z_voxels']])
        #voxelgrid_id = pcloud_pcl  
        voxel_time = datetime.now() - start_time

        # ROI filter
        start_time = datetime.now()
        pcloud_roi = pcl_utils.roi_filter(voxelgrid_id, [params['roi_x_min'], params['roi_x_max']], 
                                        [params['roi_y_min'], params['roi_y_max']], 
                                        [params['roi_z_min'], params['roi_z_max']],)
        roi_time = datetime.now() - start_time


        # get cluster
        start_time = datetime.now()
        cluster_data = pcloud_roi.extract([], negative = True)
        cluster_indices = pcl_utils.clustering(cluster_data, params['tol_distance'], params['min_cluster_size'], 150000)        
        clustering_time = datetime.now() - start_time

        # get bboxes
        start_time = datetime.now()
        box_min_max_list, _ = pcl_utils.get_cluster_box_list(
                                                    cluster_indices, cluster_data, 
                                                    radius_search=params['radius_search'], 
                                                    min_neighbors_in_radius=params['min_neighbors_in_radius'])
        bbox_time = datetime.now() - start_time
    else:
        box_min_max_list, cluster_data = np.empty((0, 0)), np.empty((0, 0))
        roi_time, obstacle_time, voxel_time, clustering_time, bbox_time = 0, 0, 0, 0, 0
    
    if verbose:
        print('Execution time:')
        print('\n - ROI filtering: {:.5f} s'.format(roi_time.total_seconds()))
        print('\n - Filtering obstacles: {:.5f} s'.format(obstacle_time.total_seconds()))
        print('\n - Voxel grid: {:.5f} s'.format(voxel_time.total_seconds()))
        print('\n - Clustering: {:.5f} s'.format(clustering_time.total_seconds()))
        print('\n - Min-max cluster points: {:.5f} s \n'.format(bbox_time.total_seconds()))
        
    if exec_time:
        return box_min_max_list, cluster_data, {'roi_time': roi_time.total_seconds(),
                                        'filter_obstacle_time': obstacle_time.total_seconds(),
                                        'voxel_grid_time': voxel_time.total_seconds(),
                                        'clustering_time': clustering_time.total_seconds(),
                                        'outlier_filter_bbox_time': bbox_time.total_seconds(),
                                        'convert_time' : convert_time.total_seconds()}
    else:
        return box_min_max_list, cluster_data
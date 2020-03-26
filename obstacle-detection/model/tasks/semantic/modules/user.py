#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import imp
import yaml
import time
from PIL import Image
import __init__ as booger
import collections
import copy
import os
import numpy as np
from torch.utils.data import Dataset
from common.laserscan import LaserScan, SemLaserScan


from tasks.semantic.modules.segmentator import *
from tasks.semantic.postproc.KNN import KNN


class Inference():
    def __init__(self, ARCH, DATA, datadir, modeldir):
        # parameters
        self.ARCH = ARCH
        self.DATA = DATA
        self.datadir = datadir
        self.modeldir = modeldir

        # get the data
        parserModule = imp.load_source("parserModule",
                                       '/home/jovyan/work/obstacle-detection/model/tasks/semantic/dataset/' +
                                       self.DATA["name"] + '/parser.py')
        self.parser = parserModule.Parser(root=self.datadir,
                                          train_sequences=self.DATA["split"]["train"],
                                          valid_sequences=self.DATA["split"]["valid"],
                                          test_sequences=self.DATA["split"]["test"],
                                          labels=self.DATA["labels"],
                                          color_map=self.DATA["color_map"],
                                          learning_map=self.DATA["learning_map"],
                                          learning_map_inv=self.DATA["learning_map_inv"],
                                          sensor=self.ARCH["dataset"]["sensor"],
                                          max_points=self.ARCH["dataset"]["max_points"],
                                          batch_size=1,
                                          workers=self.ARCH["train"]["workers"],
                                          gt=False,
                                          shuffle_train=False)

        # concatenate the encoder and the head
        with torch.no_grad():
            self.model = Segmentator(self.ARCH,
                                     self.parser.get_n_classes(),
                                     self.modeldir)

        # use knn post processing?
        self.post = None

        self.sensor = self.ARCH["dataset"]["sensor"]
        sensor = self.ARCH["dataset"]["sensor"]
        self.sensor_img_H = sensor["img_prop"]["height"]
        self.sensor_img_W = sensor["img_prop"]["width"]
        self.sensor_img_means = torch.tensor(sensor["img_means"],
                                             dtype=torch.float)
        self.sensor_img_stds = torch.tensor(sensor["img_stds"],
                                            dtype=torch.float)
        self.sensor_fov_up = sensor["fov_up"]
        self.sensor_fov_down = sensor["fov_down"]

        self.nclasses = self.parser.get_n_classes()

        self.max_points = self.ARCH["dataset"]["max_points"]
        # GPU?
        self.gpu = False
        self.model_single = self.model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Infering in device: ", self.device)
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            cudnn.benchmark = True
            cudnn.fastest = True
            self.gpu = True
            self.model.cuda()

    def infer(self, scan_file):
        # do test set
        self.model.eval()

        # empty the cache to infer in high res
        if self.gpu:
            torch.cuda.empty_cache()
        with torch.no_grad():
            # open a semantic laserscan
            scan = LaserScan(project=True,
                             H=self.sensor_img_H,
                             W=self.sensor_img_W,
                             fov_up=self.sensor_fov_up,
                             fov_down=self.sensor_fov_down)
            # open and obtain scan
            scan.open_scan(scan_file)
            # make a tensor of the uncompressed data (with the max num points)
            unproj_n_points = scan.points.shape[0]
            unproj_xyz = torch.full((self.max_points, 3), -1.0, dtype=torch.float)
            unproj_xyz[:unproj_n_points] = torch.from_numpy(scan.points)
            unproj_range = torch.full([self.max_points], -1.0, dtype=torch.float)
            unproj_range[:unproj_n_points] = torch.from_numpy(scan.unproj_range)
            unproj_remissions = torch.full([self.max_points], -1.0, dtype=torch.float)
            unproj_remissions[:unproj_n_points] = torch.from_numpy(scan.remissions)
            unproj_labels = []

            # get points and labels
            proj_range = torch.from_numpy(scan.proj_range).clone()
            proj_xyz = torch.from_numpy(scan.proj_xyz).clone()
            proj_remission = torch.from_numpy(scan.proj_remission).clone()
            proj_mask = torch.from_numpy(scan.proj_mask)
            proj_labels = []
            proj_x = torch.full([self.max_points], -1, dtype=torch.long)
            proj_x[:unproj_n_points] = torch.from_numpy(scan.proj_x)
            proj_y = torch.full([self.max_points], -1, dtype=torch.long)
            proj_y[:unproj_n_points] = torch.from_numpy(scan.proj_y)
            proj = torch.cat([proj_range.unsqueeze(0).clone(),
                              proj_xyz.clone().permute(2, 0, 1),
                              proj_remission.unsqueeze(0).clone()])
            proj = (proj - self.sensor_img_means[:, None, None]
                    ) / self.sensor_img_stds[:, None, None]
            proj = proj * proj_mask.float()

            proj_x = proj_x.unsqueeze(0)
            proj_y = proj_y.unsqueeze(0)
            print("proj_x shape = ", proj_x.shape)
            print("proj_y shape = ", proj_y.shape)
            p_x = proj_x[0, :unproj_n_points]
            p_y = proj_y[0, :unproj_n_points]

            end = time.time()
            if self.gpu:
                proj = proj.cuda()
                proj_mask = proj_mask.cuda()
                #p_x = p_x.cuda()
                #p_y = p_y.cuda()

            # compute output
            proj = proj.unsqueeze(0)
            proj_mask = proj_mask.unsqueeze(0)
            print(proj.shape)
            print(proj_mask.shape)
            proj_output = self.model(proj, proj_mask)
            proj_argmax = proj_output[0].argmax(dim=0)

            unproj_argmax = proj_argmax[p_y, p_x]

            # measure elapsed time
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            print("Infered scan ", scan_file,
                  "in", time.time() - end, "sec")
            end = time.time()

            # save scan
            # get the first scan in batch and project scan
            pred_np = unproj_argmax.cpu().numpy()
            # pred_np = pred_np.reshape((-1)).astype(np.int32)
            return pred_np

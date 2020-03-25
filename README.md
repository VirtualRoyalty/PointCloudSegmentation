# PointCloudSegmentation
---
<img src="https://github.com/VirtualRoyalty/PointCloudSegmentation/blob/dev/obstacle-detection/examples/img/MainGifforGit.gif" width="1000">

---
**Project sctracture:**
```
├───docker-env/
├───obstacle-detection/
│   ├───dataset/
│   │   └───sequences/
│   │       └───00/
│   │           ├───clusters/
│   │           ├───labels/
│   │           └───velodyne/
│   ├───examples/
│   │   
│   ├───pipeline/
│   │  
│   └───scripts/
│       
└───visualization/
```
<br>

## How to dockerize this:
---
- In *base-notebook/* folder start Docker and build an image:
  `$ docker build -t jupyter .`
- After that you can verify a successful build by running: `$ docker images`
- Then start container by running:<br><br>
  `$  docker run -it --rm -p 8888:8888  -v /path/to/obstacle-detection:/home/jovyan/work jupyter` <br><br>
  **NOTE:**  on Windows  you need to convert your path into a quasi-Linux format (*e.g. //c/path/to/obstacle-detection*). More details [here](https://medium.com/@kale.miller96/how-to-mount-your-current-working-directory-to-your-docker-container-in-windows-74e47fa104d7) <br>
  Also, if you want to use drive *D:/* you need to check whether it is mounted or not and if not mount it manually. More details [here](http://support.divio.com/en/articles/646695-how-to-use-a-directory-outside-c-users-with-docker-toolbox-docker-for-windows) if you use Docker toolbox <br><br>
- After correct running you will see URL to access jupyter, e.g.: <br><br>
             *httр://127.0.0.1:8888?token=0cccd15e74216ed2dbe681738ed0f9c78bf65515e94f27a8*<br><br>
- To access jupyter you need to go for **Docker IP**:8888?token=xxxx... <br>( e.g.  httр://192.168.99.100:8888/?token=0cccd15e74216ed2dbe681738ed0f9c78bf65515e94f27a8)<br><br>
- To enter a docker container run `$ docker exec -it *CONTAINER ID* bash` (find out ID by running `$ docker ps`)

## References and useful links:
---
<br>Dataset:

1. [Web-site Semantic KITTI](http://semantic-kitti.org/)
2. [Paper Semantic KITTI](https://arxiv.org/abs/1904.01416)

<br>Segmentation:

3. [Segmentation approaches Point Clouds](https://habr.com/ru/post/459088/)
4. [Also about point cloud segmentation](http://primo.ai/index.php?title=Point_Cloud)
5. [PointNet](http://stanford.edu/~rqi/pointnet/)
6. [PointNet++ from Stanford](http://stanford.edu/~rqi/pointnet2/)
7. [PointNet++](https://towardsdatascience.com/understanding-machine-learning-on-point-clouds-through-pointnet-f8f3f2d53cc3)
8. [RangeNet++](http://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/milioto2019iros.pdf)

<br> Obstacle detection:

9. [Obstacle Detection and Avoidance System for Drones](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5469666/)
10. [3D Lidar-based Static and Moving Obstacle Detection](https://home.isr.uc.pt/~cpremebida/files_cp/3D%20Lidar-based%20static%20and%20moving%20obstacle%20detection%20in%20driving%20environments_Preprint.pdf)
11. [USER-TRAINABLE OBJECT RECOGNITION SYSTEMS](http://www.alexteichman.com/files/dissertation.pdf)
12. [Real-Time Plane Segmentation and Obstacle Detection](https://vk.com/doc136761433_537895530?hash=da67f1d282ddb72f49&dl=957ba302f8b35cd695)

<br> Useful Github links:

13. https://github.com/PRBonn/semantic-kitti-api
14. https://github.com/jbehley/point_labeler
15. https://github.com/daavoo/pyntcloud
16. https://github.com/strawlab/python-pcl
17. https://github.com/kuixu/kitti_object_vis
18. https://github.com/lilyhappily/SFND-P1-Lidar-Obstacle-Detection
19. https://github.com/kcg2015/lidar_ground_plane_and_obstacles_detections
20. https://github.com/enginBozkurt/LidarObstacleDetection

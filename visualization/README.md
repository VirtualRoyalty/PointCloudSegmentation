# Visualization
To visualize the data (in this example sequence 00):
```sh
$ ./visualize.py -d /path/to/dataset/ -s 00
```

To visualize the predictions (in this example sequence 00):

```sh
$ ./visualize.py -d /path/to/dataset/ -p /path/to/predictions/ -s 00
```

If you want to visualize oriented bounding boxes using 8 vertex coordinates use flag -b or --bboxes (in this example sequence 00):

```sh
$ ./visualize.py -d /path/to/dataset/ -p /path/to/predictions/ -s 00 -b
```

If you want to visualize oriented bounding boxes using width, depth, height, center coordinate and angle of rotation use flag -m or --use_bbox_measurements (in this example sequence 00):

```sh
$ ./visualize.py -d /path/to/dataset/ -p /path/to/predictions/ -s 00 -b -m
```

If you want to add cluster label use flag -l or --use_bbox_labels (in this example sequence 00):

```sh
$ ./visualize.py -d /path/to/dataset/ -p /path/to/predictions/ -s 00 -b -l
```

If you want to visualize region of interest use flag -r or --roi_filter (in this example sequence 00):

```sh
$ ./visualize.py -d /path/to/dataset/ -p /path/to/predictions/ -s 00 -r
```

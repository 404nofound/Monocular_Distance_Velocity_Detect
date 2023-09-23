#coding=utf-8
import os
#import subprocess
files=r'Your Video Folder Path'
for file in os.listdir(files):
    # change yolo weights and deep sort checkpoints here
    print(os.system("python track.py --source "+os.path.join(files,file)+" --yolo_model yolov5l6.pt --deep_sort_model osnet_x1_0_imagenet --save-csv"))
    #break

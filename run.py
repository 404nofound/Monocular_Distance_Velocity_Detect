#coding=utf-8
import os
#import subprocess
files=r'F:\\fcw_ldw_0620'
for file in os.listdir(files):
    #print(os.path.join(files,file))
    #print(os.system("activate yolov5"))
    #print(os.system("cd C:\\Users\\Eddy\\Desktop\\Yolov5_DeepSort_Distance_Pytorch"))
    print(os.system("python track.py --source "+os.path.join(files,file)+" --yolo_model yolov5l6.pt --deep_sort_model osnet_x1_0_imagenet --save-csv"))
    #break

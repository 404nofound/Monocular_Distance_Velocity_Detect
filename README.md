# Monocular_Distance_Velocity_Detect
Algorithm based on Yolo v5 and Deep Sort to detect surrounding vehicles' distance and relative velocity

Improved from [Monocular_Distance_Detect](https://github.com/404nofound/Monocular_Distance_Detect)

# The Video Demo
HD Video can be found [Here](https://github.com/404nofound/Monocular_Distance_Velocity_Detect/blob/main/HD_test_ouput.gif)

<img align="center" src="https://github.com/404nofound/Monocular_Distance_Velocity_Detect/blob/main/test_output.gif" alt="" width="640" height="360" style="display: inline; float: right"/>

## Install

- `vs2019` `CUDA` `cuDNN`: Make sure the version fit the requirement of your hardwares
- `Anaconda(Recommend)` `Python>=3.8`
- `Pytorch`: Check the version (MUST be GPU instead of CPU) of packages provided by `conda` before install
- `pip install -r requirement.txt`: install libraries before run the track.py script

Note: `GPU NVIDIA 3060` and above should use `pytorch>=1.11`

## Weights / Checkpoints
- `Yolo_path: ./Monocular_Distance_Velocity_Detect/` 
- `deep sort: C:\Users\Your_Computer_Name\.cache\torch\checkpoints\`

if the program doesn't download the deep sort checkponts automatically, copy the files in `/checkpoints` to the correct path manually.

## Run

```
python track.py --source YOUR_PATH\demo.mp4 --yolo_model yolov5m.pt --deep_sort_model osnet_x1_0_imagenet --show-vid --save-vid --save-csv
```

Note: **yolov5m.pt** & **osnet_x1_0_imagenet** could be selected by yourself.

## Batch Run

if there are many videos under the same folder, modify and execute `run.py`

```
python run.py
```

## Output

Output Path: `./runs/track/`

## Important Parameters in `track.py`

### Video/Image Resolution
```
#Line 58
#Your video/image resolution/size
#画面分辨率
W = 1280
H = 720
```

### Vertical Height
```
#vertical height(m) from camera to the ground/road
#相机离地面高度
H = 0.4
```

### Angle
```
#The angle between the camera len and the horizontal line(the moving direction of vehicle), default is 0
#相机与水平线夹角, 默认为0 相机镜头正对前方，无倾斜
angle_a = 0
```

### Detection Classes

In track.py, we only detect the `['person', 'car', 'truck', 'bicycle', 'motorcycle', 'bus']`,
follow [this](https://blog.csdn.net/weixin_44026604/article/details/115016636) and modified the `code(line 301, 325)` to add more.

### Output Path (saved file)
```
#please update the path before running the track.py script
if save_csv:
  df = pd.DataFrame(storage)
  df.to_excel('Your_path/test.xlsx',index=False)
```

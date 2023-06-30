# limit the number of cpus used by high performance libraries
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.insert(0, './yolov5')

import math

import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import torch
import torch.backends.cudnn as cudnn

from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadImages, LoadStreams, VID_FORMATS
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, save_one_box
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort

from collections import deque

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

pts = [deque(maxlen=30) for _ in range(9999)]

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

W = 1280
H = 720
excel_path = r'./camera_parameters.xlsx'

def camera_parameters(excel_path):
    df_intrinsic = pd.read_excel(excel_path, sheet_name='内参矩阵', header=None)
    df_p = pd.read_excel(excel_path, sheet_name='外参矩阵', header=None)

    print('外参矩阵形状：', df_p.values.shape)
    print('内参矩阵形状：', df_intrinsic.values.shape)

    return df_p.values, df_intrinsic.values

def object_point_world_position(u, v, w, h, p, k):
    u1 = u
    v1 = v + h / 2
    print('图像坐标系关键点：', u1, v1)

    #alpha = -(90 + 0) / (2 * math.pi)
    #peta = 0
    #gama = -90 / (2 * math.pi)

    fx = k[0, 0]
    fy = k[1, 1]
    #相机高度
    #关键参数，不准会导致结果不对
    Height = 0.5
    #相机与水平线夹角, 默认为0 相机镜头正对前方，无倾斜
    #关键参数，不准会导致结果不对
    angle_a = 0
    angle_b = math.atan((v1 - H / 2) / fy)
    angle_c = angle_b + angle_a
    print('angle_b', angle_b)

    depth = (Height / np.sin(angle_c)) * math.cos(angle_b)
    print('depth', depth)

    print('k', k)
    print('p', p)

    k_inv = np.linalg.inv(k)
    p_inv = np.linalg.inv(p)

    point_c = np.array([u1, v1, 1])
    point_c = np.transpose(point_c)
    print('point_c', point_c)
    print('k_inv', k_inv)
    print('p_inv', p_inv)
    #相机坐标系下的关键点位置
    c_position = np.matmul(k_inv, depth * point_c)
    print('相机坐标系c_position', c_position)
    #世界坐标系下
    c_position = np.append(c_position, 1)
    c_position = np.transpose(c_position)

    c_position = np.matmul(p_inv, c_position)
    print('世界坐标系position', c_position)
    d1 = np.array((c_position[0], c_position[1]), dtype=float)

    return d1

def distance_func(kuang, xw=5, yw=0.1):
    print('\n','=' * 50)
    print('开始测距')
    #fig = go.Figure()
    #p外参矩阵, k内参矩阵
    p, k = camera_parameters(excel_path)
    if len(kuang):
        obj_position = []
        #u, v, w, h = kuang[1] * W, kuang[2] * H, kuang[3] * W, kuang[4] * H
        u, v, w, h = kuang[1], kuang[2], kuang[3], kuang[4]
        # u,v中心点坐标 w,h框宽和框高
        print('中心点', u, v)
        print('框宽/高', w, h)
        d1 = object_point_world_position(u, v, w, h, p, k)
    distance = 0
    print('距离', d1)
    if d1[0] <= 0:
        d1[:] = 0
    else:
        distance = math.sqrt(math.pow(d1[0], 2) + math.pow(d1[1], 2))

    return distance, d1

def detect(opt):
    out, source, yolo_model, deep_sort_model, show_vid, save_vid, save_txt, save_csv, imgsz, evaluate, half, \
        project, exist_ok, update, save_crop = \
        opt.output, opt.source, opt.yolo_model, opt.deep_sort_model, opt.show_vid, opt.save_vid, \
        opt.save_txt, opt.save_csv, opt.imgsz, opt.evaluate, opt.half, opt.project, opt.exist_ok, opt.update, opt.save_crop
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = select_device(opt.device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder

    # Directories
    if type(yolo_model) is str:  # single yolo model
        exp_name = yolo_model.split(".")[0]
    elif type(yolo_model) is list and len(yolo_model) == 1:  # single models after --yolo_model
        exp_name = yolo_model[0].split(".")[0]
    else:  # multiple models after --yolo_model
        exp_name = "ensemble"
    exp_name = exp_name + "_" + deep_sort_model.split('/')[-1].split('.')[0]
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run if project name exists
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    model = DetectMultiBackend(yolo_model, device=device, dnn=opt.dnn)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    # Dataloader
    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = len(dataset)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = 1
    vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)

    # Create as many trackers as there are video sources
    deepsort_list = []
    for i in range(nr_sources):
        deepsort_list.append(
            DeepSort(
                deep_sort_model,
                device,
                max_dist=cfg.DEEPSORT.MAX_DIST,
                max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
            )
        )
    outputs = [None] * nr_sources

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run tracking
    model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0

    storage = []
    distance_list = pd.DataFrame()
    outputs_list = pd.DataFrame()

    name_str=''

    for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
        print('frame_idx',frame_idx)
        name_str=path.split('\\')[-1].split('.')[0]

        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if opt.visualize else False
        pred = model(im, augment=opt.augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, max_det=opt.max_det)
        dt[2] += time_sync() - t3

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            print('i:::::::::',i)
            seen += 1
            if webcam:  # nr_sources >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                p = Path(p)  # to Path
                s += f'{i}: '
                txt_file_name = p.name
                save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                # video file
                if source.endswith(VID_FORMATS):
                    txt_file_name = p.stem
                    save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                # folder with imgs
                else:
                    txt_file_name = p.parent.name  # get folder name containing current img
                    save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...

            txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop

            annotator = Annotator(im0, line_width=2, pil=not ascii)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                #det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape)
                #print(det[:, :4])

                # Print results
                for c in det[:, -1].unique():
                    if not names[int(c)] in ['person', 'car', 'truck', 'bicycle', 'motorcycle', 'bus']:
                        continue
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to deepsort
                t4 = time_sync()
                outputs[i] = deepsort_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                t5 = time_sync()
                dt[3] += t5 - t4

                outputs_list=outputs_list.append([[frame_idx,outputs[0]]])



                # 这里提供roi的坐标   [173, 456], [966, 91], [1240, 122], [574, 515]
                # pts1 = np.array([[200, 100], [500, 100], [500, 300], [200, 300]], np.int32)
                # pts1 = pts1.reshape((-1, 1, 2))
                # cv2.polylines(im0, [pts1], True, (0, 255, 255), thickness=2)

                # draw boxes for visualization
                distance_temp=[]
                if len(outputs[i]) > 0:
                    for j, (output, conf) in enumerate(zip(outputs[i], confs)):

                        cls = output[5]

                        if not names[int(cls)] in ['person', 'car', 'truck', 'bicycle', 'motorcycle', 'bus']:
                            continue

                        bboxes = output[0:4]
                        points_list=[[output[0],output[1]],[output[2],output[3]],[output[2],output[1]],[output[0],output[3]]]
                        # next=False
                        # for e in points_list:
                        #     if (200<=e[0]<=500) and (100<=e[1]<=300):
                        #         next=True
                        #         break
                        # if not next:
                        #     continue

                        #print(output)
                        id = output[4]

                        pre_width=-999
                        pre_frame=-999
                        if len(outputs_list)>0:
                            for ind in range(frame_idx-1,-1,-1):
                                # if (len(outputs_list)-1)<ind:
                                #     continue
                                # if len(outputs_list[ind])<2:
                                #     continue
                                if len(outputs_list[outputs_list[0]==ind])!=0:
                                    # [[1,2,3]]->[1,2,3]
                                    temp_outputs_list=outputs_list[outputs_list[0]==ind][1].to_list()[0]
                                    for output_item in temp_outputs_list:
                                        if len(output_item)>0:
                                            print('output',output_item)
                                            if output_item[4]==id:
                                                pre_width=abs(output_item[2]-output_item[0])
                                                pre_frame=ind
                                                break
                                    if pre_width!=-999:
                                        break
                        print('pre_frame:',pre_frame)

                        # pre_distance=-999
                        # for distance_item in distance_list:
                        #     if distance_item[1]==id:
                        #         pre_distance=distance_item[2]
                        #         break

                        thickness=2
                        color=compute_color_for_labels(id)
                        center=(int((output[0]+output[2])/2),int((output[1]+output[3])/2))
                        pts[id].append(center)

                        center_bottom=(int(output[0]+abs(output[2]-output[0])/2),int(output[1]+abs(output[3]-output[1])))

                        kuang = [cls, (output[0]+output[2])/2, (output[1]+output[3])/2, abs(output[2]-output[0]), abs(output[3]-output[1])]
                        distance, d = distance_func(kuang)

                        if d[0]>0:
                            print('d',d)
                            distance_temp.append([id,d[0]])

                        pre_distance=-999
                        if len(distance_list)>0 and pre_frame!=-999:
                            #print(distance_list)
                            print('pre_frame',pre_frame)
                            #if (len(distance_list)-1)>=pre_frame:
                            if len(distance_list[distance_list[0]==pre_frame])!=0:
                                temp_distance_list=distance_list[distance_list[0]==pre_frame][1].to_list()[0]
                                for distance_item in temp_distance_list:
                                    if len(distance_item)>0:
                                        if distance_item[0]==id:
                                            pre_distance=distance_item[1]
                                            break

                        velocity=-999
                        ttc=-999
                        if pre_width!=-999 and pre_distance!=-999:
                            velocity=(((abs(output[2]-output[0])-pre_width)/abs(output[2]-output[0]))*pre_distance)/((frame_idx-pre_frame)/20)
                            if d[0]>0 and velocity!=0:
                                ttc=d[0]/velocity

                        #ttt=[abs(output[2]-output[0]),pre_width,pre_distance,frame_idx,pre_frame]

                        if save_txt:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            # Write MOT compliant results to file

                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                               bbox_top, bbox_w, bbox_h, -1, -1, -1, i))

                        if save_csv:
                            storage.append([name_str, frame_idx+1, id, names[int(cls)], output[0], output[1], distance, d[0], d[1], output[2] - output[0], output[3] - output[1], velocity, ttc])

                        if save_vid or save_crop or show_vid:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = f'{id} {names[c]} {conf:.2f} {d[0]:.2f}m {velocity:.2f}m/s'
                            #draw_boxes(im0, bboxes, id)

                            cv2.circle(im0,(center),1,color,thickness)

                            cv2.circle(im0,(center_bottom),1,color,thickness*2)

                            for j in range(1,len(pts[id])):
                                if pts[id][j-1] is None or pts[id][j] is None:
                                    continue
                                cv2.line(im0,(pts[id][j-1]),(pts[id][j]),(color),thickness)

                            annotator.box_label(bboxes, label, color=colors(c, True))
                            if save_crop:
                                txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                                save_one_box(bboxes, imc, file=save_dir / 'crops' / txt_file_name / names[c] / f'{id}' / f'{p.stem}.jpg', BGR=True)

                distance_list=distance_list.append([[frame_idx,distance_temp]])
                LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), DeepSort:({t5 - t4:.3f}s)')

            else:
                deepsort_list[i].increment_ages()
                LOGGER.info('No detections')

            # Stream results
            im0 = annotator.result()
            if show_vid:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_vid:
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms deep sort update \
        per image at shape {(1, 3, *imgsz)}' % t)

    if save_csv:
        df = pd.DataFrame(storage)
        #df.to_excel(str(save_dir / name_str+'.xlsx'),index=False)
        df.to_excel('C:\\Users\\Eddy\\Desktop\\output_new2\\' +name_str+'.xlsx',index=False)

    if save_txt or save_vid:
        s = f"\n{len(list(save_dir.glob('tracks/*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(yolo_model)  # update model (to fix SourceChangeWarning)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_model', nargs='+', type=str, default='yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--deep_sort_model', type=str, default='osnet_ibn_x1_0_MSMT17')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    parser.add_argument('--save-csv', action='store_true', help='save results to *.csv')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detection per image')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    with torch.no_grad():
        detect(opt)

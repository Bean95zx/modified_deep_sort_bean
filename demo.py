#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import math as m
import os
from timeit import time
import warnings
import sys
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet

warnings.filterwarnings('ignore')

count = 0


def main(yolo):
    # Definition of the parameters
    max_cosine_distance = 0.3  # 最大余弦距离
    nn_budget = None  # ？？
    nms_max_overlap = 1.0  # 非极大值抑制？？

    # deep_sort
    model_filename = 'model_data/mars-small128.pb'
    #  model_filename = 'model_data/darknet_yolov3_model.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)  # encoder编码器

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)  # 余弦度量
    tracker = Tracker(metric)  # 追踪函数

    writeVideo_flag = True

    #video_capture = cv2.VideoCapture(0)  # 获取摄像头数据
    video_capture = cv2.VideoCapture('test5.mp4')  # 获取视频数据

    if writeVideo_flag:
        # Define the codec and create VideoWriter object  定义编码器，并创建 videowriter 对象
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        print("video的w:", w, "。video的h:", h)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # 使用MJPG将视频提取成图片

        out = cv2.VideoWriter('out.avi', fourcc, 15, (w, h))
        out_m = cv2.VideoWriter('out_m.avi', fourcc, 15, (w, h))
        list_file = open('detection.txt', 'w')
        frame_index = -1

    fps = 0.0
    while True:
        print("当前是第", int(video_capture.get(1)), "帧")

        ret, frame = video_capture.read()  # frame 是3维矩阵
        # print("frame是：", frame)
        if ret != True:
            break
        t1 = time.time()

        # image = Image.fromarray(frame)
        image = Image.fromarray(frame[..., ::-1])  # bgr to rgb  bgr图像转rgb图像

        # 视频帧里读取出来的图像一帧(image)是PIL.Image.Image属性
        img = np.asarray(image)  # 对原来image的Image属性，转换为numpy.ndarray属性

        boxs = yolo.detect_image(image)  # 转换后的图像，用yolo去检测
        # print("boxs的type", type(boxs))  boxs的type <class 'list'>
        # print("boxs=", boxs, "\n\n") # 和下面的boxes一样
        features = encoder(frame, boxs)  # 编码器提取features

        # score to 1.0 here.  这里得分是1分
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        # deep_sort在内存中的位置
        # print("detections", detections)  #  deep_sort.detection.Detection object at 0x000002A8FBC6D7B8

        # Run non-maxima suppression.  运行非极大值抑制
        # boxes是左上角的（x,y)坐标
        boxes = np.array([d.tlwh for d in detections])  # tlwh是检测框的(x,y)和宽和高 ，是ndarray数组

        # print("boxes", boxes)
        # print("boxes的shape", boxes.shape, "\nshape元组有几个数：", len(boxes.shape))  # boxes 的形状
        # print("boxes的shape", boxes.shape, "\n")  # boxes 的形状
        a = boxes.shape[0]
        if len(boxes.shape) == 1:
            b = 0
        else:
            b = boxes.shape[1]
        # print("(a,b)=", a, b)  # （a,b)即为boxes的shape

        i = 1
        if b != 0:
            # print("into !=0")
            while i <= a:
                box_temp = boxes[i - 1, :]  # 获取到每一个检测目标的左上角坐标
                # print("boxes的第", i, "行：", box_temp)
                x_center, y_center = travel(box_temp)  # 调用travel函数，得到检测框中心点位置
                anchor_width = box_temp[2]  # anchor_box的宽
                anchor_height = box_temp[3]  # anchor_box的高
                #  print("得到的x_center=", x_center, "y_center", y_center)
                i += 1
            print("one frame detection end\n")

        scores = np.array([d.confidence for d in detections])  # 检测的置信度
        # 目录indices(类型type是indices)，将检测目标通过nms(最大值抑制），减小重叠造成的影响，再存入indices
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]  # detection的类型type是indices的list，就是把indices复制过来
        # print("detection的内容：", detections)  # detection的内容是检测目标在内存中的位置

        # 先看看track更新之前id是什么
        for track in tracker.tracks:
            a = track.track_id
            print("当前的track id是", a)
            name = track.age
            print("当前的track name是", name)
            cov = track.covariance
            print("当前的track covariance是", cov)
            fea = track.features
            print("当前的track features是", fea)
            hit = track.hits
            print("当前的track hits是", hit)
            mea = track.mean
            print("当前的track mean是", mea)
            sta = track.state
            print("当前的track sta是", sta)
            tsu = track.time_since_update
            print("当前的track id是", tsu)

        # Call the tracker 调用追踪器
        tracker.predict()
        tracker.update(detections)

        follow_id = []
        follow = []

        # 画阈值线
        cv2.rectangle(img, (0, int(h / 2)), (int(w), int(h / 2)), (255, 0, 255), 1)
        cv2.rectangle(img, (int(w / 2), 0), (int(w / 2), int(h)), (255, 0, 255), 1)
        # 废弃id数
        cv2.putText(img, "waste id  is" + str(count), (660, 100), 0, 1, (125, 255, 125), 2)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            # print("tlbr之前bbox", bbox)
            bbox = track.to_tlbr()
            print("bbox是：", bbox)

            print("在添加之前，follow_id当前是：", follow_id)
            print("在添加之前，follow当前是：", follow)
            follow_id.append(track.track_id)
            follow.append((bbox[0], bbox[1], bbox[2], bbox[3]))
            # 添加之后follow_id和follow数组的输出，在compare函数里
            follow, follow_id = compare(follow, follow_id, bbox, track, w, h)  # 进入此函数比较位置信息
            length2 = len(follow_id)  # 计算follow_id数组长度
            print("length2", length2)
            print("follow", follow)
            print("follow_id", follow_id)
            print("follow_id[length2-1]", follow_id[length2 - 1])
            print("follow[follow_id.index(follow_id[length2 - 1])][0]",
                  follow[follow_id.index(follow_id[length2 - 1])][0])
            print("type follow[follow_id.index(follow_id[length2 - 1])][0]",
                  type(follow[follow_id.index(follow_id[length2 - 1])][0]))
            # 白色框  （追踪框）BGR
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
            # cv2.rectangle(img, (
            #     int(follow[follow_id.index(follow_id[length2 - 1])][0]),
            #     int(follow[follow_id.index(follow_id[length2 - 1])][1])),
            #               (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)

            # 绿色文本 （追踪ID）putText(画面，id，位置坐标，字体，字体大小，颜色，字体厚度，线型
            #  cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200, (255, 255, 0), 2)
            cv2.putText(frame, str(track.track_id), (int(bbox[0]) + 8, int(bbox[1]) + 8), 0, 1, (0, 255, 0), 2)
            # print("int(follow_id[length2 - 1])", (int(follow_id[length2 - 1]) - int(count)))
            # 把folwaste_id减为实际追踪到
            a1 = int(follow_id[length2 - 1]) - 1
            # a1 = int(follow_id[length2 - 1])
            # cv2.putText(img, str(a1), (int(bbox[0]) + 8, int(bbox[1]) + 8), 0, 1, (0, 255, 0), 2)
            cv2.putText(img, str(a1), (int(bbox[0]) + 8, int(bbox[1]) + 8), 0, 1, (0, 255, 0), 2)
            print("追踪ID是：", track.track_id)

            # 　视频检测框上方输出当前ID
            cv2.putText(frame, "current id is " + str(track.track_id), (int(bbox[0]), 40), 0, 1, (125, 155, 125), 2)

            cv2.putText(img, "current id is" + str(a1), (int(bbox[0]), 80), 0, 1, (125, 255, 0), 2)

            print("one frame track end")

        for fol in follow:
            print("fol是", fol)
            cv2.rectangle(img, (int(fol[0]), int(fol[1])), (int(fol[2]), int(fol[3])), (255, 255, 255), 2)

        for det in detections:
            bbox = det.to_tlbr()
            # 蓝色框（检测框）BGR，而不是RGB
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)

        # h, w, l = np.shape(frame)
        # print("hwl:", h, w, l)

        # 每一帧视频的显示
        cv2.imshow('origin Frame', frame)
        # cv2.rectangle(img, (400, 200), (500, 600), (255, 255, 0), 2)  # 图像中指定位置画框 记得删除
        cv2.imshow('modify frame', img)  # 显示当前帧，会随视频走动

        if writeVideo_flag:
            # save a frame  保存每一帧
            out.write(frame)  # 存入out.avi里
            out_m.write(img)  # 第二个视频窗口的保存
            frame_index = frame_index + 1
            list_file.write(str(frame_index) + ' ')  # 存入detection.txt(list_file)
            if len(boxs) != 0:
                for i in range(0, len(boxs)):
                    list_file.write(
                        str(boxs[i][0]) + ' ' + str(boxs[i][1]) + ' ' + str(boxs[i][2]) + ' ' + str(boxs[i][3]) + ' ')

            list_file.write('\n')

        # 控制台输出fps
        fps = (fps + (1. / (time.time() - t1))) / 2

        print("fps= %f" % (fps))
        print("一帧结束\n\n")

        # Press Q to stop!  关闭窗口
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 清空视频流
    video_capture.release()
    if writeVideo_flag:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()


def compare(follow, follow_id, bbox, track, w, h):
    global count
    print("进入compare:")
    print("compare里count=", count)
    print("compare处理前follow_id是", follow_id)
    length = len(follow_id)
    print("follow id 数组长度是", len(follow_id))
    print("compare处理前follow是", follow)
    print("画面尺寸是:长：", w, "高", h)
    center_x = w / 2
    center_y = h / 2

    for xy1 in follow:
        print("这里xy1是", xy1)
        for xy2 in follow:
            print("这里xy2是", xy2)
            if follow.index(xy1) >= follow.index(xy2):  # 如果xy1位置在xy2位置后，就别删
                continue
            print("follow[xy1]", xy1)
            print("follow[xy2]", xy2)
            print("compare的for循环里count=", count)
            print("follow中xy1的位置", follow.index(xy1))
            print("follow中xy2的位置", follow.index(xy2))
            if follow.index(xy1) != follow.index(xy2):
                distance = OuDistance(xy1, xy2)  # 两个预测框的距离
                distance2 = CenterDistance(xy1, w, h)  # 旧预测框距离中心的距离
                print("distance返回的是=", distance)
                print("disntance2返回的是=", distance2)
                if distance != 0 & (follow.index(xy1) != follow.index(xy2)):
                    print(xy1)
                    # 设定两框距离差多少，距离阈值线多远，判断为一个
                    # 对距离中心点的距离进行分类，1280*720画面，距中心距离最大是734
                    # if (abs(xy1[1] - 420) <= 50 and distance <= 50) or (abs(xy1[1] - 420) >= 30 and distance <= 120):
                    if ((distance2 < 100 and distance <= 10) or
                            (100 <= distance2 < 300 and 10 < distance <= 15) or
                            (300 <= distance2 < 550 and 15 < distance < 30) or
                            (550 <= distance2 and 30 < distance < 80)):
                        # or (distance <= 3)):
                        # if (abs(xy1[1] - 400) <= 10):  # (abs(xy1[1] - 400) >= 10):
                        # 对很近的同两个个目标处理
                        print("coming")
                        count = count + 1
                        print("compare的for的if里count加一了是=", count)
                        print("这里follow改之前是是", follow)
                        print("这里follow_id改之前是", follow_id)
                        print("follow.index(xy2)是（新的位置在follow的下标）", follow.index(xy2))
                        print("follow_id[follow.index(xy2)]（新的位置在follow_id的下标)", follow_id[follow.index(xy2)])
                        print("follow[follow.index(xy2)]", follow[follow.index(xy2)])
                        # follow_id[follow.index(xy1)] = follow_id[follow.index(xy2)]  # 把之前的id给新的目标
                        temp = follow.index(xy2)  # 获取新位置的坐标
                        temp2 = follow.index(xy1)  # 获取旧位置的坐标
                        print("temp", temp, "temp2", temp2)
                        print("temp位置上的follow是", follow[temp])
                        print("temp2位置上的follow是", follow[temp2])
                        print("xy2", xy2)
                        follow[temp2] = xy2  # 将旧的位置 的坐标更新为新的位置
                        print("删除的是", follow[temp])
                        follow.remove(follow[temp])  # 并把新位置的值删除，因为旧的位置已经有了
                        print("改后follow是", follow)
                        a = follow_id[temp]  # 暂存新位置id
                        follow_id.remove((follow_id[temp]))
                        f = open('save.txt', 'a')  # compare修改记录文件
                        f.write("\n原来位置ID是:")
                        f.write(str(follow_id[temp2]))
                        f.write("原来位置坐标是:")
                        f.write(str(xy1))
                        f.write("\n新的位置ID是:")
                        f.write(str(a))
                        f.write("新的位置坐标是:")
                        f.write(str(xy2))
                        f.close()
    return follow, follow_id


def OuDistance(f1, f2):
    # 计算两个boundingbox的距离
    print("进入OuDistance")
    # print("f1=", f1, "f2=", f2)
    distance = m.pow((f1[0] - f2[0]), 2) + m.pow((f1[1] - f2[1]), 2)

    distance = m.sqrt(distance)
    print("OuDistance 计算的 distance = ", distance)
    # print("开平方后的distance是")
    return distance


def CenterDistance(xy1, w, h):
    # 计算预测框距离中心点的距离
    # print("CenterDistance中的w和h,w是", w, "h是", h)
    print("CenterDistance中xy1是", xy1)
    a = int(xy1[0])
    b = int(xy1[1])
    w = w / 2
    h = h / 2
    distance2 = m.pow((a - w), 2) + m.pow((b - h), 2)
    distance2 = m.sqrt(distance2)
    # print("distance2的结果", distance2)
    return distance2


def travel(box_temp):
    # 获取anchor_box的左上角坐标和宽和高
    x_left = box_temp[0]  # 左上角的x坐标
    y_left = box_temp[1]  # 左上角的y坐标
    anchor_box_width = box_temp[2]  # anchor的width
    anchor_box_height = box_temp[3]  # anchor的height
    print("anchor_box的width", anchor_box_width, "。anchor_box的height", anchor_box_height)

    # 转换为中心点的坐标
    x_center = x_left + (anchor_box_width / 2)
    y_center = y_left + (anchor_box_height / 2)
    print("x_left=", x_left, "。y_left", y_left)
    print("x_center=", x_center, "y_center", y_center)
    #  print("travel end")
    return x_center, y_center  # 返回anchor_box的中心点，x_center和y_center


if __name__ == '__main__':
    main(YOLO())

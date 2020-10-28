# vim: expandtab:ts=4:sw=4
import numpy as np
import cv2


def non_max_suppression(boxes, max_bbox_overlap, scores=None):
    """Suppress overlapping detections.  抑制重叠检测

    Original code from [1]_ has been adapted to include confidence score.  原始代码来自[1]的网址已修改，以包括置信度评分。
        [1] http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/

    Examples
    --------
        >>> boxes = [d.roi for d in detections]
        >>> scores = [d.confidence for d in detections]
        >>> indices = non_max_suppression(boxes, max_bbox_overlap, scores)
        >>> detections = [detections[i] for i in indices]

    Parameters
    ----------
    boxes : ndarray
        Array of ROIs (x, y, width, height).
    max_bbox_overlap : float
        ROIs that overlap more than this values are suppressed.
    scores : Optional[array_like]
        Detector confidence score.

    Returns
    -------
    List[int]
        Returns indices of detections that have survived non-maxima suppression.

    """
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats -- 如果边界框为整数，则将它们转换为浮点数
    # this is important since we'll be doing a bunch of divisions  这很重要，因为我们将进行很多划分
    boxes = boxes.astype(np.float)
    # initialize the list of picked indexes  初始化选择的索引列表
    pick = []

    # grab the coordinates of the bounding boxes 抓取边界框的坐标
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2] + boxes[:, 0]
    y2 = boxes[:, 3] + boxes[:, 1]

    # compute the area of the bounding boxes and sort the bounding 计算边界框的面积并对边界排序
    # boxes by the bottom-right y-coordinate of the bounding box 框由边界框的右下角y坐标
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    if scores is not None:
        idxs = np.argsort(scores)
    else:
        idxs = np.argsort(y2)
    
    # keep looping while some indexes still remain in the indexes  保持循环，而某些索引仍保留在索引中
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the 获取索引列表中的最后一个索引，然后添加
        # index value to the list of picked indexes  索引值到选择的索引列表
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of 找到最大的（x，y）坐标作为起点
        # the bounding box and the smallest (x, y) coordinates 边界框和最小（x，y）坐标
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])


        # compute the width and height of the bounding box  计算边界框的宽度和高度
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap 计算重叠率
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have 从索引列表中删除所有具有
        idxs = np.delete(
            idxs, np.concatenate(
                ([last], np.where(overlap > max_bbox_overlap)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return pick

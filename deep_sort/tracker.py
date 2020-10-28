# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track


class Tracker:
    """
    This is the multi-target tracker.

    Parameters  参数
    ----------
    metric（度量单位） : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association. 跟踪关联的度量单位
    max_age : int
        Maximum number of missed misses before a track is deleted.  ？？在跟踪结束前的最大丢失数
    n_init : int
        Number of consecutive detections before the track is confirmed.
            在追踪上之前的连续检测次数
        The track state is set to `Deleted` if a miss occurs within the first `n_init` frames.
            当第一个’n_init框内发生丢失‘追踪状态被设定为Deleted

    Attributes  属性
    ----------
    metric（度量单位） : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association. 跟踪关联的度量单位
    max_age : int
        Maximum number of missed misses before a track is deleted.  ？？在跟踪结束前的最大丢失数
    n_init : int
        Number of frames that a track remains in initialization phase.  在追踪初始化阶段的帧数量
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.  在图像里过滤目标轨迹的卡尔曼滤波器
    tracks : List[Track]
        The list of active tracks at the current time step.  当前时间步下，活动追踪器的集合。

    track in tracks：有好几种状态属性函数：
    is_confirmed（）即为trackstated.confirmed   确认的
    is_deleted()即为trackstated.deleted         删除的
    is_tentatived()即为trackstated.tentative    暂定的

    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=30, n_init=3):
        self.metric = metric  # 跟踪关联的度量单位
        self.max_iou_distance = max_iou_distance  # 最大交并比距离
        self.max_age = max_age  # 在跟踪结束前的最大丢失数
        self.n_init = n_init  # 在追踪初始化阶段的帧数量

        self.kf = kalman_filter.KalmanFilter()  # 卡尔曼滤波器
        self.tracks = []  # 活动追踪器的集合
        self._next_id = 1

    def predict(self):
        """Propagate track state distributions one time step forward.  传递追踪状态（每一步都传播一次）

        This function should be called once every time step, before `update`.  这个方法在update之前，每次都要调用
        """
        print("追踪预测开始")
        for track in self.tracks:
            track.predict(self.kf)  # 活动追踪器的集合每一步都卡尔曼滤波预测

    def update(self, detections):
        """Perform measurement update and track management. 执行测量更新和跟踪管理

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.  detections集合，当前步骤检测到的集合

        """
        # Run matching cascade. 运行级联匹配
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)  # _match方法在后面实现
        print("matches", matches, "  unmatched_tracks", unmatched_tracks, "  unmatched_detections",
              unmatched_detections)
        # Update track set. 更新追踪器集合
        for track_idx, detection_idx in matches:
            print("track_idx是", track_idx, "detection_idx是", detection_idx)
            self.tracks[track_idx].update(self.kf, detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])  # _initiate_track方法在后面实现
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.  更新距离度量
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]  # 活跃目标
        features, targets = [], []  # 定义features和targets两个集合
        for track in self.tracks:
            if not track.is_confirmed():  # 如果track不是is_confirmed（）状态，则跳过
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        # 使用features和targets数据，更新距离
        self.metric.partial_fit(np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            # 获取features 和 targets
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])

            # 计算cost_matrix 代价矩阵  ,通过nn_matching的distance函数，计算距离
            cost_matrix = self.metric.distance(features, targets)
            # print("match的cost_matrix:", cost_matrix)
            cost_matrix = linear_assignment.gate_cost_matrix(  # 调用liner_assignment线性规划的gate_cost_matrix()函数
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks. 将追踪集合，分为确认追踪，和未确认追踪
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.  # 通过appearance features关联确认的跟踪目标
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.  # 通过交并比IOU，关联未确认的追踪
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):  # 初始化追踪
        # mean 均值  convariance 协方差
        mean, covariance = self.kf.initiate(detection.to_xyah())  # 初始化卡尔曼滤波器
        # 往tracks集合里添加
        self.tracks.append(Track(mean, covariance, self._next_id, self.n_init, self.max_age, detection.feature))
        self._next_id += 1

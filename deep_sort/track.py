# vim: expandtab:ts=4:sw=4
class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.
    单一目标跟踪状态的枚举类型。在收集到足够的证据之前，新创建的追踪被归类为“暂时的”
    。然后，追踪状态改为“已确认”。不再存在的追踪状态被归类为“已删除”，以标记它们
    将从活动追踪集中删除。

    """

    Tentative = 1  # 暂时的追踪状态
    Confirmed = 2  # 已确认的追踪状态
    Deleted = 3  # 已删除的追踪状态


class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.
    (x,y)追踪框的中心，a是纵横比（tanα），h是高

    Parameters
    ----------
    mean : ndarray 均值
        Mean vector of the initial state distribution. 初始状态分布的均值向量
    covariance : ndarray  协方差
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        连续(连续)检测前跟踪确认。如果在第一个' n_init '帧中出现错误，跟踪状态将被设置为' Deleted '
        Number of consecutive（连续） detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first `n_init` frames.
    max_age : int
        在跟踪状态设置为“已删除”之前，连续丢失的最大次数。
        The maximum number of consecutive misses before the track state is set to `Deleted`.
    feature : Optional[ndarray]
        该追踪检测的特征向量来源于此。如果非空，则将此特征，添加到“特征”缓存中。
        Feature vector of the detection this track originates from.
        If not None,this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray 均值
        Mean vector of the initial state distribution.
    covariance : ndarray  协方差
        Covariance matrix of the initial state distribution.
    track_id : int  追踪ID
        A unique track identifier.
    hits : int　测量更新的总数
        Total number of measurement updates.
    age : int　自首次出现的总帧数
        Total number of frames since first occurance.　
    time_since_update : int　自上次测量更新以来的帧总数
        Total number of frames since last measurement update.
    state : TrackState　当前追踪状态
        The current track state.
    features : List[ndarray]　
        特征的缓存。在每次测量更新时，相关的特征向量将添加到此列表中。
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.

    """

    def __init__(self, mean, covariance, track_id, n_init, max_age,
                 feature=None):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self.state = TrackState.Tentative
        self.features = []
        if feature is not None:
            self.features.append(feature)

        self._n_init = n_init
        self._max_age = max_age

    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:4].copy()  # ret [0,1,2,3]
        ret[2] *= ret[3]  # ret[2] = ret[2] * ret[3]
        ret[:2] -= ret[2:] / 2  # ret[0] = ret[0]-ret[1]/2  。ret[1] = ret[1]-ret[3]/2
        return ret

    def to_tlbr(self):
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`.

        # demo.py中 bbox = track.to_tlbr()调用的是这个
        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def predict(self, kf):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.

        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.

        """
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah())
        self.features.append(detection.feature)

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted

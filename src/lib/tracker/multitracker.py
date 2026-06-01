from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

from lib.models.model import create_model, load_model
from lib.models.ecdet_jde import ECDetJDEPostProcessor
from lib.tracker import matching as matching
from lib.tracking_utils.kalman_filter import KalmanFilter
from lib.tracking_utils.log import logger
from lib.tracking_utils.utils import *
from .basetrack import BaseTrack, MCBaseTrack, TrackState

from lib.tracking_utils.gmc import GMC

from gen_dataset_visdrone import cls2id, id2cls  # visdrone

# exp2 在exp1的基础上，保留轨迹的最新观测，但是感觉没写对，故建立exp3进行进一步修改
# TODO: Multi-class Track class
class MCTrack(MCBaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, temp_feat, num_classes, cls_id, buff_size=30):
        """
        :param tlwh:
        :param score:
        :param temp_feat:
        :param num_classes:
        :param cls_id:
        :param buff_size:
        """
        # object class id
        self.cls_id = cls_id

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float64)

        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.track_len = 0

        self.smooth_feat = None
        self.update_features(temp_feat)
        self.features = deque([], maxlen=buff_size)  # 指定了限制长度
        self.alpha = 0.9

        self.curr_tlwh = np.asarray(tlwh, dtype=np.float64)

        self.tlwh_deque = deque([], maxlen=30)

    def update_features(self, feat, alpha=None):
        # L2 normalizing
        feat /= np.linalg.norm(feat)
        if alpha is not  None:
            self.alpha= 1-alpha
        else:
            self.alpha = 0.9

        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1.0 - self.alpha) * feat

        self.features.append(feat)

        # L2 normalizing
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(tracks):
        if len(tracks) > 0:
            multi_mean = np.asarray([track.mean.copy() for track in tracks])
            multi_covariance = np.asarray([track.covariance for track in tracks])

            for i, st in enumerate(tracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0

            multi_mean, multi_covariance = MCTrack.shared_kalman.multi_predict(multi_mean, multi_covariance)

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                tracks[i].mean = mean
                tracks[i].covariance = cov


    def multi_gmc(stracks, H=np.eye(2, 3)):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            R = H[:2, :2]

            # keep larger scale factor only // 23.05.03 inpyosong
            larger_scale = max(R[0, 0], R[1, 1])
            uniform_scale_matrix = np.array([[larger_scale, 0], [0, larger_scale]])
            R = uniform_scale_matrix

            R8x8 = np.kron(np.eye(4, dtype=float), R)
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())

                stracks[i].mean = mean
                stracks[i].covariance = cov

    def reset_track_id(self):
        self.reset_track_count(self.cls_id)

    def activate(self, kalman_filter, frame_id):
        """Start a new track"""
        self.kalman_filter = kalman_filter  # assign a filter to each track?

        # update track id for the object class
        self.track_id = self.next_id(self.cls_id)

        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))
        self.curr_tlwh = self._tlwh
        self.track_len = 0
        self.state = TrackState.Tracked  # set flag 'tracked'

        self.tlwh_deque.append((frame_id, self._tlwh))

        # self.is_activated = True
        if frame_id == 1:  # to record the first frame's detection result
            self.is_activated = True

        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        # kalman update
        self.mean, self.covariance = self.kalman_filter.update(self.mean,
                                                               self.covariance,
                                                               self.tlwh_to_xyah(new_track.tlwh))

        # feature vector update
        self.update_features(new_track.curr_feat)

        self.curr_tlwh = new_track.curr_tlwh
        self.tlwh_deque.append((frame_id, new_track.curr_tlwh))

        self.track_len = 0
        self.frame_id = frame_id

        self.state = TrackState.Tracked  # set flag 'tracked'
        self.is_activated = True

        if new_id:  # update track id for the object class
            self.track_id = self.next_id(self.cls_id)

    def update_retrack(self, curr_tlwh, frame_id):
        """
        Update a matched track
        :type new_track: Track
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.track_len += 1

        self.state = TrackState.Tracked  # set flag 'tracked'

        self.curr_tlwh = curr_tlwh

        self.frame_id = frame_id

        # self.mean, self.covariance = self.kalman_filter.update(
        #     self.mean, self.covariance, self.tlwh_to_xyah(curr_tlwh)
        # )

    def update(self, new_track, frame_id, alpha=None,update_feature=True):
        """
        Update a matched track
        :type new_track: Track
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.track_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(self.mean,
                                                               self.covariance,
                                                               self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked  # set flag 'tracked'
        self.is_activated = True  # set flag 'activated'

        self.score = new_track.score

        self.curr_tlwh = new_tlwh
        self.tlwh_deque.append((frame_id, new_track.curr_tlwh))

        if update_feature:
            self.update_features(new_track.curr_feat, alpha)

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()

        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()  # numpy中的.copy()是深拷贝
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def tlwh_to_center(tlwh):
        x, y, w, h = tlwh
        return np.array([x + w / 2.0, y + h / 2.0]), np.array([w, h])

    def __repr__(self):
        return 'OT_({}-{})_({}-{})'.format(self.cls_id, self.track_id, self.start_frame, self.end_frame)



# rewrite a post processing(without using affine matrix)
def map2orig(dets, h_out, w_out, h_orig, w_orig, num_classes):
    """
    :param dets:
    :param h_out:
    :param w_out:
    :param h_orig:
    :param w_orig:
    :param num_classes:
    :return: dict of detections(key: cls_id)
    """

    def get_padding():
        """
        :return: pad_1, pad_2, pad_type('pad_x' or 'pad_y'), new_shape(w, h)
        """
        ratio_x = float(w_out) / w_orig
        ratio_y = float(h_out) / h_orig
        ratio = min(ratio_x, ratio_y)
        new_shape = (round(w_orig * ratio), round(h_orig * ratio))  # new_w, new_h

        pad_x = (w_out - new_shape[0]) * 0.5  # width padding
        pad_y = (h_out - new_shape[1]) * 0.5  # height padding
        top, bottom = round(pad_y - 0.1), round(pad_y + 0.1)
        left, right = round(pad_x - 0.1), round(pad_x + 0.1)
        if ratio == ratio_x:  # pad_y
            return top, bottom, 'pad_y', new_shape
        else:  # pad_x
            return left, right, 'pad_x', new_shape

    pad_1, pad_2, pad_type, new_shape = get_padding()

    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])  # default: 1×128×6
    dets = dets[0]  # 128×6

    dets_dict = {}

    if pad_type == 'pad_x':
        dets[:, 0] = (dets[:, 0] - pad_1) / new_shape[0] * w_orig  # x1
        dets[:, 2] = (dets[:, 2] - pad_1) / new_shape[0] * w_orig  # x2
        dets[:, 1] = dets[:, 1] / h_out * h_orig  # y1
        dets[:, 3] = dets[:, 3] / h_out * h_orig  # y2
    else:  # 'pad_y'
        dets[:, 0] = dets[:, 0] / w_out * w_orig  # x1
        dets[:, 2] = dets[:, 2] / w_out * w_orig  # x2
        dets[:, 1] = (dets[:, 1] - pad_1) / new_shape[1] * h_orig  # y1
        dets[:, 3] = (dets[:, 3] - pad_1) / new_shape[1] * h_orig  # y2

    classes = dets[:, -1]
    for cls_id in range(num_classes):
        inds = (classes == cls_id)
        dets_dict[cls_id] = dets[inds, :]

    return dets_dict


class MCJDETracker(object):
    def __init__(self, opt, frame_rate=30):
        self.opt = opt

        # ----- init model
        print('Creating model...')
        self.model = create_model(opt.arch, opt)
        self.model = load_model(self.model, opt.load_model)
        self.model = self.model.to(opt.device)
        self.model.eval()

        # ----- ECDetJDE postprocessor
        self.postprocessor = ECDetJDEPostProcessor(
            num_classes     = opt.num_classes,
            conf_thres      = opt.conf_thres,
            low_conf_thres  = getattr(opt, 'low_conf_thres', 0.25),
            num_top_queries = getattr(opt, 'num_queries', 300),
        )

        # ----- track_lets
        self.tracked_tracks_dict = defaultdict(list)  # value type: list[STrack]
        self.lost_tracks_dict = defaultdict(list)  # value type: list[STrack]
        self.removed_tracks_dict = defaultdict(list)  # value type: list[STrack]

        self.frame_id = 0
        self.det_thresh = opt.conf_thres
        self.buffer_size = int(frame_rate / 30.0 * opt.track_buffer)  # int(frame_rate / 30.0 * opt.track_buffer)
        self.max_time_lost = self.buffer_size
        self.max_per_image = self.opt.K  # max objects per image
        self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)

        # ----- using kalman filter to stabilize tracking
        self.kalman_filter = KalmanFilter()

        self.past_id_feature = deque([], maxlen=2)
        self.past_reg = deque([], maxlen=2)

        self.gmc = GMC(method='sparseOptFlow', verbose=[None, False])

    def reset(self):
        """
        :return:
        """
        # Reset tracks dict
        self.tracked_tracks_dict = defaultdict(list)  # value type: list[Track]
        self.lost_tracks_dict = defaultdict(list)  # value type: list[Track]
        self.removed_tracks_dict = defaultdict(list)  # value type: list[Track]

        # Reset frame id
        self.frame_id = 0

        # Reset kalman filter to stabilize tracking
        self.kalman_filter = KalmanFilter()


    def update_tracking(self,im_blob, img_0):
        """
        :param im_blob:
        :param img_0:
        :return:
        """
        # update frame id
        self.frame_id += 1

        # ----- reset the track ids for all object classes in the first frame
        if self.frame_id == 1:
            MCTrack.init_count(self.opt.num_classes)

        # record tracking results, key: class_id
        activated_tracks_dict = defaultdict(list)
        refined_tracks_dict = defaultdict(list)
        lost_tracks_dict = defaultdict(list)
        removed_tracks_dict = defaultdict(list)
        output_tracks_dict = defaultdict(list)

        height, width = img_0.shape[0], img_0.shape[1]  # H, W of original input image
        net_height, net_width = im_blob.shape[2], im_blob.shape[3]  # H, W of net input

        c = np.array([width * 0.5, height * 0.5], dtype=np.float32)
        s = max(float(net_width) / float(net_height) * height, width) * 1.0
        h_out = net_height // self.opt.down_ratio
        w_out = net_width // self.opt.down_ratio

        ''' Step 1: Network forward, get detections & embeddings'''
        with torch.no_grad():
            output = self.model.forward(im_blob)
            # ECDetJDE postprocessor: returns high-conf, low-conf dets and reid embeddings
            high_dets_t, low_dets_t, cls_id_feats_t = self.postprocessor(
                output,
                orig_hw=(height, width),
                net_hw=(net_height, net_width),
            )

        # Convert tensors to numpy for downstream tracker
        dets_np      = {cls_id: d.cpu().numpy() for cls_id, d in high_dets_t.items()}
        low_dets_np  = {cls_id: d.cpu().numpy() for cls_id, d in low_dets_t.items()}
        cls_id_feats = {cls_id: r.cpu().numpy() for cls_id, r in cls_id_feats_t.items()}

        # Keep past features for reid_motion (motion-based re-id offset)
        self.past_id_feature.append(cls_id_feats_t)
        self.past_reg.append(None)
        # ----- parse each object class
        for cls_id in range(self.opt.num_classes):
            # High-conf dets already filtered by postprocessor at conf_thres
            cls_dets = dets_np.get(cls_id, np.zeros((0, 6)))
            cls_id_feature = cls_id_feats.get(cls_id, np.zeros((0, self.opt.reid_dim)))

            # Low-conf dets from postprocessor (low_conf_thres <= score < conf_thres)
            # Class 4 disabled for second-pass (was special-cased before)
            if cls_id == 4:
                cls_dets_second       = np.zeros((0, 6))
                cls_id_feature_second = np.zeros((0, self.opt.reid_dim))
            else:
                cls_dets_second = low_dets_np.get(cls_id, np.zeros((0, 6)))
                # Low-conf dets don't have separate reid; use zero embeddings (IoU-only stage)
                cls_id_feature_second = np.zeros((len(cls_dets_second), self.opt.reid_dim))

            if len(cls_dets) > 0:
                '''Detections, tlbrs: top left bottom right score'''
                cls_detects = [
                    MCTrack(MCTrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], feat, self.opt.num_classes, cls_id, 30)
                    for (tlbrs, feat) in zip(cls_dets[:, :5], cls_id_feature)
                ]
            else:
                cls_detects = []

            if len(cls_dets_second) > 0:
                cls_detects_second = [
                    MCTrack(MCTrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], feat, self.opt.num_classes, cls_id, 30)
                    for (tlbrs, feat) in zip(cls_dets_second[:, :5], cls_id_feature_second)
                ]
            else:
                cls_detects_second = []


            ''' Add newly detected tracks to tracked_tracks'''
            unconfirmed_dict = defaultdict(list)
            tracked_tracks_dict = defaultdict(list)
            for track in self.tracked_tracks_dict[cls_id]:
                if not track.is_activated:
                    unconfirmed_dict[cls_id].append(track)
                else:
                    tracked_tracks_dict[cls_id].append(track)

            # building tracking pool for the current frame
            # Predict the current location with KF
            MCTrack.multi_predict(self.lost_tracks_dict[cls_id])
            MCTrack.multi_predict(tracked_tracks_dict[cls_id])

            track_pool_dict = defaultdict(list)
            track_pool_dict[cls_id] = join_tracks(tracked_tracks_dict[cls_id], self.lost_tracks_dict[cls_id])

            ''' Step 2: First association, with embedding'''
            dist_off = matching.reid_motion(track_pool_dict[cls_id], cls_detects, self.past_id_feature,
                                            self.past_reg, h_out, w_out, height, width)
            dists = matching.embedding_distance(track_pool_dict[cls_id], cls_detects)
            dist_iou = matching.iou_distance(track_pool_dict[cls_id], cls_detects)* dist_off
            dist_iou = matching.fuse_score_three(dist_iou, dists, cls_detects)

            matches, u_track, u_detection = matching.linear_assignment(dist_iou, thresh=0.6)

            for i_tracked, i_det in matches:
                track = track_pool_dict[cls_id][i_tracked]
                det = cls_detects[i_det]
                if track.state == TrackState.Tracked:
                    track.update(cls_detects[i_det], self.frame_id)
                    activated_tracks_dict[cls_id].append(track)  # for multi-class
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refined_tracks_dict[cls_id].append(track)

            ''' Step 3: Second association, with IOU'''
            cls_detects = [cls_detects[i] for i in u_detection]
            r_tracked_tracks = [track_pool_dict[cls_id][i]
                                 for i in u_track if track_pool_dict[cls_id][i].state]
            dist_iou = matching.iou_distance(r_tracked_tracks, cls_detects)

            matches, u_track, u_detection = matching.linear_assignment(dist_iou, thresh=0.8)

            for i_tracked, i_det in matches:
                track = r_tracked_tracks[i_tracked]
                det = cls_detects[i_det]
                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_tracks_dict[cls_id].append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refined_tracks_dict[cls_id].append(track)

            """association the untrack to the low score detections"""

            second_tracked_tracks = [r_tracked_tracks[i] for i in u_track]
            dist_iou = matching.iou_distance(second_tracked_tracks, cls_detects_second)
            matches, u_track, u_detection_second = matching.linear_assignment(dist_iou, thresh=0.2)

            for i_tracked, i_det in matches:
                track = second_tracked_tracks[i_tracked]
                det = cls_detects_second[i_det]
                if track.state == TrackState.Tracked:
                    track.update(cls_detects_second[i_det], self.frame_id)
                    activated_tracks_dict[cls_id].append(track)  # for multi-class
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refined_tracks_dict[cls_id].append(track)

            for it in u_track:
                track = second_tracked_tracks[it]

                if track.state == TrackState.Lost:
                    continue

                if self.frame_id - track.end_frame != 1:
                    track.mark_lost()
                    lost_tracks_dict[cls_id].append(track)
                    continue

                if len(track.tlwh_deque) < 10: # 10 for frame1-2-3
                    track.mark_lost()
                    lost_tracks_dict[cls_id].append(track)
                    continue

                frame_id_1, tlwh_1 = track.tlwh_deque[-1]
                frame_id_2, tlwh_2 = track.tlwh_deque[-2]
                frame_id_3, tlwh_3 = track.tlwh_deque[-3]

                if not (frame_id_3 + 1 == frame_id_2 and frame_id_2 + 1 == frame_id_1):
                    track.mark_lost()
                    lost_tracks_dict[cls_id].append(track)
                    continue

                x, y, w, h = track.tlwh

                margin = -3
                if (
                        x <= margin or y <= margin or
                        (x + w) >= (width - margin) or
                        (y + h) >= (height - margin) and len(track.tlwh_deque) >0
                ):
                    track.mark_removed()
                    removed_tracks_dict[cls_id].append(track)
                    continue

                # ECDetJDE-aligned re-activation:
                # Use ReID cosine similarity (from pred_reid) + IoU (from pred_boxes)
                # instead of CenterNet spatial attention on dense feature maps.
                if len(cls_detects) > 0:
                    reid_costs = matching.embedding_distance([track], cls_detects)  # (1, M) cosine dist
                    iou_costs  = matching.iou_distance([track], cls_detects)        # (1, M) 1-iou

                    best_idx       = int(reid_costs[0].argmin())
                    best_reid_cost = reid_costs[0, best_idx]          # low  = same object
                    best_iou_score = 1.0 - iou_costs[0, best_idx]    # high = good spatial overlap
                    max_iou_any    = 1.0 - iou_costs[0].min()        # max iou with any det

                    if best_reid_cost < 0.4 and best_iou_score > 0.0:
                        # Strong appearance match + some spatial proximity → re-associate
                        track.re_activate(cls_detects[best_idx], self.frame_id, new_id=False)
                        refined_tracks_dict[cls_id].append(track)
                    elif max_iou_any <= 0.2:
                        # No detection overlaps this region → object occluded → propagate Kalman
                        track.update_retrack(track.tlwh, self.frame_id)
                        activated_tracks_dict[cls_id].append(track)
                    else:
                        track.mark_lost()
                        lost_tracks_dict[cls_id].append(track)
                else:
                    # No unmatched detections → object likely occluded → keep alive
                    track.update_retrack(track.tlwh, self.frame_id)
                    activated_tracks_dict[cls_id].append(track)

            '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
            cls_detects = [cls_detects[i] for i in u_detection]
            dist_off = matching.reid_motion_unconfirmed(unconfirmed_dict[cls_id], cls_detects, self.past_id_feature,
                                                        self.past_reg, h_out, w_out, height, width)
            dist_iou = matching.iou_distance(unconfirmed_dict[cls_id], cls_detects) * dist_off
            matches, u_unconfirmed, u_detection = matching.linear_assignment(dist_iou, thresh=0.5)

            for i_tracked, i_det in matches:
                unconfirmed_dict[cls_id][i_tracked].update(cls_detects[i_det], self.frame_id)
                activated_tracks_dict[cls_id].append(unconfirmed_dict[cls_id][i_tracked])
            for it in u_unconfirmed:
                track = unconfirmed_dict[cls_id][it]
                track.mark_removed()
                removed_tracks_dict[cls_id].append(track)

            """ Step 4: Init new tracks"""
            for i_new in u_detection:
                track = cls_detects[i_new]
                if track.score < self.det_thresh:
                    continue

                track.activate(self.kalman_filter, self.frame_id)
                activated_tracks_dict[cls_id].append(track)

            """ Step 5: Update state"""
            for track in self.lost_tracks_dict[cls_id]:
                if self.frame_id - track.end_frame > self.max_time_lost:
                    track.mark_removed()
                    removed_tracks_dict[cls_id].append(track)

            self.tracked_tracks_dict[cls_id] = [t for t in self.tracked_tracks_dict[cls_id] if
                                                t.state == TrackState.Tracked]
            self.tracked_tracks_dict[cls_id] = join_tracks(self.tracked_tracks_dict[cls_id],
                                                           activated_tracks_dict[cls_id])
            self.tracked_tracks_dict[cls_id] = join_tracks(self.tracked_tracks_dict[cls_id],
                                                           refined_tracks_dict[cls_id])
            self.lost_tracks_dict[cls_id] = sub_tracks(self.lost_tracks_dict[cls_id],
                                                       self.tracked_tracks_dict[cls_id])
            self.lost_tracks_dict[cls_id].extend(lost_tracks_dict[cls_id])
            self.lost_tracks_dict[cls_id] = sub_tracks(self.lost_tracks_dict[cls_id],
                                                       self.removed_tracks_dict[cls_id])
            self.removed_tracks_dict[cls_id].extend(removed_tracks_dict[cls_id])
            self.tracked_tracks_dict[cls_id], self.lost_tracks_dict[cls_id] = remove_duplicate_tracks(
                self.tracked_tracks_dict[cls_id],
                self.lost_tracks_dict[cls_id])

            # get scores of lost tracks
            output_tracks_dict[cls_id] = [track for track in self.tracked_tracks_dict[cls_id] if track.is_activated]

            logger.debug('===========Frame {}=========='.format(self.frame_id))
            logger.debug('Activated: {}'.format(
                [track.track_id for track in activated_tracks_dict[cls_id]]))
            logger.debug('Refind: {}'.format(
                [track.track_id for track in refined_tracks_dict[cls_id]]))
            logger.debug('Lost: {}'.format(
                [track.track_id for track in lost_tracks_dict[cls_id]]))
            logger.debug('Removed: {}'.format(
                [track.track_id for track in removed_tracks_dict[cls_id]]))

        return output_tracks_dict

def join_tracks(t_list_a, t_list_b):
    """
    join two track lists
    :param t_list_a:
    :param t_list_b:
    :return:
    """
    exists = {}
    res = []
    for t in t_list_a:
        exists[t.track_id] = 1
        res.append(t)
    for t in t_list_b:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_tracks(t_list_a, t_list_b):
    tracks = {}
    for t in t_list_a:
        tracks[t.track_id] = t
    for t in t_list_b:
        tid = t.track_id
        if tracks.get(tid, 0):
            del tracks[tid]
    return list(tracks.values())


def remove_duplicate_tracks(tracks_a, tracks_b):
    p_dist = matching.iou_distance(tracks_a, tracks_b)
    pairs = np.where(p_dist < 0.15)
    dup_a, dup_b = list(), list()

    for p, q in zip(*pairs):
        time_p = tracks_a[p].frame_id - tracks_a[p].start_frame
        time_q = tracks_b[q].frame_id - tracks_b[q].start_frame
        if time_p > time_q:
            dup_b.append(q)
        else:
            dup_a.append(p)

    res_a = [t for i, t in enumerate(tracks_a) if not i in dup_a]
    res_b = [t for i, t in enumerate(tracks_b) if not i in dup_b]

    return res_a, res_b


# ---------------------------------------------------------------------------
# MCByteTracker: ByteTrack-style multi-class tracker for ECDetJDE
#
# Core design differences vs MCJDETracker:
#   1. No update_retrack ghost path — unmatched tracks go to lost immediately
#   2. Postprocessor dual-threshold provides a real low-conf secondary pool
#   3. Stage 1: high-conf dets vs active+lost tracks (ReID+IoU fusion)
#      Stage 2: low-conf dets vs still-unmatched active tracks (IoU only)
#   4. Lost tracks survive max_time_lost frames via Kalman for re-ID
#      but are NEVER emitted in output — eliminating all ghost boxes
# ---------------------------------------------------------------------------
class MCByteTracker:

    def __init__(self, opt, frame_rate=30):
        self.opt = opt

        print('Creating model...')
        self.model = create_model(opt.arch, opt)
        self.model = load_model(self.model, opt.load_model)
        self.model = self.model.to(opt.device)
        self.model.eval()

        low_conf_thres = getattr(opt, 'low_conf_thres', 0.25)
        self.postprocessor = ECDetJDEPostProcessor(
            num_classes     = opt.num_classes,
            conf_thres      = opt.conf_thres,
            low_conf_thres  = low_conf_thres,
            num_top_queries = getattr(opt, 'num_queries', 300),
        )

        self.tracked_tracks_dict = defaultdict(list)
        self.lost_tracks_dict    = defaultdict(list)
        self.removed_tracks_dict = defaultdict(list)

        self.frame_id       = 0
        self.det_thresh     = opt.conf_thres
        self.buffer_size    = int(frame_rate / 30.0 * opt.track_buffer)
        self.max_time_lost  = self.buffer_size
        self.kalman_filter  = KalmanFilter()
        self.gmc            = GMC(method='sparseOptFlow', verbose=[None, False])

        self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std  = np.array(opt.std,  dtype=np.float32).reshape(1, 1, 3)
        self._dummy_feat_cache = {}  # reid_dim → unit vector

    def _dummy_feat(self, reid_dim):
        if reid_dim not in self._dummy_feat_cache:
            v = np.ones(reid_dim, dtype=np.float32)
            self._dummy_feat_cache[reid_dim] = v / np.linalg.norm(v)
        return self._dummy_feat_cache[reid_dim]

    def update_tracking(self, im_blob, img_0):
        self.frame_id += 1

        if self.frame_id == 1:
            MCTrack.init_count(self.opt.num_classes)

        activated_dict = defaultdict(list)
        refined_dict   = defaultdict(list)
        lost_dict      = defaultdict(list)
        removed_dict   = defaultdict(list)
        output_dict    = defaultdict(list)

        height, width       = img_0.shape[0], img_0.shape[1]
        net_height, net_width = im_blob.shape[2], im_blob.shape[3]

        with torch.no_grad():
            raw = self.model(im_blob)
            high_dets_t, low_dets_t, reid_t = self.postprocessor(
                raw,
                orig_hw=(height, width),
                net_hw=(net_height, net_width),
            )

        high_np = {c: d.cpu().numpy() for c, d in high_dets_t.items()}
        low_np  = {c: d.cpu().numpy() for c, d in low_dets_t.items()}
        reid_np = {c: r.cpu().numpy() for c, r in reid_t.items()}

        # Global motion compensation using all high-conf boxes
        all_high = np.concatenate([v for v in high_np.values() if len(v)], axis=0) \
                   if any(len(v) for v in high_np.values()) else np.zeros((0, 6))
        warp = self.gmc.apply(img_0, all_high)

        for cls_id in range(self.opt.num_classes):
            h_dets   = high_np.get(cls_id, np.zeros((0, 6)))   # (M1, 6) xyxy+score+cls
            l_dets   = low_np.get(cls_id,  np.zeros((0, 6)))   # (M2, 6)
            feats    = reid_np.get(cls_id, np.zeros((0, self.opt.reid_dim)))  # (M1, D)
            reid_dim = feats.shape[1] if feats.shape[0] > 0 else self.opt.reid_dim
            dummy    = self._dummy_feat(reid_dim)

            confirmed   = [t for t in self.tracked_tracks_dict[cls_id] if t.is_activated]
            unconfirmed = [t for t in self.tracked_tracks_dict[cls_id] if not t.is_activated]

            # Kalman predict + GMC for all active + lost
            track_pool = join_tracks(confirmed, self.lost_tracks_dict[cls_id])
            MCTrack.multi_predict(track_pool)
            MCTrack.multi_gmc(track_pool, warp)

            # Build detection objects
            high_objs = [
                MCTrack(MCTrack.tlbr_to_tlwh(d[:4]), d[4], feats[i], self.opt.num_classes, cls_id)
                for i, d in enumerate(h_dets)
            ] if len(h_dets) > 0 else []

            low_objs = [
                MCTrack(MCTrack.tlbr_to_tlwh(d[:4]), d[4], dummy, self.opt.num_classes, cls_id)
                for d in l_dets
            ] if len(l_dets) > 0 else []

            # ── Stage 1: high-conf dets ↔ confirmed+lost tracks (ReID+IoU) ─────
            dists_emb = matching.embedding_distance(track_pool, high_objs)
            dists_iou = matching.iou_distance(track_pool, high_objs)
            cost1     = matching.fuse_score_three(dists_iou, dists_emb, high_objs)
            matches1, u_track1, u_det1 = matching.linear_assignment(cost1, thresh=0.55)

            for i_t, i_d in matches1:
                track = track_pool[i_t]
                det   = high_objs[i_d]
                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_dict[cls_id].append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refined_dict[cls_id].append(track)

            # ── Stage 2: low-conf dets ↔ unmatched CONFIRMED tracks (IoU only) ─
            unmatched_confirmed = [track_pool[i] for i in u_track1
                                   if track_pool[i].state == TrackState.Tracked]
            dists_iou2 = matching.iou_distance(unmatched_confirmed, low_objs)
            matches2, u_conf2, _ = matching.linear_assignment(dists_iou2, thresh=0.5)

            for i_t, i_d in matches2:
                track = unmatched_confirmed[i_t]
                det   = low_objs[i_d]
                # update position but don't pollute appearance with low-conf embedding
                track.update(det, self.frame_id, update_feature=False)
                activated_dict[cls_id].append(track)

            # Unmatched confirmed tracks → lost immediately (no ghost propagation)
            for i in u_conf2:
                track = unmatched_confirmed[i]
                track.mark_lost()
                lost_dict[cls_id].append(track)

            # ── Stage 3: unconfirmed tracks ↔ remaining high-conf dets ──────────
            u_high = [high_objs[i] for i in u_det1]
            dists_iou3 = matching.iou_distance(unconfirmed, u_high)
            matches3, u_unconf, u_det3 = matching.linear_assignment(dists_iou3, thresh=0.5)

            for i_t, i_d in matches3:
                unconfirmed[i_t].update(u_high[i_d], self.frame_id)
                activated_dict[cls_id].append(unconfirmed[i_t])
            for i in u_unconf:
                unconfirmed[i].mark_removed()
                removed_dict[cls_id].append(unconfirmed[i])

            # ── Stage 4: init new tracks from remaining high-conf dets ───────────
            for i in u_det3:
                det = u_high[i]
                if det.score >= self.det_thresh:
                    det.activate(self.kalman_filter, self.frame_id)
                    activated_dict[cls_id].append(det)

            # ── Stage 5: expire long-lost tracks ─────────────────────────────────
            for track in self.lost_tracks_dict[cls_id]:
                if self.frame_id - track.end_frame > self.max_time_lost:
                    track.mark_removed()
                    removed_dict[cls_id].append(track)

            # State update
            self.tracked_tracks_dict[cls_id] = [
                t for t in self.tracked_tracks_dict[cls_id] if t.state == TrackState.Tracked
            ]
            self.tracked_tracks_dict[cls_id] = join_tracks(
                self.tracked_tracks_dict[cls_id], activated_dict[cls_id])
            self.tracked_tracks_dict[cls_id] = join_tracks(
                self.tracked_tracks_dict[cls_id], refined_dict[cls_id])

            self.lost_tracks_dict[cls_id] = sub_tracks(
                self.lost_tracks_dict[cls_id], self.tracked_tracks_dict[cls_id])
            self.lost_tracks_dict[cls_id].extend(lost_dict[cls_id])
            self.lost_tracks_dict[cls_id] = sub_tracks(
                self.lost_tracks_dict[cls_id], self.removed_tracks_dict[cls_id])

            self.removed_tracks_dict[cls_id].extend(removed_dict[cls_id])

            self.tracked_tracks_dict[cls_id], self.lost_tracks_dict[cls_id] = \
                remove_duplicate_tracks(self.tracked_tracks_dict[cls_id],
                                        self.lost_tracks_dict[cls_id])

            # Only activated tracks are visible in output — no ghost boxes
            output_dict[cls_id] = [
                t for t in self.tracked_tracks_dict[cls_id] if t.is_activated
            ]

        return output_dict

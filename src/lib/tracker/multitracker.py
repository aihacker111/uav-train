from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

from lib.models import *
from lib.models.decode import mot_decode
from lib.models.model import create_model, load_model
from lib.models.utils import _tranpose_and_gather_feat
from lib.tracker import matching as matching
from lib.tracking_utils.kalman_filter import KalmanFilter
from lib.tracking_utils.log import logger
from lib.tracking_utils.utils import *
from lib.utils.post_process import ctdet_post_process
from .basetrack import BaseTrack, MCBaseTrack, TrackState

from lib.tracking_utils.gmc import GMC

from gen_dataset_visdrone import cls2id, id2cls  # visdrone
# from gen_labels_detrac_mcmot import cls2id, id2cls  # mcmot_c5

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
        self.model = create_model(opt.arch, opt.heads, opt.head_conv,
                                  num_classes=opt.num_classes, opt=opt)
        self.model = load_model(self.model, opt.load_model)  # load specified checkpoint
        self.model = self.model.to(opt.device)
        self.model.eval()

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

        # {cls_id: np.ndarray[N, 5+]} xyxy+score in original image coords
        # populated every frame for mAP computation in track_AMOT.py
        self.last_raw_dets: dict = {}

    @staticmethod
    def _detr_to_orig(boxes_np, scores_np, labels_np, net_h, net_w, h_orig, w_orig):
        """Convert DETR normalized cxcywh [0,1] → per-class xyxy in original image coords."""
        ratio_x = float(net_w) / w_orig
        ratio_y = float(net_h) / h_orig
        ratio   = min(ratio_x, ratio_y)
        new_w   = round(w_orig * ratio)
        new_h   = round(h_orig * ratio)
        pad_x   = (net_w - new_w) * 0.5
        pad_y   = (net_h - new_h) * 0.5
        pad_type = 'pad_y' if ratio == ratio_x else 'pad_x'

        cx = boxes_np[:, 0] * net_w;  bw = boxes_np[:, 2] * net_w
        cy = boxes_np[:, 1] * net_h;  bh = boxes_np[:, 3] * net_h
        x1 = cx - bw / 2;  x2 = cx + bw / 2
        y1 = cy - bh / 2;  y2 = cy + bh / 2

        if pad_type == 'pad_x':
            x1 = np.clip((x1 - pad_x) / new_w * w_orig, 0, w_orig)
            x2 = np.clip((x2 - pad_x) / new_w * w_orig, 0, w_orig)
            y1 = np.clip(y1 / net_h * h_orig, 0, h_orig)
            y2 = np.clip(y2 / net_h * h_orig, 0, h_orig)
        else:
            x1 = np.clip(x1 / net_w * w_orig, 0, w_orig)
            x2 = np.clip(x2 / net_w * w_orig, 0, w_orig)
            y1 = np.clip((y1 - pad_y) / new_h * h_orig, 0, h_orig)
            y2 = np.clip((y2 - pad_y) / new_h * h_orig, 0, h_orig)

        dets = np.stack([x1, y1, x2, y2, scores_np], axis=1).astype(np.float32)
        n_cls = int(labels_np.max()) + 1 if len(labels_np) > 0 else 0
        return {cls_id: dets[labels_np == cls_id] for cls_id in range(n_cls)}

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

    def post_process(self, dets, meta):
        """
        2D bbox检测结果后处理
        :param dets:
        :param meta:
        :return:
        """
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])  # default: 1×128×6

        # affine transform
        dets = ctdet_post_process(dets.copy(),
                                  [meta['c']], [meta['s']],
                                  meta['out_height'],
                                  meta['out_width'],
                                  self.opt.num_classes)

        dets = dets[0]  # fetch the first image dets results(batch_size = 1 by default)

        return dets

    def merge_outputs(self, detections):
        """
        :param detections:
        :return:
        """
        results = {}
        for j in range(1, self.opt.num_classes + 1):
            results[j] = np.concatenate([detection[j] for detection in detections],
                                        axis=0).astype(np.float32)

        scores = np.hstack([results[j][:, 4] for j in range(1, self.opt.num_classes + 1)])
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.opt.num_classes + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]

        return results

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
            raw_output = self.model.forward(im_blob)

            if isinstance(raw_output, dict) and 'stage2' in raw_output:
                # ── Hybrid model: stage-1 seeds ref points for stage-2.
                # Use stage-2 (DETR) as the final refined detections.
                stage2 = raw_output['stage2']

                boxes_t  = stage2.boxes[0]              # (K, 4) cxcywh [0,1]
                probs    = stage2.logits[0].sigmoid()   # (K, C)
                scores_t, labels_t = probs.max(dim=-1)

                boxes_np  = boxes_t.cpu().numpy()
                scores_np = scores_t.cpu().numpy().astype(np.float32)
                labels_np = labels_t.cpu().numpy().astype(int)

                # Pre-filter background queries before per-class split
                keep      = scores_np >= self.opt.conf_thres
                boxes_np  = boxes_np[keep]
                scores_np = scores_np[keep]
                labels_np = labels_np[keep]

                # Convert normalized cxcywh → per-class xyxy in original image coords
                dets = MCJDETracker._detr_to_orig(
                    boxes_np, scores_np, labels_np,
                    net_height, net_width, height, width,
                )

                # Reid: use stage2.reid if available, else constant unit vectors
                _reid_dim = 256
                dummy = np.ones(_reid_dim, dtype=np.float32) / (_reid_dim ** 0.5)
                if stage2.reid is not None:
                    reid_np   = stage2.reid[0].cpu().numpy()   # (K_all, reid_dim)
                    reid_keep = reid_np[keep]                  # filter same as boxes
                else:
                    reid_np   = None
                    reid_keep = None

                cls_id_feats = []
                for cls_id in range(self.opt.num_classes):
                    cls_d = dets.get(cls_id, np.zeros((0, 5), dtype=np.float32))
                    if reid_keep is not None:
                        cls_mask = labels_np == cls_id
                        cls_id_feats.append(reid_keep[cls_mask])
                    else:
                        cls_id_feats.append(
                            np.tile(dummy, (len(cls_d), 1))
                        )

                # Dummy past feature maps so reid_motion helpers don't crash
                cn_out    = raw_output['stage1']
                B, _, H, W = cn_out.hm.shape
                _id_map   = cn_out.hm.new_ones(B, _reid_dim, H, W) / (_reid_dim ** 0.5)
                _reg_map  = cn_out.reg if self.opt.reg_offset else None
                self.past_id_feature.append(_id_map)
                self.past_reg.append(_reg_map)

            else:
                # ── Non-hybrid model: standard CenterNet JDE path
                output     = raw_output[-1]
                hm         = output['hm'].sigmoid_()
                wh         = output['wh']
                reg        = output['reg'] if self.opt.reg_offset else None
                id_feature = F.normalize(output['id'], dim=1)

                self.past_id_feature.append(id_feature)
                self.past_reg.append(reg)

                det_raw, inds, cls_inds_mask = mot_decode(
                    heatmap=hm, wh=wh, reg=reg,
                    num_classes=self.opt.num_classes,
                    cat_spec_wh=self.opt.cat_spec_wh,
                    K=self.opt.K,
                    log_wh=getattr(self.opt, 'log_wh', False),
                )

                cls_id_feats = []
                for cls_id in range(self.opt.num_classes):
                    cls_inds       = inds[:, cls_inds_mask[cls_id]]
                    cls_id_feature = _tranpose_and_gather_feat(id_feature, cls_inds)
                    cls_id_feature = cls_id_feature.squeeze(0).cpu().numpy()
                    cls_id_feats.append(cls_id_feature)

                dets = map2orig(det_raw, h_out, w_out, height, width, self.opt.num_classes)

        self.last_raw_dets = dets
        # ----- parse each object class
        _empty = np.zeros((0, 5), dtype=np.float32)
        for cls_id in range(self.opt.num_classes):
            cls_dets = dets.get(cls_id, _empty) if isinstance(dets, dict) else dets[cls_id]

            # filter out low confidence detections
            remain_inds = cls_dets[:, 4] > self.opt.conf_thres

            if cls_id == 4:
                inds_low = cls_dets[:, 4] > 1
            elif cls_id == 5 or cls_id==8:
                inds_low = cls_dets[:, 4] > 0.1
            else:
                inds_low = cls_dets[:, 4] > 0.2

            inds_high = cls_dets[:, 4] < self.opt.conf_thres
            inds_second = np.logical_and(inds_low, inds_high)
            cls_dets_second = cls_dets[inds_second]
            cls_id_feature_second = cls_id_feats[cls_id][inds_second]

            cls_dets = cls_dets[remain_inds]
            cls_id_feature = cls_id_feats[cls_id][remain_inds]

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

                center_pred, size_curr = MCTrack.tlwh_to_center(track.tlwh)

                pred_off = matching.reid_motion_lost_det(
                    track, self.past_id_feature, self.past_reg, h_out, w_out, height, width
                )

                dist = np.sqrt(np.sum((center_pred - pred_off) ** 2))
                if dist <= 3                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   :
                    if len(cls_detects) > 0:
                        iou_dist_1 = 1 - matching.iou_distance([track], cls_detects)
                        min_dist_1 = iou_dist_1.max()
                    else:
                        min_dist_1 = 0

                    if len(cls_detects_second) > 0:
                        iou_dist_2 = 1 - matching.iou_distance([track], cls_detects_second)
                        min_dist_2 = iou_dist_2.max()
                    else:
                        min_dist_2 = 0

                    if min_dist_1 <= 2 and min_dist_2 <= 2:  # IOU < 0.2
                        # print(dist, min_dist_1, min_dist_2)
                        track.update_retrack(track.tlwh, self.frame_id)
                        activated_tracks_dict[cls_id].append(track)

                    else:
                        track.mark_lost()
                        lost_tracks_dict[cls_id].append(track)
                else:
                    track.mark_lost()
                    lost_tracks_dict[cls_id].append(track)

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

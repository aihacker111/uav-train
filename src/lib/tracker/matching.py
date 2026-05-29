import lap
import numpy as np
import scipy
from cython_bbox import bbox_overlaps as bbox_ious
from scipy.spatial.distance import cdist
from lib.tracking_utils import kalman_filter
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cdist
import cv2
import matplotlib.pyplot as plt


def merge_matches(m1, m2, shape):
    O, P, Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1 * M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b

# def greedy_assignment(dist):
#   matched_indices = []
#   if dist.shape[1] == 0:
#     return np.array(matched_indices, np.int32).reshape(-1, 2)
#   for i in range(dist.shape[0]):
#     j = dist[i].argmin()
#     if dist[i][j] < 1e16:
#       dist[:, j] = 1e18
#       matched_indices.append([i, j])
#   return np.array(matched_indices, np.int32).reshape(-1, 2)

def greedy_assignment(dist, thresh=0.7):
    """
    Perform greedy bipartite matching on a distance matrix.

    Args:
        dist (np.ndarray): 2D distance matrix of shape (n_rows, n_cols).
        thresh (float): Maximum distance threshold for valid matches.

    Returns:
        matches (np.ndarray): Array of shape (N, 2) with matched [row_idx, col_idx].
        unmatched_a (np.ndarray): Array of unmatched row indices.
        unmatched_b (np.ndarray): Array of unmatched column indices.
    """
    # Handle empty matrix
    if dist.size == 0:
        return (np.empty((0, 2), dtype=np.int32),
                np.arange(dist.shape[0], dtype=np.int32),
                np.arange(dist.shape[1], dtype=np.int32))

    # Initialize arrays
    matched_indices = []
    dist = dist.copy()  # Avoid modifying input
    n_rows, n_cols = dist.shape

    # Track used columns
    used_cols = np.zeros(n_cols, dtype=bool)

    # Greedy matching
    for i in range(n_rows):
        if np.all(dist[i] >= thresh):  # Skip if no valid matches
            continue
        j = np.argmin(dist[i])
        if dist[i][j] < thresh and not used_cols[j]:
            matched_indices.append([i, j])
            used_cols[j] = True
            dist[:, j] = np.inf  # Prevent reusing column

    # Convert matches to NumPy array
    matches = np.array(matched_indices, dtype=np.int32).reshape(-1, 2) if matched_indices else np.empty((0, 2), dtype=np.int32)

    # Compute unmatched indices
    matched_rows = matches[:, 0] if matches.size > 0 else np.array([], dtype=np.int32)
    matched_cols = matches[:, 1] if matches.size > 0 else np.array([], dtype=np.int32)
    unmatched_a = np.setdiff1d(np.arange(n_rows), matched_rows)
    unmatched_b = np.setdiff1d(np.arange(n_cols), matched_cols)

    return matches, unmatched_a, unmatched_b

def linear_assignment(cost_matrix, thresh):
    """
    :param cost_matrix:
    :param thresh:
    :return:
    """
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), \
               tuple(range(cost_matrix.shape[0])), \
               tuple(range(cost_matrix.shape[1]))

    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)

    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])

    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)

    return matches, unmatched_a, unmatched_b


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float64)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float64),
        np.ascontiguousarray(btlbrs, dtype=np.float64)
    )

    return ious


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
            len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:

        # atlbrs = np.array([
        #     (track.curr_tlwh[0],
        #      track.curr_tlwh[1],
        #      track.curr_tlwh[0] + track.curr_tlwh[2],
        #      track.curr_tlwh[1] + track.curr_tlwh[3]) if track.state == 1 else
        #     (track.tlbr[0], track.tlbr[1], track.tlbr[2], track.tlbr[3])
        #     for track in atracks
        # ])

        atlbrs = [track.tlbr for track in atracks]
        # atlwh_pre = [track.tlwh for track in atracks]
        # atlbrs = [[t, l, t + w, l + h] for t, l, w, h in [track.curr_tlwh for track in atracks]]
        btlbrs = [track.tlbr for track in btracks]

    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

def iou_distance_tracked(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
            len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        # atlbrs = [track.tlbr for track in atracks]
        # atlwh_pre = [track.tlwh for track in atracks]
        atlbrs = [[t, l, t + w, l + h] for t, l, w, h in [track.curr_tlwh for track in atracks]]
        btlbrs = [track.tlbr for track in btracks]

    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

def map2orig(dets, h_out, w_out, h_orig, w_orig, num_classes=0):
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

    cx, cy =  dets[:,0],  dets[:,1]

    if pad_type == 'pad_x':
        cx = (cx - pad_1) / new_shape[0] * w_orig  # x center
        cy = cy / h_out * h_orig  # y center
    else:  # 'pad_y'
        cx = cx / w_out * w_orig  # x center
        cy = (cy - pad_1) / new_shape[1] * h_orig  # y center

    dets_center = np.stack((cx, cy), axis=1)

    return dets_center


def return_position(tensor,regs,
                    h_out, w_out, height, width):
    n,h,w = tensor.size()
    max_values, max_indices = torch.max(tensor.view(n, -1), dim=1)

    cycx_coords = torch.stack((max_indices // w, max_indices % w), dim=1).float()

    if regs is not None:
        regs = regs[0].view(2, -1)
        extracted_regs = torch.gather(regs, 1, max_indices.unsqueeze(0).expand(2, -1)).t()
        cy = cycx_coords[:, 0] + extracted_regs[:, 1]
        cx = cycx_coords[:, 1] + extracted_regs[:, 0]
    else:
        cy = cycx_coords[:, 0]
        cx = cycx_coords[:, 1]

    dets_center = torch.stack((cx, cy), dim=1)

    dets_center = map2orig(dets_center, h_out, w_out, height, width)

    return dets_center


def reid_attention(prev_feat, curr_feat_map, temperature=0.07):
    n, c = prev_feat.shape
    _, h, w = curr_feat_map.shape
    curr_feat_flat = curr_feat_map.view(c, -1).transpose(0, 1)  # [H*W, 128]
    curr_feat_flat = F.normalize(curr_feat_flat, dim=1)
    sim = torch.matmul(prev_feat, curr_feat_flat.T)
    attn = torch.softmax(sim / temperature, dim=-1).view(n, h, w)
    #
    # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    # # 在第一个子图中显示第一张图片
    # axs[0].imshow(xx[0].cpu().numpy())
    # axs[0].axis('off')
    # axs[0].set_title('Image 1')
    #
    # # 在第二个子图中显示第二张图片
    # axs[1].imshow(attn[0].cpu().numpy())
    # axs[1].axis('off')
    # axs[1].set_title('Image 2')
    #
    # plt.tight_layout()  # 自动调整子图布局
    # plt.show()

    return attn

def reid_motion_vis(img, tracks, dets, past_id_feature, curr_id_feature, past_reg, curr_reg,
                h_out, w_out, height, width, save_frame):
    cost_matrix = np.zeros((len(tracks), len(dets)), dtype=np.float64)

    save_path = '/media/jianbo/ioe/Lun_6/vis/reid_off_curr/' + str(save_frame) + '.jpg'
    img_vis = img.copy()
    # pre_img_vis = pre_img_0.copy()
    img_vis_off = img.copy()

    if cost_matrix.size == 0:
        for i, det in enumerate(dets):
            x1, y1, x2, y2 = map(int, det.tlbr)
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(img_vis, f'{i}', (x1, y2 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            cv2.circle(img_vis, (cx, cy), radius=4, color=(0, 255, 255), thickness=-1)
        fig, axs = plt.subplots(1, 2, figsize=(20, 10))

        axs[0].imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
        axs[0].set_title("Tracks and Detections")
        axs[0].axis('off')

        axs[1].imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
        axs[1].set_title("track_off")
        axs[1].axis('off')

        # 保存合并图像
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        return cost_matrix

    else:

        track_centers = np.array([
            (track.curr_tlwh[0] + track.curr_tlwh[2] / 2.0,
             track.curr_tlwh[1] + track.curr_tlwh[3] / 2.0) if track.state == 1 else
            ((track.tlbr[0] + track.tlbr[2]) / 2.0,
             (track.tlbr[1] + track.tlbr[3]) / 2.0)
            for track in tracks
        ])

        for track in tracks:
            x1, y1, x2, y2 = map(int, track.tlbr)
            # x1, y1, x2, y2 = map(int, track.curr_tlwh)
            # x2, y2 = x1+x2, y1+y2
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0,0,255), 2)
            cv2.putText(img_vis, f'{track.track_id}', (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            cv2.circle(img_vis, (cx, cy), radius=4, color=(0,0,255), thickness=-1)

        # for i, track in enumerate(dets):
        #     x1, y1, x2, y2 = map(int, track.tlbr)
        #     cv2.rectangle(img_vis, (x1, y1), (x2, y2), (255,0,0), 2)
        #     cv2.putText(img_vis, f'{i}', (x1, y1 - 5),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
        #     cx = (x1 + x2) // 2
        #     cy = (y1 + y2) // 2
        #     cv2.circle(img_vis, (cx, cy), radius=4, color=(255,0,0), thickness=-1)

        # ==== 2. 绘制 Detections（绿框） ====
        # for i, det in enumerate(dets):
        #     x1, y1, x2, y2 = map(int, det.tlbr)
        #     cv2.rectangle(img_vis_off, (x1, y1), (x2, y2), (0, 255, 255), 2)
        #     cv2.putText(img_vis_off, f'{i}', (x1, y2 + 15),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        #     cx = (x1 + x2) // 2
        #     cy = (y1 + y2) // 2
        #     cv2.circle(img_vis_off, (cx, cy), radius=4, color=(0, 255, 255), thickness=-1)

        tracks_id = torch.from_numpy(np.array([track.curr_feat for track in tracks])).to(past_id_feature.device)
        dets_id = torch.from_numpy(np.array([det.curr_feat for det in dets])).to(past_id_feature.device)

        past_feature_map = reid_attention(tracks_id, curr_id_feature[0])
        curr_feature_map = reid_attention(dets_id, past_id_feature[0])

        past_feature_past = reid_attention(tracks_id, past_id_feature[0])
        curr_feature_curr = reid_attention(dets_id, curr_id_feature[0])

        # torch.einsum('ni,ijk->njk', tracks_id, curr_id_feature[0]).sigmoid_()) # t-1--t(hm)
        # torch.einsum('ni,ijk->njk', dets_id, past_id_feature[0]).sigmoid_())   # t--t-1(hm)

        past_to_curr = return_position(past_feature_map, curr_reg, h_out, w_out, height, width)
        curr_to_past = return_position(curr_feature_map, past_reg, h_out, w_out, height, width)

        past_to_past = return_position(past_feature_past, curr_reg, h_out, w_out, height, width)
        curr_to_curr = return_position(curr_feature_curr, past_reg, h_out, w_out, height, width)



        for i, track in enumerate(tracks):
            cx, cy = past_to_curr[i][0], past_to_past[i][1]
            cv2.circle(img_vis, (int(cx), int(cy)), radius=4, color=(255, 222, 0), thickness=-1)
            cv2.putText(img_vis, f'{track.track_id}', (int(cx), int(cy) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 222, 0), 2)

        # for i, track in enumerate(dets):
        #     cx, cy = curr_to_past[i][0],  curr_to_curr[i][1]
        #     cv2.circle(img_vis, (int(cx), int(cy)), radius=4, color=(100, 222, 200), thickness=-1)
        #     cv2.putText(img_vis, f'{i}', (int(cx), int(cy) - 5),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 222, 100), 2)

        fig, axs = plt.subplots(1, 2, figsize=(20, 10))

        # 左边：框图（BGR -> RGB）
        axs[0].imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
        axs[0].set_title("Tracks and Detections")
        axs[0].axis('off')

        axs[1].imshow(cv2.cvtColor(img_vis_off, cv2.COLOR_BGR2RGB))
        axs[1].set_title("track_off")
        axs[1].axis('off')

        # 保存合并图像
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        # track_centers = np.array([[track.curr_tlwh[0] + (track.curr_tlwh[2]) / 2.0,  # cx
        #                            track.curr_tlwh[1] + (track.curr_tlwh[3]) / 2.0]  # cy
        #                           for track in tracks], dtype=np.float32)

        track_centers = np.array([[(track.tlbr[0] + track.tlbr[2]) / 2.0,  # cx
                                 (track.tlbr[1] + track.tlbr[3]) / 2.0] # cy
                                    for track in tracks], dtype=np.float32)

        # 获取 detections 的中心点
        det_centers = np.array([[(det.tlbr[0] + det.tlbr[2]) / 2.0,  # cx
                                 (det.tlbr[1] + det.tlbr[3]) / 2.0]  # cy
                                for det in dets], dtype=np.float32)

        dist_1 = np.linalg.norm(det_centers[None, :,  :] - past_to_curr[:, None, :], axis=2)
        dist_2 = np.linalg.norm(curr_to_past[None, :,  :] - track_centers[:, None, :], axis=2)
        # dist = np.maximum(dist_1, dist_2)


        sigma = 5  # 可根据场景设置
        off_matrix = np.exp(- (dist_1 + dist_2) / (2 * sigma ** 2))
        off_matrix = 1 - off_matrix

        cost_matrix = off_matrix

        return cost_matrix


def reid_motion_unconfirmed(tracks, dets, id_feature, reg,
                h_out, w_out, height, width):
    cost_matrix = np.zeros((len(tracks), len(dets)), dtype=np.float64)
    if cost_matrix.size == 0:
        return cost_matrix
    else:

        tracks_id = torch.from_numpy(np.array([track.curr_feat if track.state == 1 else track.smooth_feat
                                              for track in tracks])).to(id_feature[-1].device)
        dets_id = torch.from_numpy(np.array([det.curr_feat for det in dets])).to(id_feature[-1].device)

        past_feature_map = reid_attention(tracks_id, id_feature[-1][0])
        curr_feature_map = reid_attention(dets_id, id_feature[-2][0])

        past_to_curr = return_position(past_feature_map, reg[-1], h_out, w_out, height, width)
        curr_to_past = return_position(curr_feature_map, reg[-2], h_out, w_out, height, width)

        track_centers = np.array([[(track.tlbr[0] + track.tlbr[2]) / 2.0,  # cx
                                    (track.tlbr[1] + track.tlbr[3]) / 2.0]  # cy
                                  for track in tracks], dtype=np.float32)


        # 获取 detections 的中心点
        det_centers = np.array([[(det.tlbr[0] + det.tlbr[2]) / 2.0,  # cx
                                 (det.tlbr[1] + det.tlbr[3]) / 2.0]  # cy
                                for det in dets], dtype=np.float32)


        dist_1 = np.linalg.norm(det_centers[None, :,  :] - past_to_curr[:, None, :], axis=2)

        dist_2 = np.linalg.norm(curr_to_past[None, :,  :] - track_centers[:, None, :], axis=2)


        sigma = 5  # 可根据场景设置
        off_matrix = np.exp(- (dist_1 + dist_2) / (2 * sigma ** 2))
        off_matrix = 1 - off_matrix

        cost_matrix = off_matrix

        return cost_matrix

def reid_motion(tracks, dets, id_feature, reg,
                h_out, w_out, height, width):
    cost_matrix = np.zeros((len(tracks), len(dets)), dtype=np.float64)
    if cost_matrix.size == 0:
        return cost_matrix
    else:

        tracks_id = torch.from_numpy(np.array([track.curr_feat if track.state == 1 else track.smooth_feat
                                               for track in tracks])).to(id_feature[-1].device)
        dets_id = torch.from_numpy(np.array([det.curr_feat for det in dets])).to(id_feature[-1].device)

        past_feature_map = reid_attention(tracks_id, id_feature[-1][0])
        curr_feature_map = reid_attention(dets_id, id_feature[-2][0])

        past_to_curr = return_position(past_feature_map, reg[-1], h_out, w_out, height, width)
        curr_to_past = return_position(curr_feature_map, reg[-2], h_out, w_out, height, width)

        track_centers = np.array([
            (track.curr_tlwh[0] + track.curr_tlwh[2] / 2.0,
             track.curr_tlwh[1] + track.curr_tlwh[3] / 2.0) if track.state == 1 else
            ((track.tlbr[0] + track.tlbr[2]) / 2.0,
             (track.tlbr[1] + track.tlbr[3]) / 2.0)
            for track in tracks
        ])

        det_centers = np.array([[(det.tlbr[0] + det.tlbr[2]) / 2.0,  # cx
                                 (det.tlbr[1] + det.tlbr[3]) / 2.0]  # cy
                                for det in dets], dtype=np.float32)


        dist_1 = np.linalg.norm(det_centers[None, :,  :] - past_to_curr[:, None, :], axis=2)
        dist_2 = np.linalg.norm(curr_to_past[None, :,  :] - track_centers[:, None, :], axis=2)

        sigma = 5  # 可根据场景设置
        off_matrix = np.exp(- (dist_1 + dist_2) / (2 * sigma ** 2))
        off_matrix = 1 - off_matrix

        cost_matrix = off_matrix

        return cost_matrix



def embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float64)
    if cost_matrix.size == 0:
        return cost_matrix

    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float64)
    # for i, track in enumerate(tracks):
    # cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float64)

    # 默认计算两个特征向量之间的夹角余弦
    # Nomalized features
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))

    return cost_matrix

def fuse_score_three(iou_cost_matrix, id_sim_matrix, detections):
    if iou_cost_matrix.size == 0:
        return iou_cost_matrix
    iou_sim = 1 - iou_cost_matrix
    id_sim = 1 - id_sim_matrix
    # det_scores = np.array([det.score for det in detections])
    # det_scores = np.expand_dims(det_scores, axis=0).repeat(iou_cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * id_sim
    # fuse_sim = iou_sim * det_scores
    # fuse_sim = iou_sim * id_sim
    # fuse_sim = id_sim
    fuse_cost = 1 - fuse_sim
    return fuse_cost


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    """
    :param kf:
    :param cost_matrix:
    :param tracks:
    :param detections:
    :param only_position:
    :return:
    """
    if cost_matrix.size == 0:
        return cost_matrix

    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])

    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf

    return cost_matrix


def fuse_motion(kf,
                cost_matrix,
                tracks,
                detections,
                only_position=False,
                lambda_=0.98):
    """
    :param kf:
    :param cost_matrix:
    :param tracks:
    :param detections:
    :param only_position:
    :param lambda_:
    :return:
    """
    if cost_matrix.size == 0:
        return cost_matrix

    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])

    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(track.mean,
                                             track.covariance,
                                             measurements,
                                             only_position,
                                             metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance

    return cost_matrix

def reid_motion_lost_det(track, id_feature, reg,
                         h_out, w_out, height, width):
    track_id = torch.from_numpy(track.curr_feat)[None, :].to(id_feature[-1].device)
    past_feature_map = reid_attention(track_id, id_feature[-1][0])
    past_to_curr = return_position(past_feature_map, reg[-1], h_out, w_out, height, width)
    return past_to_curr

def associate(cost, match_thr):
    # Initialization
    matches = []

    # Run
    if cost.shape[0] > 0 and cost.shape[1] > 0:
        # Get index for minimum similarity
        min_ddx = np.argmin(cost, axis=1)
        min_tdx = np.argmin(cost, axis=0)

        # Match tracks with detections
        for tdx, ddx in enumerate(min_ddx):
            if min_tdx[ddx] == tdx and cost[tdx, ddx] < match_thr:
                matches.append([tdx, ddx])

    return matches

def iterative_assigment(cost, tracks, dets,match_thr):

    # Match
    matches = []

    while True:
        # Match tracks with detections
        matches_ = associate(cost, match_thr)
        # Check (if there are no more matchable pairs)
        if len(matches_) == 0:
            break
        # Append
        matches += matches_
        # Update cost matrix
        for t, d in matches:
            cost[t, :] = 1.
            cost[:, d] = 1.
    # Find indices of unmatched tracks and detections
    m_tracks = [t for t, _ in matches]
    u_tracks = [t for t in range(len(tracks)) if t not in m_tracks]
    m_dets = [d for _, d in matches]
    u_dets = [d for d in range(len(dets)) if d not in m_dets]

    return matches, u_tracks, u_dets
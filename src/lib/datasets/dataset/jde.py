import glob
import math
import os
import os.path as osp
import random
import copy
import time
import warnings

import cv2
# import json
import numpy as np
import torch

from collections import OrderedDict, defaultdict
# from torch.utils.data import Dataset
# from torchvision.transforms import transforms as T
# from cython_bbox import bbox_overlaps as bbox_ious
# from lib.opts import opts
from lib.utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from lib.utils.utils import xyxy2xywh, generate_anchors, xywh2xyxy, encode_delta
from lib.tracker.multitracker import id2cls


# for inference
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def _normalize(img, use_imagenet=False):
    img = img / 255.0
    if use_imagenet:
        img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
    return img


class LoadImages:
    def __init__(self, path, img_size=(1088, 608), use_imagenet_norm=False):
        """
        :param path:
        :param img_size:
        :param use_imagenet_norm: if True, apply ImageNet mean/std normalization (for ViT backbone)
        """
        self.use_imagenet_norm = use_imagenet_norm
        self.frame_rate = 10  # no actual meaning here

        if type(path) == str:
            if os.path.isdir(path):
                image_format = ['.jpg', '.jpeg', '.png', '.tif']
                self.files = sorted(glob.glob('%s/*.*' % path))
                self.files = list(filter(lambda x: os.path.splitext(x)[
                                                       1].lower() in image_format, self.files))
            elif os.path.isfile(path):
                self.files = [path]
            else:
                raise FileNotFoundError(f'[LoadImages] Path not found: {path}')
        elif type(path) == list:
            self.files = path
        else:
            self.files = []

        self.nF = len(self.files)  # number of image files
        self.width = img_size[0]
        self.height = img_size[1]
        self.count = 0

        assert self.nF > 0, 'No images found in ' + path

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1

        if self.count == self.nF:
            raise StopIteration

        img_path = self.files[self.count]

        # Read image
        img_0 = cv2.imread(img_path)  # BGR
        assert img_0 is not None, 'Failed to load ' + img_path

        # Padded resize
        img, _, _, _ = letterbox(img_0, height=self.height, width=self.width)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img = _normalize(img, self.use_imagenet_norm)

        # cv2.imwrite(img_path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
        return img_path, img, img_0

    def __getitem__(self, idx):
        idx = idx % self.nF
        img_path = self.files[idx]

        # Read image
        img_0 = cv2.imread(img_path)  # BGR
        assert img_0 is not None, 'Failed to load ' + img_path

        # Padded resize
        img, _, _, _ = letterbox(img_0, height=self.height, width=self.width)

        # Normalize RGB: BGR -> RGB and H×W×C -> C×H×W
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img = _normalize(img, self.use_imagenet_norm)

        return img_path, img, img_0

    def __len__(self):
        return self.nF  # number of files


class LoadVideo:  # for inference
    def __init__(self,
                 path,
                 img_size=(1088, 608),
                 use_imagenet_norm=False):
        """
        :param path:
        :param img_size:
        :param use_imagenet_norm: if True, apply ImageNet mean/std normalization (for ViT backbone)
        """
        self.use_imagenet_norm = use_imagenet_norm
        self.cap = cv2.VideoCapture(path)
        self.frame_rate = int(round(self.cap.get(cv2.CAP_PROP_FPS)))
        self.vw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.vh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.vn = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.width = img_size[0]
        self.height = img_size[1]
        self.count = 0

        self.w, self.h = 1920, 1080  # 设置(输出的分辨率)
        print('Lenth of the video: {:d} frames'.format(self.vn))

    def get_size(self, vw, vh, dw, dh):
        wa, ha = float(dw) / vw, float(dh) / vh
        a = min(wa, ha)
        return int(vw * a), int(vh * a)

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if self.count == len(self):
            raise StopIteration

        # Read image
        res, img_0 = self.cap.read()  # BGR
        assert img_0 is not None, 'Failed to load frame {:d}'.format(self.count)
        img_0 = cv2.resize(img_0, (self.w, self.h))

        # Padded resize
        img, _, _, _ = letterbox(img_0, height=self.height, width=self.width)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB and HWC->CHW
        img = np.ascontiguousarray(img, dtype=np.float32)
        img = _normalize(img, self.use_imagenet_norm)

        # save letterbox image
        # cv2.imwrite(img_path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])
        return self.count, img, img_0

    def __len__(self):
        return self.vn  # number of files


class LoadImagesAndLabels:  # for training
    def __init__(self,
                 path,
                 img_size=(1088, 608),
                 augment=False,
                 transforms=None,
                 pil_transform=None):
        with open(path, 'r') as file:
            self.img_files = file.readlines()
            self.img_files = [x.replace('\n', '') for x in self.img_files]
            self.img_files = list(filter(lambda x: len(x) > 0, self.img_files))

        self.label_files = [x.replace('images', 'labels_with_ids')
                            .replace('.png', '.txt')
                            .replace('.jpg', '.txt')
                            for x in self.img_files]

        self.nF = len(self.img_files)  # number of image files

        self.width = img_size[0]
        self.height = img_size[1]

        self.augment = augment
        self.transforms = transforms
        self.pil_transform = pil_transform  # PIL-based pre-letterbox transforms

    def __getitem__(self, files_index):
        img_path = self.img_files[files_index]
        label_path = self.label_files[files_index]
        return self.get_data(img_path, label_path)

    def get_data(self, img_path, label_path, width=None, height=None):
        if height is None or width is None:
            height = self.height
            width = self.width

        img = cv2.imread(img_path)  # BGR
        if img is None:
            raise ValueError('File corrupt {}'.format(img_path))

        # Load labels early — needed before PIL transforms
        if os.path.isfile(label_path):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                labels_0 = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 6)
        else:
            labels_0 = np.zeros((0, 6), dtype=np.float32)

        # HSV saturation/value jitter
        if self.augment:
            fraction = 0.50
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            S = img_hsv[:, :, 1].astype(np.float32)
            V = img_hsv[:, :, 2].astype(np.float32)
            a = (random.random() * 2 - 1) * fraction + 1
            S *= a
            if a > 1:
                np.clip(S, a_min=0, a_max=255, out=S)
            a = (random.random() * 2 - 1) * fraction + 1
            V *= a
            if a > 1:
                np.clip(V, a_min=0, a_max=255, out=V)
            img_hsv[:, :, 1] = S.astype(np.uint8)
            img_hsv[:, :, 2] = V.astype(np.uint8)
            cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)

        # PIL-based spatial augmentations (pre-letterbox, on original resolution)
        if self.augment and self.pil_transform is not None:
            import PIL.Image
            h0, w0 = img.shape[:2]
            img_pil = PIL.Image.fromarray(img[:, :, ::-1])  # BGR → RGB PIL

            if len(labels_0) > 0:
                boxes_xyxy = np.stack([
                    w0 * (labels_0[:, 2] - labels_0[:, 4] / 2),
                    h0 * (labels_0[:, 3] - labels_0[:, 5] / 2),
                    w0 * (labels_0[:, 2] + labels_0[:, 4] / 2),
                    h0 * (labels_0[:, 3] + labels_0[:, 5] / 2),
                ], axis=1).astype(np.float32)
                target = {
                    'boxes': torch.tensor(boxes_xyxy),
                    'labels': torch.tensor(labels_0[:, :2]),  # cls_id, track_id
                    'size': torch.tensor([h0, w0]),
                }
            else:
                target = {
                    'boxes': torch.zeros((0, 4), dtype=torch.float32),
                    'labels': torch.zeros((0, 2), dtype=torch.float32),
                    'size': torch.tensor([h0, w0]),
                }

            img_pil, target = self.pil_transform(img_pil, target)
            img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            h0, w0 = img.shape[:2]

            if len(target['boxes']) > 0:
                boxes = target['boxes'].numpy()
                cls_trk = target['labels'].numpy()
                labels_0 = np.zeros((len(boxes), 6), dtype=np.float32)
                labels_0[:, :2] = cls_trk
                labels_0[:, 2] = (boxes[:, 0] + boxes[:, 2]) / 2.0 / w0   # cx norm
                labels_0[:, 3] = (boxes[:, 1] + boxes[:, 3]) / 2.0 / h0   # cy norm
                labels_0[:, 4] = (boxes[:, 2] - boxes[:, 0]) / w0          # w norm
                labels_0[:, 5] = (boxes[:, 3] - boxes[:, 1]) / h0          # h norm
            else:
                labels_0 = np.zeros((0, 6), dtype=np.float32)

        h, w = img.shape[:2]
        img, ratio, pad_w, pad_h = letterbox(img, height=height, width=width)

        # Convert labels to pixel xyxy relative to letterboxed image
        if len(labels_0) > 0:
            labels = labels_0.copy()
            labels[:, 2] = ratio * w * (labels_0[:, 2] - labels_0[:, 4] / 2) + pad_w  # x1
            labels[:, 3] = ratio * h * (labels_0[:, 3] - labels_0[:, 5] / 2) + pad_h  # y1
            labels[:, 4] = ratio * w * (labels_0[:, 2] + labels_0[:, 4] / 2) + pad_w  # x2
            labels[:, 5] = ratio * h * (labels_0[:, 3] + labels_0[:, 5] / 2) + pad_h  # y2
        else:
            labels = np.array([])

        # Random affine (rotation, scale, translate) on fixed-size image
        if self.augment:
            img, labels, M = random_affine(img, labels,
                                           degrees=(-5, 5),
                                           translate=(0.10, 0.10),
                                           scale=(0.50, 1.20))

        num_labels = len(labels)
        if num_labels > 0:
            labels[:, 2:6] = xyxy2xywh(labels[:, 2:6].copy())
            labels[:, 2] /= width
            labels[:, 3] /= height
            labels[:, 4] /= width
            labels[:, 5] /= height

        # Horizontal flip only when NOT using pil_transform (which already includes it)
        if self.augment and self.pil_transform is None:
            if random.random() > 0.5:
                img = np.fliplr(img)
                if num_labels > 0:
                    labels[:, 2] = 1 - labels[:, 2]

        img = np.ascontiguousarray(img[:, :, ::-1])  # BGR → RGB

        if self.transforms is not None:
            img = self.transforms(img)

        return img, labels, img_path, (h, w)

    def __len__(self):
        return self.nF  # number of batches


def letterbox(img,
              height=608,
              width=1088,
              color=(127.5, 127.5, 127.5)):
    """
    resize a rectangular image to a padded rectangular
    :param img:
    :param height:
    :param width:
    :param color:
    :return:
    """
    shape = img.shape[:2]  # shape = [height, width]
    ratio = min(float(height) / shape[0], float(width) / shape[1])

    # new_shape = [width, height]
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))
    dw = (width - new_shape[0]) * 0.5  # width padding
    dh = (height - new_shape[1]) * 0.5  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)

    # resized, no border
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)  # padded rectangular
    return img, ratio, dw, dh


def random_affine(img, targets=None,
                  degrees=(-10, 10),
                  translate=(.1, .1),
                  scale=(.9, 1.1),
                  shear=(-2, 2),
                  borderValue=(127.5, 127.5, 127.5)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4

    border = 0  # width of added border (optional)
    height = img.shape[0]
    width = img.shape[1]

    # Rotation and Scale
    R = np.eye(3)
    a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
    # a += random.choice([-180, -90, 0, 90])  # 90deg rotations added to small rotations
    s = random.random() * (scale[1] - scale[0]) + scale[0]
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(
        img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = (random.random() * 2 - 1) * translate[0] * \
              img.shape[0] + border  # x translation (pixels)
    T[1, 2] = (random.random() * 2 - 1) * translate[1] * \
              img.shape[1] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan((random.random() * (shear[1] - shear[0]) +
                        shear[0]) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan((random.random() * (shear[1] - shear[0]) +
                        shear[0]) * math.pi / 180)  # y shear (deg)

    M = S @ T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
    imw = cv2.warpPerspective(img, M, dsize=(width, height), flags=cv2.INTER_LINEAR,
                              borderValue=borderValue)  # BGR order borderValue

    # Return warped points also
    if targets is not None:
        if len(targets) > 0:
            n = targets.shape[0]
            points = targets[:, 2:6].copy()
            area0 = (points[:, 2] - points[:, 0]) * \
                    (points[:, 3] - points[:, 1])

            # warp points
            xy = np.ones((n * 4, 3))
            xy[:, :2] = points[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
                n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = (xy @ M.T)[:, :2].reshape(n, 8)

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate(
                (x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # apply angle-based reduction
            radians = a * math.pi / 180
            reduction = max(abs(math.sin(radians)),
                            abs(math.cos(radians))) ** 0.5
            x = (xy[:, 2] + xy[:, 0]) / 2
            y = (xy[:, 3] + xy[:, 1]) / 2
            w = (xy[:, 2] - xy[:, 0]) * reduction
            h = (xy[:, 3] - xy[:, 1]) * reduction
            xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

            # reject warped points outside of image
            np.clip(xy[:, 0], 0, width, out=xy[:, 0])
            np.clip(xy[:, 2], 0, width, out=xy[:, 2])
            np.clip(xy[:, 1], 0, height, out=xy[:, 1])
            np.clip(xy[:, 3], 0, height, out=xy[:, 3])
            w = xy[:, 2] - xy[:, 0]
            h = xy[:, 3] - xy[:, 1]
            area = w * h
            ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
            i = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.1) & (ar < 10)

            targets = targets[i]
            targets[:, 2:6] = xy[i]

        return imw, targets, M
    else:
        return imw


def collate_fn(batch):
    imgs, labels, paths, sizes = zip(*batch)
    batch_size = len(labels)
    imgs = torch.stack(imgs, 0)
    max_box_len = max([l.shape[0] for l in labels])
    labels = [torch.from_numpy(l) for l in labels]
    filled_labels = torch.zeros(batch_size, max_box_len, 6)
    labels_len = torch.zeros(batch_size)

    for i in range(batch_size):
        isize = labels[i].shape[0]
        if len(labels[i]) > 0:
            filled_labels[i, :isize, :] = labels[i]
        labels_len[i] = isize

    return imgs, filled_labels, paths, sizes, labels_len.unsqueeze(1)



# ----------

class JointDataset(LoadImagesAndLabels):  # for training
    """
    joint detection and embedding dataset
    """
    mean = None
    std = None

    def __init__(self,
                 opt,
                 root,
                 paths,
                 img_size=(1088, 608),
                 augment=False,
                 transforms=None,
                 pil_transform=None):
        self.opt = opt
        # dataset_names = paths.keys()
        self.img_files = OrderedDict()
        self.label_files = OrderedDict()
        self.tid_num = OrderedDict()
        self.tid_start_index = OrderedDict()
        self.num_classes = len(opt.reid_cls_ids.split(','))  # C5: car, bicycle, person, cyclist, tricycle

        # make sure img_size equal to opt.input_wh
        if opt.input_wh[0] != img_size[0] or opt.input_wh[1] != img_size[1]:
            opt.input_wh[0], opt.input_wh[1] = img_size[0], img_size[1]

        # default input width and height
        self.default_input_wh = opt.input_wh

        # net input width and height
        self.width = self.default_input_wh[0]
        self.height = self.default_input_wh[1]

        # ----- generate img and label file path lists
        for ds, path in paths.items():
            with open(path, 'r') as file:
                self.img_files[ds] = file.readlines()
                self.img_files[ds] = [osp.join(root, x.strip()) for x in self.img_files[ds]]
                self.img_files[ds] = list(filter(lambda x: len(x) > 0, self.img_files[ds]))

            self.label_files[ds] = [x.replace('images', 'labels_with_ids')
                                    .replace('.png', '.txt')
                                    .replace('.jpg', '.txt')
                                    for x in self.img_files[ds]]

            print('Total {} image files in {} dataset.'.format(len(self.label_files[ds]), ds))

        if opt.id_weight > 0:  # If do ReID calculation
            # @even: for MCMOT training
            for ds, label_paths in self.label_files.items():  # 每个子数据集
                max_ids_dict = defaultdict(int)  # cls_id => max track id

                # 子数据集中每个label
                for lp in label_paths:
                    if not os.path.isfile(lp):
                        print('[Warning]: invalid label file {}.'.format(lp))
                        continue

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")

                        lb = np.loadtxt(lp)
                        if len(lb) < 1:  # 空标签文件
                            continue

                        lb = lb.reshape(-1, 6)
                        for item in lb:  # label中每一个item(检测目标)
                            if item[1] > max_ids_dict[int(item[0])]:  # item[0]: cls_id, item[1]: track id
                                max_ids_dict[int(item[0])] = item[1]

                # track id number
                self.tid_num[ds] = max_ids_dict  # 每个子数据集按照需要reid的cls_id组织成dict

            # @even: for MCMOT training
            self.tid_start_idx_of_cls_ids = defaultdict(dict)
            last_idx_dict = defaultdict(int)  # 从0开始
            for k, v in self.tid_num.items():  # 统计每一个子数据集
                for cls_id, id_num in v.items():  # 统计这个子数据集的每一个类别, v是一个max_ids_dict
                    self.tid_start_idx_of_cls_ids[k][cls_id] = last_idx_dict[cls_id]
                    last_idx_dict[cls_id] += id_num

            # @even: for MCMOT training
            self.nID_dict = defaultdict(int)
            for k, v in last_idx_dict.items():
                self.nID_dict[k] = int(v)  # 每个类别的tack ids数量

        self.nds = [len(x) for x in self.img_files.values()]
        self.cds = [sum(self.nds[:i]) for i in range(len(self.nds))]
        self.nF = sum(self.nds)
        self.max_objs = opt.K
        self.augment = augment
        self.transforms = transforms
        self.pil_transform = pil_transform

        print('dataset summary')
        print(self.tid_num)

        if opt.id_weight > 0:  # If do ReID calculation
            # print('total # identities:', self.nID)
            for k, v in self.nID_dict.items():
                print('Total {:d} IDs of {}'.format(v, id2cls[k]))

            # print('start index', self.tid_start_index)
            for k, v in self.tid_start_idx_of_cls_ids.items():
                for cls_id, start_idx in v.items():
                    print('Start index of dataset {} class {:d} is {:d}'
                          .format(k, int(cls_id), int(start_idx)))

    def __getitem__(self, idx):
        # 为子训练集计算起始index
        for i, c in enumerate(self.cds):
            if idx >= c:
                ds = list(self.label_files.keys())[i]
                start_index = c

        img_path = self.img_files[ds][idx - start_index]
        label_path = self.label_files[ds][idx - start_index]

        # Get image data and label
        imgs, labels, img_path, (input_h, input_w) = self.get_data(img_path, label_path)
        # print('input_h, input_w: %d %d' % (input_h, input_w))

        # 存在多个子训练集时, 为每个子训练集合(视频seq)计算正确的起始index
        # @even: for MCMOT training
        if self.opt.id_weight > 0:
            for i, _ in enumerate(labels):
                if labels[i, 1] > -1:
                    cls_id = int(labels[i][0])
                    start_idx = self.tid_start_idx_of_cls_ids[ds][cls_id]
                    labels[i, 1] += start_idx

        output_h = imgs.shape[1] // self.opt.down_ratio  # 向下取整除法
        output_w = imgs.shape[2] // self.opt.down_ratio
        # print('output_h, output_w: %d %d' % (output_h, output_w))

        # num_classes = self.num_classes

        # 图片中实际标注的目标数
        num_objs = labels.shape[0]

        # --- DETR-format targets (normalized cxcywh, kept before feature-map scaling)
        _is_hybrid = getattr(self.opt, 'task', '') == 'hybrid'
        if _is_hybrid:
            if num_objs > 0:
                _detr_boxes  = torch.from_numpy(labels[:, 2:6].copy()).float()
                _detr_labels = torch.from_numpy(labels[:, 0].copy()).long()
            else:
                _detr_boxes  = torch.zeros((0, 4), dtype=torch.float32)
                _detr_labels = torch.zeros((0,),   dtype=torch.long)

        # --- GT of detection
        hm = np.zeros((self.num_classes, output_h, output_w), dtype=np.float32)  # C×H×W: heat-map通道数即类别数
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs,), dtype=np.int64)  # K个object
        reg_mask = np.zeros((self.max_objs,), dtype=np.uint8)  # 只计算feature map有目标的像素的reg loss

        if self.opt.id_weight > 0:
            # --- GT of ReID
            ids = np.zeros((self.max_objs,), dtype=np.int64)  # 一张图最多检测并ReID K个目标, 都初始化id为0

            # @even: 每个目标类别都对应一组track ids
            cls_tr_ids = np.zeros((self.num_classes, output_h, output_w), dtype=np.int64)

            # @even, class id map: 每个(x, y)处的目标类别, 都初始化为-1
            cls_id_map = np.full((1, output_h, output_w), -1, dtype=np.int64)  # 1×H×W

        # Gauss function definition
        draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else draw_umich_gaussian

        # 遍历每一个ground truth检测目标
        for k in range(num_objs):  # 图片中实际的目标个数
            label = labels[k]

            # 计算bbox的经过网络的输出GT值
            #                       0        1        2       3
            bbox = label[2:]  # center_x, center_y, bbox_w, bbox_h

            # 检测目标的类别(索引从0开始, 0代表背景类别)
            cls_id = int(label[0])

            bbox[[0, 2]] = bbox[[0, 2]] * output_w
            bbox[[1, 3]] = bbox[[1, 3]] * output_h
            bbox[0] = np.clip(bbox[0], 0, output_w - 1)
            bbox[1] = np.clip(bbox[1], 0, output_h - 1)

            w, h = bbox[2], bbox[3]

            if h > 0 and w > 0:
                # heat-map radius
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))  # radius >= 0
                radius = self.opt.hm_gauss if self.opt.mse_loss else radius

                # bbox center coordinate
                ct = np.array([bbox[0], bbox[1]], dtype=np.float32)
                ct_int = ct.astype(np.int32)  # floor int

                # draw gauss weight for heat-map
                draw_gaussian(hm[cls_id], ct_int, radius)  # hm

                # --- GT of detection
                wh[k] = float(w), float(h)

                # 记录feature map上有目标的坐标索引
                ind[k] = ct_int[1] * output_w + ct_int[0]  # feature map index:y*w+x

                # offset regression
                reg[k] = ct - ct_int
                reg_mask[k] = 1

                # --- GT of ReID
                if self.opt.id_weight > 0:
                    # @even: 取output feature map的每个(y, x)处的目标类别
                    cls_id_map[0][ct_int[1], ct_int[0]] = cls_id  # 1×H×W

                    # @even: 记录该类别对应的track ids
                    cls_tr_ids[cls_id][ct_int[1]][ct_int[0]] = label[1] - 1  # track id从1开始的, 转换成从0开始

                    ids[k] = label[1] - 1  # 分类的idx: track id - 1

        if self.opt.id_weight > 0:
            ret = {'input': imgs,
                   'hm': hm,
                   'reg': reg,
                   'wh': wh,
                   'ind': ind,
                   'reg_mask': reg_mask,
                   'ids': ids,
                   'cls_id_map': cls_id_map,  # feature map上每个(x, y)处的目标类别id
                   'cls_tr_ids': cls_tr_ids}
        else:  # only for detection
            ret = {'input': imgs,
                   'hm': hm,
                   'reg': reg,
                   'wh': wh,
                   'ind': ind,
                   'reg_mask': reg_mask}

        # DETR-format targets — only emitted for hybrid task; collate_fn assembles into a list
        if _is_hybrid:
            ret['targets'] = {'boxes': _detr_boxes, 'labels': _detr_labels}

        return ret  # 返回一个字典(第一次见识这样的getitem)


def hybrid_collate_fn(batch):
    """
    Custom collate for hybrid training.  All keys except 'targets' are stacked
    normally; 'targets' (DETR format, variable number of GT per image) is kept
    as a plain Python list of dicts so the Hungarian matcher can iterate over it.
    """
    from torch.utils.data.dataloader import default_collate
    targets = [b['targets'] for b in batch]
    other   = [{k: v for k, v in b.items() if k != 'targets'} for b in batch]
    collated = default_collate(other)
    collated['targets'] = targets
    return collated

import glob
import math
import os
import os.path as osp
import random
import copy
import time
import warnings

import cv2
import numpy as np
import torch

from collections import OrderedDict, defaultdict
from lib.utils.utils import xyxy2xywh, generate_anchors, xywh2xyxy, encode_delta
from lib.tracker.multitracker import id2cls
from lib.datasets.augment import (
    MosaicAugmentor, photometric_distort,
    random_zoom_out, random_iou_crop, sanitize_boxes,
)

# ImageNet mean/std (matching EdgeCrafter's Normalize op)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# for inference
class LoadImages:
    def __init__(self, path, img_size=(1088, 608)):
        """
        :param path:
        :param img_size:
        """
        self.frame_rate = 10  # no actual meaning here

        if type(path) == str:
            if os.path.isdir(path):
                image_format = ['.jpg', '.jpeg', '.png', '.tif']
                self.files = sorted(glob.glob('%s/*.*' % path))
                self.files = list(filter(lambda x: os.path.splitext(x)[
                                                       1].lower() in image_format, self.files))
            elif os.path.isfile(path):
                self.files = [path]
        elif type(path) == list:
            self.files = path

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
        img /= 255.0

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
        img /= 255.0

        return img_path, img, img_0

    def __len__(self):
        return self.nF  # number of files


class LoadVideo:  # for inference
    def __init__(self,
                 path,
                 img_size=(1088, 608)):
        """
        :param path:
        :param img_size:
        """
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
        img /= 255.0

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
                 transforms=None):
        """
        :param path:
        :param img_size:
        :param augment:
        :param transforms:
        """
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

    def __getitem__(self, files_index):
        img_path = self.img_files[files_index]
        label_path = self.label_files[files_index]
        return self.get_data(img_path, label_path)

    def get_data(self, img_path, label_path, width=None, height=None):
        """
        图像数据格式转换, 增强; 标签格式化
        :param img_path:
        :param label_path:
        :param height:
        :param width:
        :return:
        """
        # 输入网络的图像分辨率
        if height is None or width is None:
            height = self.height
            width = self.width

        # 读取图片数据为numpy array格式, 3通道顺序为BGR
        img = cv2.imread(img_path)  # cv(numpy): BGR
        if img is None:
            raise ValueError('File corrupt {}'.format(img_path))

        augment_hsv = True
        if self.augment and augment_hsv:
            # SV augmentation by 50%
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

        h, w, _ = img.shape
        img, ratio, pad_w, pad_h = letterbox(img, height=height, width=width)  # resizing and padding

        # Load labels
        if os.path.isfile(label_path):
            with warnings.catch_warnings():  # No warnings for empty label file(txt)
                warnings.simplefilter("ignore")
                labels_0 = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 6)

                # reformat xywh to pixel xyxy(x1, y1, x2, y2) format
                labels = labels_0.copy()  # deep copy
                labels[:, 2] = ratio * w * (labels_0[:, 2] - labels_0[:, 4] / 2) + pad_w  # x1
                labels[:, 3] = ratio * h * (labels_0[:, 3] - labels_0[:, 5] / 2) + pad_h  # y1
                labels[:, 4] = ratio * w * (labels_0[:, 2] + labels_0[:, 4] / 2) + pad_w  # x2
                labels[:, 5] = ratio * h * (labels_0[:, 3] + labels_0[:, 5] / 2) + pad_h  # y2
        else:
            labels = np.array([])

        # Augment image and labels
        if self.augment:
            img, labels, M = random_affine(img, labels,
                                           degrees=(-5, 5),
                                           translate=(0.10, 0.10),
                                           scale=(0.50, 1.20))

        plot_flag = False
        if plot_flag:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            plt.figure(figsize=(50, 50))
            plt.imshow(img[:, :, ::-1])
            plt.plot(labels[:, [2, 4, 4, 2, 2]].T,
                     labels[:, [3, 3, 5, 5, 3]].T, '.-')
            plt.axis('off')
            plt.savefig('test.jpg')
            time.sleep(10)

        num_labels = len(labels)
        if num_labels > 0:
            # convert xyxy to xywh(center_x, center_y, b_w, b_h)
            labels[:, 2:6] = xyxy2xywh(labels[:, 2:6].copy())

            # normalize to 0~1
            labels[:, 2] /= width
            labels[:, 3] /= height
            labels[:, 4] /= width
            labels[:, 5] /= height
        if self.augment:
            # random left-right flip
            lr_flip = True
            if lr_flip & (random.random() > 0.5):
                img = np.fliplr(img)
                if num_labels > 0:
                    labels[:, 2] = 1 - labels[:, 2]

        img = np.ascontiguousarray(img[:, :, ::-1])  # BGR to RGB

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
                 transforms=None):
        """
        :param opt:
        :param root:
        :param paths:
        :param img_size:
        :param augment:
        :param transforms:
        """
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

        # ---- EdgeCrafter-style augmentation schedule ----
        self.cur_epoch         = 0
        self.mosaic_prob       = getattr(opt, 'mosaic_prob',       0.5)
        self.mosaic_epoch      = getattr(opt, 'mosaic_epoch',      25)
        self.stop_epoch        = getattr(opt, 'stop_epoch',        48)
        self.photodistort_prob = getattr(opt, 'photodistort_prob', 0.5)
        self.zoomout_max_scale = getattr(opt, 'zoomout_max_scale', 4.0)
        self.iou_crop_prob     = getattr(opt, 'iou_crop_prob',     0.8)

        tile_size = min(self.height, self.width) // 2
        self.mosaic_augmentor = MosaicAugmentor(
            output_size = tile_size,
            max_cached  = getattr(opt, 'mosaic_max_cached',  50),
            random_pop  = True,
            rotation    = getattr(opt, 'mosaic_rotation',    10.0),
            translation = (getattr(opt, 'mosaic_translate',  0.1),) * 2,
            scaling     = (getattr(opt, 'mosaic_scale_lo',   0.5),
                           getattr(opt, 'mosaic_scale_hi',   1.5)),
            fill        = 114,
        )

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

    # ------------------------------------------------------------------
    # Epoch-aware augmentation schedule (call once per epoch in train loop)
    # ------------------------------------------------------------------

    def set_epoch(self, epoch: int):
        """Set current epoch (0-indexed) for augmentation schedule."""
        self.cur_epoch = epoch

    # ------------------------------------------------------------------
    # Raw loader — no augmentation, returns cxcywh-norm labels
    # ------------------------------------------------------------------

    def _load_raw(self, img_path, label_path):
        """Load raw BGR image + normalized cxcywh labels. No resize, no aug."""
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f'File corrupt {img_path}')

        labels = np.zeros((0, 6), dtype=np.float32)
        if os.path.isfile(label_path):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                raw = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 6)
                if len(raw) > 0:
                    labels = raw   # [cls, tid, cx, cy, w, h] already normalized

        return img, labels

    # ------------------------------------------------------------------
    # Letterbox helper (adjusts labels for the added padding/scaling)
    # ------------------------------------------------------------------

    def _letterbox_labels(self, labels, orig_w, orig_h, ratio, pad_w, pad_h):
        """Adjust cxcywh-norm labels from original image to letterboxed image."""
        if len(labels) == 0:
            return labels
        out = labels.copy()
        out[:, 2] = (labels[:, 2] * orig_w * ratio + pad_w) / self.width
        out[:, 3] = (labels[:, 3] * orig_h * ratio + pad_h) / self.height
        out[:, 4] = labels[:, 4] * orig_w * ratio / self.width
        out[:, 5] = labels[:, 5] * orig_h * ratio / self.height
        return sanitize_boxes(out, self.width, self.height)

    # ------------------------------------------------------------------
    # Main __getitem__ with EdgeCrafter augmentation schedule
    # ------------------------------------------------------------------

    def __getitem__(self, idx):
        # locate sub-dataset
        for i, c in enumerate(self.cds):
            if idx >= c:
                ds          = list(self.label_files.keys())[i]
                start_index = c

        img_path   = self.img_files[ds][idx - start_index]
        label_path = self.label_files[ds][idx - start_index]

        img, labels = self._load_raw(img_path, label_path)
        orig_h, orig_w = img.shape[:2]

        epoch       = self.cur_epoch
        with_mosaic = (self.augment and
                       self.mosaic_prob > 0 and
                       epoch < self.mosaic_epoch and
                       random.random() < self.mosaic_prob)

        if not self.augment:
            # ---- validation / no-aug path ----
            pass

        elif with_mosaic:
            # ---- Mosaic path ----
            img, labels    = self.mosaic_augmentor(img, labels)
            img            = photometric_distort(img, p=self.photodistort_prob)
            orig_h, orig_w = img.shape[:2]

        elif epoch < self.stop_epoch:
            # ---- Strong aug path (no mosaic) ----
            img            = photometric_distort(img, p=self.photodistort_prob)
            img, labels    = random_zoom_out(img, labels, fill=114,
                                             max_scale=self.zoomout_max_scale)
            img, labels    = random_iou_crop(img, labels, p=self.iou_crop_prob)
            orig_h, orig_w = img.shape[:2]

        # else: epoch >= stop_epoch → clean path, no augmentation

        # ---- letterbox to network input size ----
        img, ratio, pad_w, pad_h = letterbox(img, height=self.height, width=self.width)
        labels = self._letterbox_labels(labels, orig_w, orig_h, ratio, pad_w, pad_h)

        # ---- random horizontal flip ----
        if self.augment and random.random() > 0.5:
            img = np.fliplr(img)
            if len(labels) > 0:
                labels[:, 2] = 1 - labels[:, 2]

        # ---- BGR → RGB, normalize (ImageNet mean/std) ----
        img = img[:, :, ::-1].astype(np.float32) / 255.0
        img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
        img = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))

        # ---- remap track IDs to global offsets ----
        if self.opt.id_weight > 0 and len(labels) > 0:
            for i in range(len(labels)):
                if labels[i, 1] > -1:
                    cls_id    = int(labels[i][0])
                    start_idx = self.tid_start_idx_of_cls_ids[ds].get(cls_id, 0)
                    labels[i, 1] += start_idx

        # ---- pack DETR-format targets ----
        num_objs       = min(len(labels), self.max_objs)   # mosaic can exceed K
        detr_boxes     = np.zeros((self.max_objs, 4), dtype=np.float32)
        detr_labels    = np.full((self.max_objs,), -1, dtype=np.int64)
        detr_track_ids = np.full((self.max_objs,), -1, dtype=np.int64)

        for k in range(num_objs):
            lb = labels[k]
            detr_boxes[k]     = lb[2:6]
            detr_labels[k]    = int(lb[0])
            detr_track_ids[k] = int(lb[1]) - 1   # 1-indexed → 0-indexed

        return {
            'input':          img,
            'detr_boxes':     detr_boxes,
            'detr_labels':    detr_labels,
            'detr_track_ids': detr_track_ids,
            'detr_num_objs':  np.array(num_objs, dtype=np.int64),
        }


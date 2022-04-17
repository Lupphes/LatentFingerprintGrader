import math
from typing import Tuple
import numpy as np
from pathlib import Path

from tensorflow.compat.v1 import disable_v2_behavior, get_default_graph, Session
from tensorflow.compat.v1.train import import_meta_graph
from tensorflow import Graph
from tensorflow import train
import scipy.misc


class ImportGraph():
    SHAPE = 128

    def __init__(self):
        # Defaults to Tensorflow 1.x behaviour
        disable_v2_behavior()
        self.graph = Graph()
        self.sess = Session(graph=self.graph)

    def load_graph(self, model_dir: Path, input_name='QueueInput/input_deque:0', output_name='reconstruction/gen:0', special=False):
        with self.graph.as_default():
            meta_file, ckpt_file = get_model_filenames(
                model_dir, special=special
            )

            saver = import_meta_graph(meta_file)
            saver.restore(self.sess, ckpt_file)

            self.images_placeholder = get_default_graph(
            ).get_tensor_by_name(input_name)
            self.phase_train_placeholder = get_default_graph(
            ).get_tensor_by_name(output_name)


class MinutiaeExtraction(ImportGraph):
    SHAPE = ImportGraph.SHAPE

    def __init__(self, model_dir: Path):
        super().__init__()
        self.weight = get_weights(self.SHAPE, self.SHAPE, 12, sigma=None)
        print(model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(f'{model_dir} folder was not found. Check your model directory')
        super().load_graph(model_dir)
        self.shape = self.phase_train_placeholder.get_shape()

    def run(self, img, minu_thr=0.2):
        h, w = img.shape
        weight = get_weights(128, 128, 12)
        nrof_samples = len(range(0, h, self.SHAPE // 2)) * \
            len(range(0, w, self.SHAPE // 2))
        patches = np.zeros((nrof_samples, self.SHAPE, self.SHAPE, 1))
        n = 0
        x = []
        y = []
        for i in range(0, h - self.SHAPE + 1, self.SHAPE // 2):
            for j in range(0, w - self.SHAPE + 1, self.SHAPE // 2):
                patch = img[i:i + self.SHAPE, j:j + self.SHAPE, np.newaxis]
                x.append(j)
                y.append(i)
                patches[n, :, :, :] = patch
                n = n + 1
        feed_dict = {self.images_placeholder: patches}
        minutiae_cylinder_array = self.sess.run(
            self.phase_train_placeholder, feed_dict=feed_dict)
        minutiae_cylinder = np.zeros((h, w, 12))
        minutiae_cylinder_array[:, -10:, :, :] = 0
        minutiae_cylinder_array[:, :10, :, :] = 0
        minutiae_cylinder_array[:, :, -10:, :] = 0
        minutiae_cylinder_array[:, :, 10, :] = 0

        for i in range(n):
            minutiae_cylinder[y[i]:y[i] + self.SHAPE, x[i]:x[i] + self.SHAPE, :] = minutiae_cylinder[
                y[i]:y[i] + self.SHAPE,
                x[i]:x[i] + self.SHAPE, :] + \
                minutiae_cylinder_array[i] * weight
        minutiae = get_minutiae_from_cylinder(
            minutiae_cylinder, thr=minu_thr)

        minutiae = refine_minutiae(
            minutiae, dist_thr=10, ori_dist=np.pi / 4)

        minutiae = self.remove_crowded_minutiae(minutiae)
        minutiae = np.asarray(minutiae)
        return minutiae

    def extract_minutiae(self, img, minu_thr=0.2):
        img = img / 128.0 - 1
        img = np.expand_dims(img, axis=2)
        img = np.expand_dims(img, axis=0)
        feed_dict = {self.images_placeholder: img}
        minutiae_cylinder = self.sess.run(
            self.phase_train_placeholder, feed_dict=feed_dict)

        minutiae_cylinder = np.squeeze(minutiae_cylinder, axis=0)
        minutiae = get_minutiae_from_cylinder2(
            minutiae_cylinder, thr=minu_thr)

        minutiae = refine_minutiae(
            minutiae, dist_thr=20, ori_dist=np.pi / 4)

        minutiae = self.remove_crowded_minutiae(minutiae)
        return minutiae

    def remove_crowded_minutiae(self, rawMinu):
        if rawMinu is None or len(rawMinu) == 0:
            return []

        if type(rawMinu) == 'list':
            rawMinu = np.asarray(rawMinu)
        dists = scipy.spatial.distance.cdist(
            rawMinu[:, :2], rawMinu[:, :2], 'euclidean')
        minu_num = rawMinu.shape[0]

        flag = np.ones((minu_num,), bool)
        neighor_num = 3
        neighor_thr = 12

        neighor_num2 = 5
        neighor_thr2 = 25
        if minu_num < neighor_num:
            return rawMinu
        for i in range(minu_num):
            # if two points are too close, both are removed
            ind = np.argsort(dists[i, :])
            if dists[i, ind[1]] < 5:
                flag[i] = False
                flag[ind[1]] = False
                continue
            if np.mean(dists[i, ind[1:neighor_num + 1]]) < neighor_thr:
                flag[i] = False
            if minu_num > neighor_num2 and np.mean(dists[i, ind[1:neighor_num2 + 1]]) < neighor_thr2:
                flag[i] = False
        rawMinu = rawMinu[flag, :]
        return rawMinu


class AutoEncoder(ImportGraph):
    SHAPE = ImportGraph.SHAPE

    def __init__(self, model_dir: Path):
        super().__init__()
        self.weight = get_weights(self.SHAPE, self.SHAPE, 1, sigma=None)
        super().load_graph(model_dir)
        self.shape = self.phase_train_placeholder.get_shape()

    def run(self, img):
        h, w = img.shape
        nrof_samples = len(range(0, h, self.SHAPE // 2)) * \
            len(range(0, w, self.SHAPE // 2))
        patches = np.zeros((nrof_samples, self.SHAPE, self.SHAPE, 1))
        n = 0
        x = []
        y = []
        for i in range(0, h - self.SHAPE + 1, self.SHAPE // 2):
            for j in range(0, w - self.SHAPE + 1, self.SHAPE // 2):
                patch = img[i:i + self.SHAPE, j:j + self.SHAPE, np.newaxis]
                x.append(j)
                y.append(i)
                patches[n, :, :, :] = patch
                n = n + 1
        feed_dict = {self.images_placeholder: patches}
        minutiae_cylinder_array = self.sess.run(
            self.phase_train_placeholder, feed_dict=feed_dict)

        minutiae_cylinder = np.zeros((h, w, 1))
        for i in range(n):
            minutiae_cylinder[y[i]:y[i] + self.SHAPE, x[i]:x[i] + self.SHAPE, :] = minutiae_cylinder[y[i]                                                                                                     :y[i] + self.SHAPE, x[i]:x[i] + self.SHAPE, :] + minutiae_cylinder_array[i] * self.weight

        minutiae_cylinder = minutiae_cylinder[:, :, 0]
        minV = np.min(minutiae_cylinder)
        maxV = np.max(minutiae_cylinder)
        minutiae_cylinder = (minutiae_cylinder - minV) / (maxV - minV) * 255

        return minutiae_cylinder

    def enhance_image(self, img):
        img = np.asarray(img)
        img = img / 128.0 - 1
        h, w = img.shape
        img = np.expand_dims(img, axis=2)
        img = np.expand_dims(img, axis=0)
        feed_dict = {self.images_placeholder: img}
        minutiae_cylinder = self.sess.run(
            self.phase_train_placeholder, feed_dict=feed_dict)

        minutiae_cylinder = np.squeeze(minutiae_cylinder, axis=0)
        minutiae_cylinder = np.squeeze(minutiae_cylinder, axis=2)
        minutiae_cylinder = minutiae_cylinder[:h, :w]
        minV = np.min(minutiae_cylinder)
        maxV = np.max(minutiae_cylinder)
        minutiae_cylinder = (minutiae_cylinder - minV) / (maxV - minV) * 255

        return minutiae_cylinder


class Descriptor(ImportGraph):
    SHAPE = ImportGraph.SHAPE

    def __init__(self, model_dir, input_name="batch_join:0", output_name="phase_train:0", result_name="Add:0"):
        super().__init__()
        super().load_graph(model_dir, input_name, output_name, special=True)
        with self.graph.as_default():
            self.embeddings = get_default_graph().get_tensor_by_name(result_name)

    def run(self, imgs):
        feed_dict = {self.images_placeholder: imgs,
                     self.phase_train_placeholder: False}
        return self.sess.run(self.embeddings, feed_dict=feed_dict)


def get_patch_index(patchSize_L, patchSize_H, oriNum, isMinu=1):
    if isMinu == 1:
        PI2 = 2 * math.pi
    else:
        PI2 = math.pi
    x = list(range(-patchSize_L // 2 + 1, patchSize_L // 2 + 1))
    x = np.array(x)
    y = list(range(-patchSize_H // 2 + 1, patchSize_H // 2 + 1))
    y = np.array(y)
    xv, yv = np.meshgrid(x, y)
    patchIndexV = {}
    patchIndexV['x'] = []
    patchIndexV['y'] = []
    for i in range(oriNum):

        th = i * PI2 / oriNum
        u = xv * np.cos(th) - yv * np.sin(th)
        v = xv * np.sin(th) + yv * np.cos(th)
        u = np.around(u)
        v = np.around(v)
        patchIndexV['x'].append(u)
        patchIndexV['y'].append(v)
    return patchIndexV


def get_weights(h, w, c, sigma=None):
    Y, X = np.mgrid[0:h, 0:w]
    x0 = w // 2
    y0 = h // 2
    if sigma is None:
        sigma = (np.max([h, w]) * 1. / 3)**2
    weight = np.exp(-((X - x0) * (X - x0) + (Y - y0) * (Y - y0)) / sigma)
    weight = np.stack((weight,) * c, axis=2)
    return weight


def get_model_filenames(model_dir: Path, special=False) -> Tuple[Path, str]:
    meta_file = None
    for p in Path(model_dir).rglob("*"):
        if p.suffix == '.meta':
            meta_file = p.as_posix()
        elif special and p.suffix == '.index':
            ckpt_file = (model_dir/p.stem).as_posix()
    if not special:
        ckpt_file = train.latest_checkpoint(model_dir.as_posix())
    return meta_file, ckpt_file


def get_minutiae_from_cylinder(minutiae_cylinder, thr=0.5):
    h, w, c = minutiae_cylinder.shape

    max_arg = np.argmax(minutiae_cylinder, axis=2)
    max_val = np.max(minutiae_cylinder, axis=2)
    r = 2
    minutiae = []
    for i in range(r, h - r):
        for j in range(r, w - r):
            v = max_val[i, j]
            ind = max_arg[i, j]
            if v < thr:
                continue
            if ind == 0:
                local_value = np.concatenate((minutiae_cylinder[i - r:i + r + 1, j - r:j + r + 1, -1::],
                                              minutiae_cylinder[i - r:i + r + 1, j - r:j + r + 1, 0:2]), 2)
            elif ind == c - 1:
                local_value = np.concatenate((minutiae_cylinder[i - r:i + r + 1, j - r:j + r + 1, -2:-1],
                                              minutiae_cylinder[i - r:i +
                                                                r + 1, j - r:j + r + 1, -1::],
                                              minutiae_cylinder[i - r:i + r + 1, j - r:j + r + 1, 0:1]), 2)
            else:
                local_value = minutiae_cylinder[i - r:i +
                                                r + 1, j - r:j + r + 1, ind - 1:ind + 2]
            local_value = local_value.copy()
            local_value[r, r, 1] = 0
            local_max_ind = np.argmax(local_value)
            local_max_ind = np.unravel_index(local_max_ind, local_value.shape)
            local_max_v = local_value[local_max_ind]

            if local_max_v > v:
                continue

            # refine the minutiae orientation
            ind_1 = ind - 1
            if ind_1 < 0:
                ind_1 = ind_1 + c

            ind_2 = ind + 1
            if ind_2 >= c:
                ind_2 = ind_2 - c

            y1 = minutiae_cylinder[i, j, ind_1]
            y2 = minutiae_cylinder[i, j, ind] - y1
            y3 = minutiae_cylinder[i, j, ind_2] - y1
            pred = 0.5 * (y3 - 4 * y2) / (y3 - 2 * y2)
            confidence = -(2 * y2 - 0.5 * y3) * (2 * y2 -
                                                 0.5 * y3) / (2 * y3 - 4 * y2) + v
            if confidence < v:
                print(confidence, v)
            ori_ind = ind_1 + pred
            ori = ori_ind * 1.0 / c * 2 * math.pi
            minutiae.append([j, i, ori, confidence])  # v is the confidence
    if len(minutiae) > 0:
        minutiae = np.asarray(minutiae, dtype=np.float32)
        I = np.argsort(minutiae[:, 3])
        I = I[::-1]
        minutiae = minutiae[I, :]
    return minutiae


def get_minutiae_from_cylinder2(minutiae_cylinder, thr=0.5):
    h, w, c = minutiae_cylinder.shape

    max_arg = np.argmax(minutiae_cylinder, axis=2)
    max_val = np.max(minutiae_cylinder, axis=2)

    candi_ind = np.where(max_val > thr)

    candi_num = len(candi_ind[0])

    r = 15
    r2 = int(r / 2)
    minutiae = []

    for k in range(candi_num):
        i = candi_ind[0][k]
        j = candi_ind[1][k]
        if i < r2 or j < r2 or i > h - r2 - 1 or j > w - r2 - 1:
            continue
        v = max_val[i, j]
        if v > max_val[i - 1, j - 1] and v > max_val[i - 1, j] and v > max_val[i - 1, j + 1] \
                and v > max_val[i, j - 1] and v > max_val[i, j + 1] \
                and v > max_val[i + 1, j - 1] and v > max_val[i + 1, j] and v > max_val[i + 1, j + 1]:

            v = max_val[i, j]
            ind = max_arg[i, j]

            # refine the minutiae orientation
            ind_1 = ind - 1
            if ind_1 < 0:
                ind_1 = ind_1 + c
            ind_2 = ind + 1
            if ind_2 >= c:
                ind_2 = ind_2 - c

            y1 = minutiae_cylinder[i, j, ind_1]
            y2 = minutiae_cylinder[i, j, ind] - y1
            y3 = minutiae_cylinder[i, j, ind_2] - y1
            pred = 0.5 * (y3 - 4 * y2) / (y3 - 2 * y2)
            confidence = -(2 * y2 - 0.5 * y3) * (2 * y2 -
                                                 0.5 * y3) / (2 * y3 - 4 * y2) + v
            ori_ind = ind_1 + pred
            ori = ori_ind * 1.0 / c * 2 * math.pi
            minutiae.append([j, i, ori, confidence])  # v is the confidence
    if len(minutiae) > 0:
        minutiae = np.asarray(minutiae, dtype=np.float32)
        I = np.argsort(minutiae[:, 3])
        I = I[::-1]
        minutiae = minutiae[I, :]
    return minutiae


def refine_minutiae(minutiae, dist_thr=10, ori_dist=np.pi / 4):
    minu_num = len(minutiae)
    flag = np.ones((minu_num,), dtype=np.int64)
    if len(minutiae) == 0:
        return minutiae
    for i in range(minu_num):
        x0 = minutiae[i, 0]
        y0 = minutiae[i, 1]
        ori0 = minutiae[i, 2]
        for j in range(i + 1, minu_num):
            x1 = minutiae[j, 0]
            y1 = minutiae[j, 1]
            ori1 = minutiae[j, 2]

            dist = np.sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0))

            if dist < dist_thr:
                flag[j] = 0
                continue

            ori_diff = np.fabs(ori1 - ori0)
            ori_diff = np.min([ori_diff, np.pi * 2 - ori_diff])
            if dist < 20 and ori_diff < ori_dist:
                flag[j] = 0

    minutiae = minutiae[flag == 1, :]
    return minutiae

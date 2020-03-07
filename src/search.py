import torch
import torch.nn as nn
import numpy as np
from src.utils import *
from PIL import Image
from torch.nn import functional as F


class Search(object):
    def __init__(self,
                 model,
                 root_path,
                 class_names,
                 data_size=2,
                 dtype='train'):

        self.model = model
        self.model.eval()

        self.cls_names = class_names
        self.cls_paths = []
        self.cls = None

        self.labels = []

        for i, name in enumerate(self.cls_names):
            paths = get_class_path(root_path, dtype, name)
            np.random.shuffle(paths)
            self.cls_paths.append(paths[:data_size])
            self.labels.append(i)

    def get_conv_grad(self):
        grads = []

        for m in self.model.modules():
            if type(m) == nn.Conv2d:
                grads.append(m.weight.grad.cpu().detach().numpy())

        return grads

    def backprop(self, image_path, inverse=None):
        self.model.zero_grad()
        img = pil_to_tensor(Image.open(image_path))
        # forward
        output = self.model(img).to('cuda')
        # acc
        h_x = F.softmax(output, dim=1).data.squeeze()
        pred = h_x.argmax(0).item()

        if pred is not self.cls:
            return None

        if inverse is not None:
            pred = inverse

        print(f"pred: {pred} label {self.cls}")

        one_hot_output = torch.zeros(1, h_x.size()[0]).to('cuda')
        one_hot_output[0][pred] = 1

        output.backward(gradient=one_hot_output)

        grads = self.get_conv_grad()

        return grads

    @staticmethod
    def diff(t, f):
        t = abs(t)
        f = abs(f)

        sum_t = t.reshape(t.shape[0], -1).sum(1)
        sum_f = f.reshape(f.shape[0], -1).sum(1)

        return (sum_t - sum_f) / (sum_t + 1e-5)

    def get_diffs(self, cls):
        self.cls = cls
        print(f"Get Diff {self.cls_names[cls]}")

        diff_labels = [l for l in self.labels if l is not cls]

        total_diff = 0

        for image_paths in zip(self.cls_paths[cls]):
            f_grad = []
            for img in image_paths:
                diffs = []

                t_grad = self.backprop(img, cls)

                if t_grad is None:
                    continue

                for i in range(len(diff_labels) - 1):
                    f_grad1 = self.backprop(img, inverse=diff_labels[i])
                    f_grad2 = self.backprop(img, inverse=diff_labels[i + 1])

                    if i == 0:
                        for g1, g2 in zip(f_grad1, f_grad2):
                            f_grad.append(np.minimum(g1, g2))
                    else:
                        print("Not implement")

                for t, f in zip(t_grad, f_grad):
                    diffs.append(self.diff(t, f))

                total_diff += np.array(diffs)

        return total_diff

    def get_binary_diffs(self, cls):
        self.cls = cls
        print(f"Get Diff {self.cls_names[cls]}")

        total_diff = 0

        for image_paths in zip(self.cls_paths[cls]):
            for img in image_paths:
                diffs = []

                t_grad = self.backprop(img, 1)
                f_grad = self.backprop(img, inverse=0)

                if t_grad is None:
                    continue

                for t, f in zip(t_grad, f_grad):
                    diffs.append(self.diff(t, f))

                total_diff += np.array(diffs)

        return total_diff


def get_filter_idx(diffs):
    filter_idx = [[] for _ in range(len(diffs))]

    for i, diff in enumerate(diffs):
        for j, d in enumerate(diff):
            if i < 2:
                filter_idx[i].append(j)
            else:
                if d >= 0:
                    filter_idx[i].append(j)

    return filter_idx



import numpy as np
from PIL import Image, ImageDraw
from collections import Counter
import heapq
import math
from typing import List, Optional, Tuple

import torch
from torch import tensor
from torchvision.transforms.functional import resize as imresize, pil_to_tensor
from torchvision.transforms import ToTensor, ToPILImage
import torch.nn.functional as F

from copy import deepcopy


AREA_POWER = 0.25
ERROR_CALC_DOWNSAMPLE = ("min_patch", "once")


class QuadTree:
    def __init__(self,
                 image: torch.FloatTensor,
                 num_patches: int,
                 min_patch_size: int = 16,
                 max_patch_size: Optional[int] = None,
                 crop_size: Optional[int] = None,
                 resize_size: Optional[int] = None,
                 error_calc_downsample: str = "min_patch"):
        assert error_calc_downsample in ERROR_CALC_DOWNSAMPLE
        self.image = image
        self.num_patches = num_patches
        self.error_calc_downsample = error_calc_downsample

        _, self.width, self.height = self.image.shape
        assert self.width == self.height, self.image.shape
        assert np.abs(int(np.log2(self.width))- np.log2(self.width)) < 1e-6 # dims are power of 2

        self.min_patch_size = min_patch_size
        self.max_patch_size = max_patch_size if max_patch_size else self.width
        assert (self.min_patch_size % 1 == 0) and (self.max_patch_size % 1 == 0)
        self.min_patch_size, self.max_patch_size = int(
            self.min_patch_size), int(self.max_patch_size)

        self.heap = []

        self.root = Quad(full_image=self.image, box=(0, 0, self.width, self.height),
                         depth=0, small_patch_size=self.min_patch_size,
                         error_calc_downsample=self.error_calc_downsample)
        self.error_sum = self.root.error * self.root.area
        self.push(self.root)
        self.initial_split()

    def run(self):
        """
        Returns:
            patches : patch representing (left, top, right, bottom)
            boxes   : patch box positions: (left, top, right, bottom)
        """
        while len(self.heap) < self.num_patches:
            self.split()
        boxes = self.boxes
        boxes[:,:2]//=self.min_patch_size
        boxes[:,2]=torch.log2(boxes[:,2]/self.min_patch_size)

        return self.patches, boxes

    @property
    def quads(self):
        return [quad for _, _, quad in self.heap]

    @property
    def boxes(self):
        return torch.stack([torch.Tensor([
            (quad.box[2]+quad.box[0])/2,
            (quad.box[3]+quad.box[1])/2,
            quad.box[3]-quad.box[1],
            ]) for quad in self.quads]).long()

    @property
    def patches(self):
        patches = torch.stack([quad.color.flatten() for quad in self.quads])
        return patches

    def average_error(self):
        return self.error_sum / (self.width * self.height)

    def push(self, quad):
        score = -quad.error * (quad.area ** AREA_POWER)
        heapq.heappush(self.heap, (quad.leaf, score, quad))

    def pop(self):
        return heapq.heappop(self.heap)[-1]

    def initial_split(self):
        if self.max_patch_size is None:
            return
        num_splits = math.ceil(math.log2(max(self.width, self.height) / self.max_patch_size))
        for _ in range(num_splits):
            heap = list(self.heap)
            self.heap = []
            for _, _, quad in heap:
                self.split(quad)

    def split(self, quad=None):
        if quad is None:
            quad = self.pop()

        small_quads = []
        while (quad.size() == self.min_patch_size) and (len(self.heap) > 0):
            small_quads.append(quad)
            quad = self.pop()

        if quad.size() > self.min_patch_size:
            self.split_quad(quad)
        else:
            small_quads.append(quad)

        for small_quad in small_quads:
            self.push(small_quad)

    def split_quad(self, quad):
        self.error_sum -= quad.error * quad.area
        children = quad.split()
        for child in children:
            self.push(child)
            self.error_sum += child.error * child.area

    def render(self, path):
        im = Image.new('RGB', (self.width, self.height))
        to_pil= ToPILImage()
        for (x, y, size), patch in zip(self.boxes, self.patches):
            l = int(x-size//2)
            r = int(x+size//2)
            t = int(y-size//2)
            b = int(y+size//2)
            #patch = self.image.crop((l, t, r, b))
            print(patch.max(), patch.min(), patch.mean())
            tmp = to_pil((patch.reshape((3, 16, 16))*255).type(torch.uint8)).resize((size, size))
            print(np.max(tmp), np.min(tmp), np.mean(tmp))
            ImageDraw.Draw(tmp).rectangle((0, 0, size, size), fill=None)
            im.paste(tmp, (l, t, r, b))
        im.save(path)


class Quad:
    def __init__(self, full_image, box, depth, small_patch_size, error_calc_downsample):
        assert error_calc_downsample in ERROR_CALC_DOWNSAMPLE
        self.box = box
        self.depth = depth
        self.small_patch_size = small_patch_size
        self.full_image = full_image
        self.error_calc_downsample = error_calc_downsample

        self.color, self.error = self.calc_error()

        self.leaf = self.is_leaf()
        self.area = self.compute_area()
        self.children = []

    def calc_error(self):
        l, t, r, b = self.box
        quad_size = r - l
        quad_features = self.full_image[:, t:b, l:r]
        if quad_size == self.small_patch_size:
            return quad_features, 0.

        downsample_size = (int(quad_size / 2) if self.error_calc_downsample == "once"
                           else self.small_patch_size)

        # interpolate expect batch dim. so add it and then remove it
        small_quad_features = F.interpolate(quad_features[None,:], size=downsample_size)[0]
        blurry_quad_features = F.interpolate(small_quad_features[None,:], size=quad_size)[0]

        #small_quad_features = imresize(quad_features, [downsample_size])
        #blurry_quad_features = imresize(small_quad_features, [quad_size])
        mse = torch.mean((quad_features - blurry_quad_features) ** 2)
        error = torch.sqrt(mse)
        return small_quad_features, error

    def is_leaf(self):
        l, t, r, b = self.box
        is_leaf = (r - l <= self.small_patch_size) or (b - t <= self.small_patch_size)
        return int(is_leaf)

    def compute_area(self):
        l, t, r, b = self.box
        return (r - l) * (b - t)

    def split(self):
        l, t, r, b = self.box
        lr = l + (r - l) // 2
        tb = t + (b - t) // 2
        depth = self.depth + 1
        tl = Quad(self.full_image, (l, t, lr, tb), depth,
                  self.small_patch_size, self.error_calc_downsample)
        tr = Quad(self.full_image, (lr, t, r, tb), depth,
                  self.small_patch_size, self.error_calc_downsample)
        bl = Quad(self.full_image, (l, tb, lr, b), depth,
                  self.small_patch_size, self.error_calc_downsample)
        br = Quad(self.full_image, (lr, tb, r, b), depth,
                  self.small_patch_size, self.error_calc_downsample)
        self.children = (tl, tr, bl, br)
        return self.children

    def get_leaf_nodes(self, max_depth=None):
        if not self.children:
            return [self]
        if max_depth is not None and self.depth >= max_depth:
            return [self]
        result = []
        for child in self.children:
            result.extend(child.get_leaf_nodes(max_depth))
        return result

    def size(self):
        return self.box[2] - self.box[0]

    def __lt__(self, other):
        return self.error < other.error

    def __le__(self, other):
        return self.error <= other.error

if __name__=='__main__':
    impath = '/mnt/datasets/Imagenet/imaganet_db/val/n01484850/ILSVRC2012_val_00004311.JPEG'
    im = Image.open(impath).convert('RGB').resize((256, 256))
    im = ToTensor()(im)
    model = QuadTree(im, num_patches = 100).run()
    model.render('output.png')
    print( '-' * 32)
    depth = Counter(x.depth for x in model.quads)
    for key in sorted(depth):
        value = depth[key]
        n = 4 ** key
        pct = 100.0 * value / n
        print ('%3d %8d %8d %8.2f%%' % (key, n, value, pct))
    print ('-' * 32)
    print ('             %8d %8.2f%%' % (len(model.quads), 100))


    Image.open('output.png').convert('RGB')

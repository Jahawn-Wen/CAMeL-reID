import json
import os
import random
import numpy as np
from random import random as rand
from PIL import Image
from PIL import ImageFile

import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import hflip, resize


ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


class reid_train_dataset(Dataset):
    def __init__(self, config, transform):
        self.image_root = config['image_root']
        self.transform = transform

        print('train_file', config['train_file'])
        ann_file = config['train_file']
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))

        self.img_ids = {}
        n = 0
        print('img_ids：n-->start:', n)
        for ann in self.ann:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1
        print('img_ids：n-->end:', n)

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]
        try:
            image_path = os.path.join(self.image_root, ann['image'])
        except:
            print("self.image_root", self.image_root)
            print("ann['image']", ann['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        img_id = ann['image_id']

        return image, self.img_ids[img_id]


class reid_test_dataset(Dataset):
    def __init__(self, config, transform):
        query = json.load(open(config['query_file'], 'r'))
        gallery = json.load(open(config['gallery_file'], 'r'))
        self.transform = transform
        self.image_root = config['image_root']

        self.q = []
        self.g = []
        self.q_pids = []
        self.g_pids = []
        self.q_cam = []
        self.g_cam = []

        for img_id, ann in enumerate(query):
            self.q_pids.append(int(ann['image_id']))
            self.q.append(ann['image'])
            self.q_cam.append(ann['cam'])

        for img_id, ann in enumerate(gallery):
            self.g_pids.append(int(ann['image_id']))
            self.g.append(ann['image'])
            self.g_cam.append(ann['cam'])

        # print(len(self.q), len(self.q_pids))
        # for i, t in enumerate(self.q):
        #     print(t)
        #     print(self.q_pids[i])
        # print(len(self.q), len(self.q_pids))

        # print(len(self.g), len(self.g_pids))
        # for i, t in enumerate(self.g):
        #     print(t)
        #     print(self.g_pids[i])
        # print(len(self.g), len(self.g_pids))

    def __len__(self):
        return len(self.g)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_root, self.g[index])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image, index

import os
import json
import torch.utils.data as data
import numpy as np
from PIL import Image
import torch

import warnings

warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


# class IN22KDATASET(data.Dataset):
#     def __init__(self, root, ann_file='', transform=None, target_transform=None):
#         super(IN22KDATASET, self).__init__()

#         self.data_path = root
#         self.ann_path = os.path.join(self.data_path, ann_file)
#         self.transform = transform
#         self.target_transform = target_transform
#         # id & label: https://github.com/google-research/big_transfer/issues/7
#         # total: 21843; only 21841 class have images: map 21841->9205; 21842->15027
#         self.database = json.load(open(self.ann_path))

#     def _load_image(self, path):
#         try:
#             im = Image.open(path)
#         except:
#             print("ERROR IMG LOADED: ", path)
#             random_img = np.random.rand(224, 224, 3) * 255
#             im = Image.fromarray(np.uint8(random_img))
#         return im

#     def __getitem__(self, index):
#         """
#         Args:
#             index (int): Index
#         Returns:
#             tuple: (image, target) where target is class_index of the target class.
#         """
#         idb = self.database[index]

#         # images
#         images = self._load_image(self.data_path + '/' + idb[0]).convert('RGB')
#         if self.transform is not None:
#             images = self.transform(images)

#         # target
#         target = int(idb[1])
#         if self.target_transform is not None:
#             target = self.target_transform(target)

#         return images, target

#     def __len__(self):
#         return len(self.database)

class   IN22KDATASET(data.Dataset):
    def __init__(self, data_dir, file, augment,
                 num_class=14, img_depth=3, heatmap_path=None,
                 pretraining=False):
        self.img_list = []
        self.img_label = []
        with open(file, "r")    as fileDescriptor:
            line = True
            while line:
                line = fileDescriptor.readline()
                if line:
                    lineItems = line.split()
                    imagePath = os.path.join(data_dir, lineItems[0])
                    imageLabel = lineItems[1:num_class + 1]
                    imageLabel = [int(i) for i in imageLabel]
                    self.img_list.append(imagePath)
                    self.img_label.append(imageLabel)

        self.augment = augment
        self.img_depth = img_depth
        if heatmap_path is not None:
            # self.heatmap = cv2.imread(heatmap_path)
            self.heatmap = Image.open(heatmap_path).convert('RGB')
        else:
            self.heatmap = None
        self.pretraining = pretraining

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):

        file = self.img_list[index]
        label = self.img_label[index]

        imageData = Image.open(file).convert('RGB')
        if self.heatmap is None:
            imageData = self.augment(imageData)
            img = imageData
            label = torch.tensor(label, dtype=torch.float)
            if self.pretraining:
                label = -1
            return img, label
        else:
            # heatmap = Image.open('nih_bbox_heatmap.png')
            heatmap = self.heatmap
            # heatmap = torchvision.transforms.functional.to_pil_image(self.heatmap)
            imageData, heatmap = self.augment(imageData, heatmap)
            img = imageData
            # heatmap = torch.tensor(np.array(heatmap), dtype=torch.float)
            heatmap = heatmap.permute(1, 2, 0)
            label = torch.tensor(label, dtype=torch.float)
            if self.pretraining:
                label = -1
            return [img, heatmap], label

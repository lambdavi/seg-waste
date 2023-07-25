import os

from PIL import Image
from torch.utils import data
import numpy as np
from config import cfg
from torch import from_numpy

processed_train_path = os.path.join(cfg.DATA.DATA_PATH, 'train')
processed_val_path = os.path.join(cfg.DATA.DATA_PATH, 'val')


def default_loader(path):
    return Image.open(path)


def make_dataset(mode):
    images = []
    if mode == 'train':
        processed_train_img_path = processed_train_path
        processed_train_mask_path = cfg.DATA.DATA_PATH
        for img_name in os.listdir(processed_train_img_path):
            item = (os.path.join(processed_train_img_path, img_name),
                    os.path.join(processed_train_mask_path + '/labels/train/', img_name))
            images.append(item)
    elif mode == 'val':
        processed_val_img_path = processed_val_path
        processed_val_mask_path = cfg.DATA.DATA_PATH
        for img_name in os.listdir(processed_val_img_path):
            item = (os.path.join(processed_val_img_path, img_name),
                    os.path.join(processed_val_mask_path + '/labels/val/', img_name))
            images.append(item)
    return images

#translator = {0: 255, 1:0, 2:1, 3:2, 4:3}

class resortit(data.Dataset):
    def __init__(self, mode, simul_transform=None, transform=None, target_transform=None):
        self.imgs = make_dataset(mode)
        if len(self.imgs) == 0:
            raise (RuntimeError('Found 0 images, please check the data set'))
        self.loader = default_loader
        self.simul_transform = simul_transform
        self.transform = transform
        #self.mapping = self.get_mapping()
        self.target_transform = target_transform


    """ @staticmethod
    def get_mapping():s
        mapping = np.zeros((256,), dtype=np.int64) + 255
        for gta_idx, data_idx in translator.items():
            mapping[gta_idx] = data_idx
        return lambda x: from_numpy(mapping[x])"""
    
    def __getitem__(self, index):
        img_path, mask_path = self.imgs[index]
        img = self.loader(img_path)
        mask = np.array(self.loader(mask_path))
        if cfg.TASK == "binary":
            mask[mask>0] = 1   ##########Only Binary Segmentation#####
        else:
            mask[mask==0] = 255
        
        mask = Image.fromarray(mask)
        if self.simul_transform is not None:
            img, mask = self.simul_transform(img, mask)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)

        return img, mask

    def __len__(self):
        return len(self.imgs)
    

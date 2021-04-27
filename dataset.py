import numpy as np
import os
import random
from PIL import Image
import torch
from torchvision.transforms import Compose
from torchvision.transforms import ToPILImage
from torch.utils.data import Dataset

EXTENSIONS = ['.jpg', '.png']

normalise_params = [1./255, # SCALE
                    np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3)),  # MEAN
                    np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))]  # STD

#color_transform = Colorize()
#image_transform = ToPILImage()


# 把3通道的RBG图像转成1通道的label
def color2label(img):

    def t(gt, a, b, c, t):
        w1 = gt[:, :, 0] == a
        w2 = gt[:, :, 1] == b
        w3 = gt[:, :, 2] == c
        x = w1 * w2 * w3
        f = np.where(x > 0, t, 0)
        return f

    img = np.array(img, dtype=np.int)
    gt = np.zeros([img.shape[0], img.shape[1]])
    gt += t(img, 0, 0, 255, 0)  # 住宅
    gt += t(img, 255, 255, 255, 1)  # 道路
    gt += t(img, 0, 255, 0, 2)  # 树
    gt += t(img, 0, 255, 255, 3)  # 灌木
    gt += t(img, 255, 255, 0, 4)  # 汽车
    gt += t(img, 255, 0, 0, 5)  # clutter
    return gt


class Potsdam(Dataset):
    def __init__(self, root, iterations=50000):
        self.stage = "train"
        self.iterations = iterations
        self.images_root = os.path.join(root, 'top', 'train')
        self.dsm_images_root = os.path.join(root, 'dsm')

        self.labels_root = os.path.join(root, 'gt')

        self.filenames = [f for f in os.listdir(self.images_root)]
        self.filenames.sort()

        self.train_transform = Compose([
            RandomCrop(256),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            Normalise(*normalise_params),
            ToTensor(),
        ])

        self.pkg = []
        for i in self.filenames:
            image = Image.open(os.path.join(self.images_root, i))
            dsm_image = Image.open(os.path.join(self.dsm_images_root, i.replace("top", "dsm")))
            label = Image.open(os.path.join(self.labels_root, i))
            label = color2label(label)
            self.pkg.append((np.array(image), np.array(dsm_image), np.array(label)))

    def __getitem__(self, index):
        index = index % len(self.filenames)
        pkg = self.pkg[index]

        sample = {'image': pkg[0], 'dsm_img': pkg[1], 'label': pkg[2]}
        sample = self.train_transform(sample)

        return sample

    def __len__(self):
        if self.stage == "train":
            return self.iterations
        else:
            return len(self.filenames)

class RandomCrop(object):
    def __init__(self, size):

        self.shape = (int(size), int(size))

    @staticmethod
    def get_params(sample, output_size):
        assert sample['image'].shape == sample['label'].shape

        w, h = sample['image'].shape
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, sample):

        i, j, h, w = self.get_params(sample, self.shape)
        sample = {
            'image': Crop(sample['image'], i, j, h, w),
            'dsm_img': Crop(sample['dsm_img'], i, j, h, w),
            'label': Crop(sample['label'], i, j, h, w),
        }
        return sample


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            sample = {
                'image': HorizontalFlip(sample['image']),
                'dsm_img': HorizontalFlip(sample['dsm_img']),
                'label': HorizontalFlip(sample['label']),
            }
            return sample
        return sample


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            sample = {
                'image': VerticalFlip(sample['image']),
                'dsm_img': VerticalFlip(sample['dsm_img']),
                'label': VerticalFlip(sample['label']),
            }
            return sample
        return sample


class Normalise(object):

    def __init__(self, scale, mean, std):
        self.scale = scale
        self.mean = mean
        self.std = std

    def __call__(self, sample):

        return {'image': (self.scale * sample['image'] - self.mean) / self.std,
                'dsm_img': (self.scale * sample['dsm_img'] - self.mean) / self.std,
                'label' : sample['label']}


class Normalise_for_val(object):

    def __init__(self, scale, mean, std):
        self.scale = scale
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image = np.array(image)

        return (self.scale * image - self.mean) / self.std


def Crop(img, i, j, h, w):
    return img[j: j + w, i:i + h, :]


def VerticalFlip(img):
    b = np.zeros_like(img)
    for i in range(img.shape[0]):
        b[:, img.shape[0] - i - 1] += img[:, i]
    del img
    return b

def HorizontalFlip(img):
    b = np.zeros_like(img)
    for i in range(img.shape[0]):
        b[img.shape[0] - i - 1, :] += img[i, :]
    del img
    return b

def normalise_for_val(image):
    image = np.array(image)
    if image.shape[0]==1:
        return image
    normalise_params = [1. / 255,  # SCALE
                        np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3)),  # MEAN
                        np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))]  # STD
    scale = normalise_params[0]
    mean = normalise_params[1]
    std = normalise_params[2]
    return (image * scale - mean)/std


class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, sample):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        image, dsm_img, mask = sample['image'], sample['dsm_img'], sample['label']
        image = image.transpose((2, 0, 1))
        dsm_img = dsm_img.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).float(),
                'dsm_img': torch.from_numpy(dsm_img).float(),
                'label': mask}

    def __repr__(self):
        return self.__class__.__name__ + '()'

class ToTensor_for_val(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, image):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        if image.size == 256*256:
            return torch.from_numpy(image).float().unsqueeze(0)
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image).float()

    def __repr__(self):
        return self.__class__.__name__ + '()'



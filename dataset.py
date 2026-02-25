import torch.utils.data as data
from PIL import Image
from torchvision.datasets import VisionDataset
from typing import Any, Callable, Optional, Tuple
from base_dataset import get_params, get_transform
import os
import glob
import numpy as np
import torch
import tifffile as tiff

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    if os.path.isfile(dir):
        images = [i for i in np.genfromtxt(
            dir, dtype=np.str, encoding='utf-8')]
    else:
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    return images


def pil_loader(path):
    return Image.open(path).convert('RGB')

class Sen2_MTC_New_Multi(data.Dataset):
    def __init__(self, data_root, mode='train'):
        self.data_root = data_root
        self.mode = mode
        self.filepair = []
        self.image_name = []

        if mode == 'train':
            self.tile_list = np.loadtxt(os.path.join(
                self.data_root, 'train.txt'), dtype=str)
        elif mode == 'val':
            self.data_augmentation = None
            self.tile_list = np.loadtxt(
                os.path.join(self.data_root, 'val.txt'), dtype=str)
        elif mode == 'test':
            self.data_augmentation = None
            self.tile_list = np.loadtxt(
                os.path.join(self.data_root, 'test.txt'), dtype=str)
        for tile in self.tile_list:
            image_name_list = [image_name.split('.')[0] for image_name in os.listdir(
                os.path.join(self.data_root, 'Sen2_MTC', tile, 'cloudless'))]

            for image_name in image_name_list:
                image_cloud_path0 = os.path.join(
                    self.data_root, 'Sen2_MTC', tile, 'cloud', image_name + '_0.tif')
                image_cloud_path1 = os.path.join(
                    self.data_root, 'Sen2_MTC', tile, 'cloud', image_name + '_1.tif')
                image_cloud_path2 = os.path.join(
                    self.data_root, 'Sen2_MTC', tile, 'cloud', image_name + '_2.tif')
                image_cloudless_path = os.path.join(
                    self.data_root, 'Sen2_MTC', tile, 'cloudless', image_name + '.tif')

                self.filepair.append(
                    [image_cloud_path0, image_cloud_path1, image_cloud_path2, image_cloudless_path])
                self.image_name.append(image_name)

        self.augment_rotation_param = np.random.randint(
            0, 4, len(self.filepair))
        self.augment_flip_param = np.random.randint(0, 3, len(self.filepair))
        self.index = 0

    def __getitem__(self, index):
        cloud_image_path0, cloud_image_path1, cloud_image_path2 = self.filepair[
            index][0], self.filepair[index][1], self.filepair[index][2]
        cloudless_image_path = self.filepair[index][3]
        image_cloud0 = self.image_read(cloud_image_path0)
        image_cloud1 = self.image_read(cloud_image_path1)
        image_cloud2 = self.image_read(cloud_image_path2)
        image_cloudless = self.image_read(cloudless_image_path)

        # return [image_cloud0, image_cloud1, image_cloud2], image_cloudless, self.image_name[index]
        ret = {}
        ret['gt_image'] = image_cloudless[:3, :, :]
        ret['cond_image'] = torch.cat([image_cloud0[:3, :, :].unsqueeze(0), image_cloud1[:3, :, :].unsqueeze(0),
                                       image_cloud2[:3, :, :].unsqueeze(0)], dim=0)
        ret['path'] = self.image_name[index]+".png"
        return ret

    def __len__(self):
        return len(self.filepair)

    def image_read(self, image_path):
        img = tiff.imread(image_path)
        img = (img / 1.0).transpose((2, 0, 1))

        if self.mode == 'train':
            if not self.augment_flip_param[self.index // 4] == 0:
                img = np.flip(img, self.augment_flip_param[self.index // 4])
            if not self.augment_rotation_param[self.index // 4] == 0:
                img = np.rot90(
                    img, self.augment_rotation_param[self.index // 4], (1, 2))
            self.index += 1

        if self.index // 4 >= len(self.filepair):
            self.index = 0

        image = torch.from_numpy((img.copy())).float()
        image = image / 10000.0
        mean = torch.as_tensor([0.5, 0.5, 0.5, 0.5],
                               dtype=image.dtype, device=image.device)
        std = torch.as_tensor([0.5, 0.5, 0.5, 0.5],
                              dtype=image.dtype, device=image.device)
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)
        image.sub_(mean).div_(std)

        return image

class MultipleDataset(VisionDataset):

    SUBFOLDERS = ("cloudy", "clear")

    def __init__(
        self,
        data_root: str,
        band: int = 4,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(data_root, transform, target_transform, transforms)
        root = data_root
        self.band = band

        cloudy_0_rgb_pathname = os.path.join(
            root, self.SUBFOLDERS[0], "*_0.jpg")
        cloudy_1_rgb_pathname = os.path.join(
            root, self.SUBFOLDERS[0], "*_1.jpg")
        cloudy_2_rgb_pathname = os.path.join(
            root, self.SUBFOLDERS[0], "*_2.jpg")

        cloudy_0_ir_pathname = os.path.join(
            root, self.SUBFOLDERS[0], "*_0_ir.jpg")
        cloudy_1_ir_pathname = os.path.join(
            root, self.SUBFOLDERS[0], "*_1_ir.jpg")
        cloudy_2_ir_pathname = os.path.join(
            root, self.SUBFOLDERS[0], "*_2_ir.jpg")

        clear_pathname = os.path.join(root, self.SUBFOLDERS[1], "*")

        self.cloudy_0_rgb_paths = sorted(glob.glob(cloudy_0_rgb_pathname))
        self.cloudy_1_rgb_paths = sorted(glob.glob(cloudy_1_rgb_pathname))
        self.cloudy_2_rgb_paths = sorted(glob.glob(cloudy_2_rgb_pathname))

        self.cloudy_0_ir_paths = sorted(glob.glob(cloudy_0_ir_pathname))
        self.cloudy_1_ir_paths = sorted(glob.glob(cloudy_1_ir_pathname))
        self.cloudy_2_ir_paths = sorted(glob.glob(cloudy_2_ir_pathname))

        self.clear_paths = sorted(glob.glob(clear_pathname))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        cloudy_0_rgb = Image.open(
            self.cloudy_0_rgb_paths[index]).convert('RGB')
        cloudy_1_rgb = Image.open(
            self.cloudy_1_rgb_paths[index]).convert('RGB')
        cloudy_2_rgb = Image.open(
            self.cloudy_2_rgb_paths[index]).convert('RGB')

        if self.band == 4:
            cloudy_0_ir = Image.open(
                self.cloudy_0_ir_paths[index]).convert('RGB')
            cloudy_1_ir = Image.open(
                self.cloudy_1_ir_paths[index]).convert('RGB')
            cloudy_2_ir = Image.open(
                self.cloudy_2_ir_paths[index]).convert('RGB')
        else:
            pass

        clear = Image.open(self.clear_paths[index]).convert('RGB')

        params = get_params(size=clear.size)
        transform_params = get_transform(self.band, params)

        cloudy_0_rgb_tensor = transform_params(cloudy_0_rgb)
        cloudy_1_rgb_tensor = transform_params(cloudy_1_rgb)
        cloudy_2_rgb_tensor = transform_params(cloudy_2_rgb)
        if self.band == 4:
            cloudy_0_ir_tensor = transform_params(cloudy_0_ir)[:1, ...]
            cloudy_1_ir_tensor = transform_params(cloudy_1_ir)[:1, ...]
            cloudy_2_ir_tensor = transform_params(cloudy_2_ir)[:1, ...]
        else:
            cloudy_0_ir_tensor = None
            cloudy_1_ir_tensor = None
            cloudy_2_ir_tensor = None
        clear_tensor = transform_params(clear)

        cloudy_0 = torch.cat(
            [i for i in [cloudy_0_rgb_tensor, cloudy_0_ir_tensor] if i is not None])
        cloudy_1 = torch.cat(
            [i for i in [cloudy_1_rgb_tensor, cloudy_1_ir_tensor] if i is not None])
        cloudy_2 = torch.cat(
            [i for i in [cloudy_2_rgb_tensor, cloudy_2_ir_tensor] if i is not None])

        clear = clear_tensor

        ret = {}
        ret['gt_image'] = clear
        ret['cond_image'] = torch.cat([cloudy_0.unsqueeze(0), cloudy_1.unsqueeze(0),
                                       cloudy_2.unsqueeze(0)], dim=0)
        ret['path'] = self.clear_paths[index].replace("jpg",'png')

        return ret

    def __len__(self) -> int:
        return len(self.clear_paths)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    for item in MultipleDataset(root="F:\多时相云修复数据集\multipleImage", band=3):
        print(item["cond_image"].shape, item["cond_image"].dtype)
        print(item["gt_image"].shape, item["gt_image"].dtype)
        print(item["path"].replace("\\", "/"))

        plt.figure(figsize=(8, 32), dpi=300)

        plt.subplot(1, 4, 1)
        plt.title("cond_image_0")
        plt.imshow(item["cond_image"][0, ...].permute(1, 2, 0)*0.5+0.5)

        plt.subplot(1, 4, 2)
        plt.title("cond_image_1")
        plt.imshow(item["cond_image"][1, ...].permute(1, 2, 0)*0.5+0.5)

        plt.subplot(1, 4, 3)
        plt.title("cond_image_2")
        plt.imshow(item["cond_image"][2, ...].permute(1, 2, 0)*0.5+0.5)

        plt.subplot(1, 4, 4)
        plt.title("gt_image")
        plt.imshow(item["gt_image"].permute(1, 2, 0)*0.5+0.5)

        plt.savefig("paired.png", bbox_inches="tight")

        break

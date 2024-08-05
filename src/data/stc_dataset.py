import os
from PIL import Image
import cv2
import numpy as np
import torch
from torchvision.io import read_video, write_jpeg
from torch.utils.data import Dataset
from torchvision import transforms as T

from data.noise import Simplex_CLASS

__all__ = "StcDataset"

STC_CLASS_NAMES = [
    "01",
    "02",
    "03",
    "04",
    "05",
    "06",
    "07",
    "08",
    "09",
    "10",
    "11",
    "12",
]  # , '13' - no ground-truth]


class ToTensor(object):
    def __call__(self, image):
        try:
            image = torch.from_numpy(image.transpose(2, 0, 1))
        except:
            print(
                "Invalid_transpose, please make sure images have shape (H, W, C) before transposing"
            )
        if not isinstance(image, torch.FloatTensor):
            image = image.float()
        return image


class Normalize(object):
    """
    Only normalize images
    """

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, image):
        image = (image - self.mean) / self.std
        return image


def get_data_transforms(size, isize):
    data_transforms = T.Compose([Normalize(), ToTensor()])
    gt_transforms = T.Compose([T.Resize((size, size)), T.ToTensor()])
    return data_transforms, gt_transforms


class StcDataset_train(Dataset):
    def __init__(self, c, transform):
        assert c.classes in STC_CLASS_NAMES, "class_name: {}, should be in {}".format(
            c.classes, STC_CLASS_NAMES
        )
        self.class_name = c.classes
        self.cropsize = c.crp_size
        #
        self.dataset_path = os.path.join(c.data_path, "training")
        self.dataset_vid = os.path.join(self.dataset_path, "videos")
        self.dataset_dir = os.path.join(self.dataset_path, "frames")
        self.dataset_files = sorted(
            [f for f in os.listdir(self.dataset_vid) if f.startswith(self.class_name)]
        )
        if not os.path.isdir(self.dataset_dir):
            os.mkdir(self.dataset_dir)
        done_file = os.path.join(
            self.dataset_path, "frames_{}.pt".format(self.class_name)
        )
        print(done_file)
        H, W = 480, 856
        if os.path.isfile(done_file):
            assert torch.load(done_file) == len(
                self.dataset_files
            ), "train frames are not processed!"
        else:
            count = 0
            for dataset_file in self.dataset_files:
                print(dataset_file)
                data = read_video(
                    os.path.join(self.dataset_vid, dataset_file)
                )  # read video file entirely -> mem issue!!!
                vid = data[
                    0
                ]  # weird read_video that returns byte tensor in format [T,H,W,C]
                fps = data[2]["video_fps"]
                print(
                    "video mu/std: {}/{} {}".format(
                        torch.mean(vid / 255.0, (0, 1, 2)),
                        torch.std(vid / 255.0, (0, 1, 2)),
                        vid.shape,
                    )
                )
                assert [H, W] == [vid.size(1), vid.size(2)], "same H/W"
                dataset_file_dir = os.path.join(
                    self.dataset_dir, os.path.splitext(dataset_file)[0]
                )
                os.mkdir(dataset_file_dir)
                count = count + 1
                for i, frame in enumerate(vid):
                    filename = "{0:08d}.jpg".format(i)
                    write_jpeg(
                        frame.permute((2, 0, 1)),
                        os.path.join(dataset_file_dir, filename),
                        80,
                    )
            torch.save(torch.tensor(count), done_file)
            #
        self.x, self.y, self.mask = self.load_dataset_folder()

        # Simplex Noise
        self.simplexNoise = Simplex_CLASS()
        # set transforms
        # self.transform = T.Compose(
        #     [
        #         T.Resize(c.image_size, Image.ANTIALIAS),
        #         T.RandomRotation(5),
        #         T.CenterCrop(c.crp_size),
        #         T.ToTensor(),
        #     ]
        # )
        self.transform = transform
        # self.normalize = T.Compose([T.Normalize(c.norm_mean, c.norm_std)])

    def __getitem__(self, idx):
        img_path, _, _ = self.x[idx], self.y[idx], self.mask[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img / 255.0, (256, 256))
        ## Normal
        img_normal = self.transform(img)
        ## Simplex Noise
        size = 256
        h_noise = np.random.randint(10, int(size // 8))
        w_noise = np.random.randint(10, int(size // 8))
        start_h_noise = np.random.randint(1, size - h_noise)
        start_w_noise = np.random.randint(1, size - w_noise)
        noise_size = (h_noise, w_noise)
        simplex_noise = self.simplexNoise.rand_3d_octaves((3, *noise_size), 6, 0.6)
        init_zero = np.zeros((256, 256, 3))
        init_zero[
            start_h_noise : start_h_noise + h_noise,
            start_w_noise : start_w_noise + w_noise,
            :,
        ] = 0.2 * simplex_noise.transpose(1, 2, 0)
        img_noise = img + init_zero
        img_noise = self.transform(img_noise)
        return img_normal, img_noise, img_path.split("/")[-1]

        # x = Image.open(x).convert("RGB")
        # x = self.normalize(self.transform_x(x))
        # mask = torch.zeros([1, self.cropsize[0], self.cropsize[1]])
        # #
        # return x, y, mask

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        x, y, mask = list(), list(), list()
        img_dir = os.path.join(self.dataset_path, "frames")
        img_types = sorted(
            [f for f in os.listdir(img_dir) if f.startswith(self.class_name)]
        )
        for i, img_type in enumerate(img_types):
            print("Folder:", img_type)
            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            img_fpath_list = sorted(
                [
                    os.path.join(img_type_dir, f)
                    for f in os.listdir(img_type_dir)
                    if f.endswith(".jpg")
                ]
            )
            x.extend(img_fpath_list)
            # labels for every test image
            mask.extend([None] * len(img_fpath_list))
            y.extend([0] * len(img_fpath_list))
        #
        return list(x), list(y), list(mask)


class StcDataset_test(Dataset):
    def __init__(self, c, transform, mask_transform):
        assert c.classes in STC_CLASS_NAMES, "class_name: {}, should be in {}".format(
            c.classes, STC_CLASS_NAMES
        )
        self.class_name = c.classes
        self.cropsize = c.crp_size
        self.dataset_path = os.path.join(c.data_path, "testing")
        self.x, self.y, self.mask = self.load_dataset_folder()
        # self.transform = T.Compose(
        #     [
        #         T.Resize(c.image_size, Image.ANTIALIAS),
        #         T.CenterCrop(c.crp_size),
        #         T.ToTensor(),
        #     ]
        # )
        # # mask
        # self.transform_mask = T.Compose(
        #     [
        #         T.ToPILImage(),
        #         T.Resize(c.image_size, Image.NEAREST),
        #         T.CenterCrop(c.crp_size),
        #         T.ToTensor(),
        #     ]
        # )
        self.simplexNoise = Simplex_CLASS()
        self.transform = transform
        self.mask_transform = mask_transform
        # self.normalize = T.Compose([T.Normalize(c.norm_mean, c.norm_std)])

    def __getitem__(self, idx):
        img_path, label, mask = self.x[idx], self.y[idx], self.mask[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img / 255.0, (256, 256))
        ## Normal
        img = self.transform(img)
        ## Simplexe Noise
        mask = Image.fromarray(mask * 255)
        mask = self.mask_transform(mask)
        mask = (mask > 0.5).int()
        img_type = "anomaly"
        return img, mask, label, img_type, img_path.split("/")[-1]
        # x = Image.open(x).convert("RGB")
        # x = self.normalize(self.transform(x))
        # mask = self.transform_mask(mask)
        # #
        # return x, y, mask

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        x, y, mask = list(), list(), list()
        img_dir = os.path.join(self.dataset_path, "frames")
        img_types = sorted(
            [f for f in os.listdir(img_dir) if f.startswith(self.class_name)]
        )
        gt_frame_dir = os.path.join(self.dataset_path, "test_frame_mask")
        gt_pixel_dir = os.path.join(self.dataset_path, "test_pixel_mask")
        for i, img_type in enumerate(img_types):
            print("Folder:", img_type)
            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            img_fpath_list = sorted(
                [
                    os.path.join(img_type_dir, f)
                    for f in os.listdir(img_type_dir)
                    if f.endswith(".jpg")
                ]
            )
            x.extend(img_fpath_list)
            # labels for every test image
            gt_pixel = np.load("{}.npy".format(os.path.join(gt_pixel_dir, img_type)))
            gt_frame = np.load("{}.npy".format(os.path.join(gt_frame_dir, img_type)))
            if i == 0:
                m = gt_pixel
                y = gt_frame
            else:
                m = np.concatenate((m, gt_pixel), axis=0)
                y = np.concatenate((y, gt_frame), axis=0)
            #
            mask = [e for e in m]  # np.expand_dims(e, axis=0)
            assert len(x) == len(y), "number of x and y should be same"
            assert len(x) == len(mask), "number of x and mask should be same"
        #
        return list(x), list(y), list(mask)

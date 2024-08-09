import os
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
import torch
from torchvision.io import read_video, write_jpeg
from torch.utils.data import Dataset
from torchvision import transforms as T

from data.noise import SimplexNoise

__all__ = "RailsDataset"


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


class RailsTrainDataset(Dataset):
    def __init__(self, c, transform):
        self.training_dataset_pth = Path(
            "/home/mmhamdi/workspace/unsupervised/Unsupervised-Anomlay-Detection/rails/training"
        )
        self.training_videos_pth = self.training_dataset_pth / "Videos"
        self.training_frames_pth = self.training_dataset_pth / "Frames"
        self.processed_frames = self.training_dataset_pth / "frames.pt"
        self.training_frames_pth.mkdir(parents=True, exist_ok=True)
        self.load_videos()
        self.load_annotations()
        self.simplexNoise = SimplexNoise()
        self.transform = transform

    def __getitem__(self, idx):
        img_pth, _, _ = self.imgs[idx], self.labels[idx], self.masks[idx]
        img = cv2.imread(img_pth)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img / 255.0, (256, 256))
        img_normal = self.transform(img)
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
        return img_normal, img_noise, img_pth.split("/")[-1]

    def __len__(self):
        return len(self.imgs)

    def load_videos(self):
        count = 0
        for video_pth in self.training_videos_pth.iterdir():
            data = read_video(video_pth.as_posix())
            video = data[0]
            fps = data[2]["video_fps"]
            for idx, frame in enumerate(video):
                filename = "{0:08d}.jpg".format(idx)
                write_jpeg(
                    frame.permute((2, 0, 1)),
                    (self.training_frames_pth / filename).as_posix(),
                    80,
                )
            count += 1
        torch.save(torch.tensor(count), self.processed_frames)

    def load_annotations(self):
        self.imgs = sorted(list(map(str, list(self.training_frames_pth.iterdir()))))
        self.masks = [None] * len(self.imgs)
        self.labels = [0] * len(self.imgs)


class RailsTestDataset(Dataset):
    def __init__(self, c, transform, mask_transform):
        self.testing_dataset_pth = Path(
            "/home/mmhamdi/workspace/unsupervised/Unsupervised-Anomlay-Detection/rails/testing"
        )
        self.testing_frames_pth = self.testing_dataset_pth / "Frames"
        self.load_annotations()
        self.input_shape = (480, 480)
        self.simplexNoise = SimplexNoise()
        self.transform = transform
        self.mask_transform = mask_transform

    def __getitem__(self, idx, crop_roi=False):
        img_path, label, mask = self.imgs[idx], self.labels[idx], self.masks[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if crop_roi:
            margin = 500
            img = img[1000 - margin - 100 : 1000 + margin, 500:]
        img = cv2.resize(img / 255.0, self.input_shape)
        mask = cv2.resize(mask, self.input_shape)
        ## Normal
        img = self.transform(img)
        # mask = Image.fromarray(mask * 255)
        # mask = self.mask_transform(mask)
        # mask = (mask > 0.5).int()
        mask[mask > 0] = 255
        mask = torch.from_numpy(mask // 255)
        img_type = "rail"
        return img, mask.unsqueeze(0), label, img_type, img_path.split("/")[-1]

    def __len__(self):
        return len(self.imgs)

    def load_annotations(self):
        self.imgs, self.labels, self.masks = list(), list(), list()
        self.testing_frames = sorted(list(self.testing_frames_pth.iterdir()))
        self.testing_masks_pth = self.testing_dataset_pth / "SegmentationClass"
        self.testing_masks = list(self.testing_masks_pth.iterdir())
        frames_indices = [
            str(name).split(".")[0].split("_")[-1] for name in self.testing_frames
        ]
        masks_indices = [
            str(name).split(".")[0].split("_")[-1] for name in self.testing_masks
        ]
        negative_masks = [
            self.testing_masks_pth / f"mask_{idx}.png"
            for idx in frames_indices
            if idx not in masks_indices
        ]
        self.testing_masks.extend(negative_masks)
        self.testing_masks = sorted(self.testing_masks)
        for idx, mask_pth in enumerate(self.testing_masks):
            if mask_pth.is_file():
                mask = cv2.imread(str(mask_pth), 0)
                mask[mask > 0] = 255
                self.masks.append(mask)
                self.labels.append(1)
                self.imgs.append(str(self.testing_frames[idx]))
            else:
                self.masks.append(np.zeros((1080, 1920)))
                self.labels.append(0)
                self.imgs.append(str(self.testing_frames[idx]))

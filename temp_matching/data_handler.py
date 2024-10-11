import numpy as np
from typing import Tuple, Optional, Dict, Any
import torch
import json
import os
import cv2
from torch.utils.data import Dataset
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
import albumentations as A
from enum import Enum
import warnings


def normalization(x):
    return x / 255


def denormalization(x):
    return x * 255


class DataConfig(BaseModel):
    data_root: str
    annotation_path: str
    read_all_data: bool = True
    # w,h
    image_size: Tuple[int, int] = (224, 224)
    min_query_hw: int = 32
    # augmentation
    aug_rate: float = 0.5
    rotate_anlge: Tuple[int, int] = -10, 10
    flip_rate: float = 0.5
    # after rotation, original region will be black, fill it by what was there before?
    fill_blank_by_original: bool = True

    random_seed: int = 100
    max_data: int = -1
    train_size: float = 0.9
    wrong_query_rate: float = 0.5


class DataType(Enum):
    TRAIN = "train"
    VALID = "valid"


class CustomCocoDataset(Dataset):
    def __init__(
        self,
        config: DataConfig,
        normalization: callable = normalization,
        denormalization: callable = denormalization,
        data_type: DataType = DataType.TRAIN,
        image_augmentation: Optional[A.Compose] = A.Compose(
            [
                A.Blur(),
                A.ColorJitter(),
                A.GaussNoise(),
                A.Emboss(),
                A.MultiplicativeNoise(multiplier=(0.97, 1.03)),
            ],
            p=0.5,
        ),
    ):
        """
        Custom Dataset to load images and their respective bounding boxes.

        Args:
            config (DataConfig): Configuration object containing dataset paths and parameters.
        """
        self.config = config
        self.data_root = config.data_root
        self.image_size = config.image_size
        self.min_query_hw = config.min_query_hw
        self.data_type = data_type
        self.random_state = np.random.RandomState(config.random_seed)
        self.normalization = normalization
        self.denormalization = denormalization
        self.image_augmentation = image_augmentation
        # Load the annotation file and create the bounding box dictionary
        self.bbox_dict = self._load_annotations(config.annotation_path)
        bbox_keys = list(self.bbox_dict.keys())
        if self.config.train_size < 1:
            train_keys, valid_keys = train_test_split(
                bbox_keys,
                train_size=self.config.train_size,
                random_state=self.random_state,
            )
            if self.data_type == DataType.TRAIN:
                self.keys = train_keys
            else:
                self.keys = valid_keys
        else:
            self.keys = bbox_keys
        print(f"Data for {data_type}: {len(self)}")

    def _load_annotations(self, annotation_path: str) -> Dict[int, Tuple[Any, str]]:
        """
        Reads the annotation file and extracts bounding boxes with image paths,
        filtering out boxes smaller than min_query_hw.

        Args:
            annotation_path (str): Path to the annotation JSON file.

        Returns:
            bbox_dict (Dict[int, Tuple[Any, str]]): A dictionary mapping index to bounding box and image path.
        """
        with open(annotation_path, "r") as f:
            annotations = json.load(f)

        bbox_dict = {}
        total_annotations = len(annotations["annotations"])
        print(f"Total annotations: {total_annotations}")
        selected_annotations = 0

        # Assuming COCO-style annotation structure
        for ann in annotations["annotations"]:

            bbox = ann["bbox"]  # Get the bounding box coordinates [x, y, width, height]
            x, y, w, h = map(int, bbox)

            # Check if bounding box meets the minimum size requirement
            if w >= self.min_query_hw and h >= self.min_query_hw:
                image_id = ann["image_id"]

                # Find the corresponding image file name
                image_info = next(
                    (img for img in annotations["images"] if img["id"] == image_id),
                    None,
                )

                if image_info:
                    image_path = os.path.join(self.data_root, image_info["file_name"])
                    # bbox_dict[idx] = (bbox, image_path)
                    if image_path not in bbox_dict:
                        bbox_dict[image_path] = []
                    bbox_dict[image_path].append(bbox)
                    selected_annotations += 1
            if (
                self.config.max_data > 0
                and len(bbox_dict) > self.config.max_data
                and not self.config.read_all_data
            ):
                break
        print(
            f"Total images: {len(bbox_dict)}, Num. Annotations: {selected_annotations}"
        )
        return bbox_dict

    def __len__(self):
        """Returns the total number of bounding boxes."""
        return (
            len(self.keys)
            if self.config.max_data < 0
            else min(len(self.keys), self.config.max_data)
        )

    def get_query_label(self, idx):
        image_path = self.keys[idx]
        bboxes = self.bbox_dict.get(image_path)

        if not os.path.exists(image_path) or bboxes is None:
            warnings.warn(f"Image not found: {image_path}, skipping.")
            return self.get_query_label(self.random_state.randint(0, len(self)))

        # Read the image using OpenCV
        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        if len(bboxes) == 0:
            return (
                image,
                np.zeros_like(image),
                np.zeros(image.shape[:2], dtype=np.uint8),
            )
        # Randomly select a bounding box
        bbox = bboxes[self.random_state.randint(0, len(bboxes))]
        x, y, w, h = map(int, bbox)

        # Create a mask of the same size as the original image, initialized to zeros
        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # Mark the area of the bounding box in the mask
        mask[y : y + h, x : x + w] = 255

        # Crop the region of the bounding box from the original image
        cropped_region = image[y : y + h, x : x + w]

        # Create a black image of the same size as the original
        query = np.zeros_like(image)

        # Calculate center positions to place the cropped region
        centered_x = (query.shape[1] - w) // 2
        centered_y = (query.shape[0] - h) // 2

        # Place the cropped region in the center of the black image
        query[centered_y : centered_y + h, centered_x : centered_x + w] = cropped_region

        # Convert the images to RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        query = cv2.cvtColor(query, cv2.COLOR_BGR2RGB)

        # Resize the images to the specified size
        image = cv2.resize(image, self.image_size)
        oimage = image.copy()
        query = cv2.resize(query, self.image_size)
        mask = cv2.resize(mask, self.image_size)

        # apply augmentation
        if self.random_state.uniform() < self.config.aug_rate:
            angle = self.random_state.uniform(
                self.config.rotate_anlge[0], self.config.rotate_anlge[1]
            )
            rotation_matrix = cv2.getRotationMatrix2D(
                (query.shape[1] // 2, query.shape[0] // 2), angle, 1
            )
            # query = cv2.warpAffine(query, rotation_matrix, (query.shape[1], query.shape[0]))
            image = cv2.warpAffine(
                image, rotation_matrix, (image.shape[1], image.shape[0])
            )
            mask = cv2.warpAffine(mask, rotation_matrix, (mask.shape[1], mask.shape[0]))
            # print(f"Angle: {angle}")

            if self.random_state.uniform() < self.config.flip_rate:
                query = cv2.flip(query, 1)
                image = cv2.flip(image, 1)
                mask = cv2.flip(mask, 1)
            if self.config.fill_blank_by_original:
                immask = (image == [0, 0, 0]).sum(axis=2) == 3
                image[immask] = oimage[immask]

        return image, query, mask

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fetches the original image, the query (centered cropped region), and the mask.

        Args:
            index (int): The index of the bounding box.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Original image, query (centered cropped region), and mask.
        """
        if index == len(self) - 1:
            self.random_state.shuffle(self.keys)
        image, query, mask = self.get_query_label(index)

        if self.random_state.uniform() < self.config.wrong_query_rate:
            # find an index thats not equal to current one
            while True:
                nidx = self.random_state.randint(len(self))
                if nidx != index:
                    img2, query, mask2 = self.get_query_label(nidx)
                    break

            mask = np.zeros_like(mask)

        if self.image_augmentation:
            augmented = self.image_augmentation(image=image)
            image = augmented["image"]

        image_query = np.concatenate((image, query)).reshape(2, *image.shape)

        image = self.normalization(image)
        query = self.normalization(query)
        mask = mask / 255
        image_query = self.normalization(image_query)

        image_query_tensor = (
            torch.from_numpy(image_query).permute(0, 3, 1, 2).to(torch.float32)
        )
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(torch.float32)

        return image_query_tensor, mask_tensor

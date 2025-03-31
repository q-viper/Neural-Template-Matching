import numpy as np
from typing import Tuple, Dict, Any
import json
import os
import cv2
from torch.utils.data import Dataset
from pydantic import BaseModel


def normalization(x):
    return x / 255


def denormalization(x):
    return x * 255


class DataConfig(BaseModel):
    data_root: str = r"D:/MSc Works/temp_matching/assets/training_data/val2017/val2017"
    annotation_path: str = (
        r"D:/MSc Works/temp_matching/assets/training_data/annotations_trainval2017/annotations/instances_val2017.json"
    )
    read_all_data: bool = True
    # w,h
    image_size: Tuple[int, int] = (224, 224)
    max_data: int = -1
    min_query_hw: int = 32


class CustomCocoDataset(Dataset):
    def __init__(
        self,
        config: DataConfig,
        normalization: callable = normalization,
        denormalization: callable = denormalization,
    ):
        """
        Custom Dataset to load images and their respective bounding boxes.

        Args:
            config (DataConfig): Configuration object containing dataset paths and parameters.
        """
        self.config = config
        self.data_root = config.data_root
        self.image_size = config.image_size
        self.normalization = normalization
        self.denormalization = denormalization
        # Load the annotation file and create the bounding box dictionary
        self.bbox_dict = self._load_annotations(config.annotation_path)
        bbox_keys = list(self.bbox_dict.keys())
        if config.max_data > 0:
            bbox_keys = bbox_keys[: config.max_data]
        self.keys = bbox_keys

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
        categories = {k["id"]: k["name"] for k in annotations["categories"]}
        bbox_dict = {}
        total_annotations = len(annotations["annotations"])
        print(f"Total annotations: {total_annotations}")
        selected_annotations = 0

        # Assuming COCO-style annotation structure
        for ann in annotations["annotations"]:
            bbox = ann["bbox"]  # Get the bounding box coordinates [x, y, width, height]
            x, y, w, h = map(int, bbox)
            lbl = categories[int(ann["category_id"])]
            ann_id = ann["id"]

            if (w >= self.config.min_query_hw) and (h >= self.config.min_query_hw):
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
                    bbox_dict[image_path].append([bbox, lbl, ann_id])
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
            raise FileNotFoundError(f"Image not found: {image_path}")
            # return self.get_query_label(self.random_state.randint(0, len(self)))

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

        pairs = []
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for bbox, lbl, ann_id in bboxes:
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
            query[centered_y : centered_y + h, centered_x : centered_x + w] = (
                cropped_region
            )

            pairs.append(
                (
                    image,
                    query,
                    mask,
                    cropped_region,
                    lbl,
                    image_path,
                    (x, y, w, h),
                    ann_id,
                )
            )

        return pairs

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fetches the original image, the query (centered cropped region), and the mask.

        Args:
            index (int): The index of the bounding box.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Original image, query (centered cropped region), and mask.
        """
        pairs = self.get_query_label(index)
        return pairs

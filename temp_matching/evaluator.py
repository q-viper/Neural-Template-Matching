import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Tuple

# else cant load model
from temp_matching.model import CustomUnet, EncodingCombination


def normalization(x):
    return x / 255


def denormalization(x):
    return x * 255


class Evaluator:
    def __init__(
        self,
        model_path: Path = Path(r"train_res\2024-09-24\best_model.pth"),
        is_state_dict: bool = True,
        model_args: dict = dict(
            unet_args={
                "encoder_name": "resnet34",
                # 'encoder_weights': 'imagenet',
                "classes": 1,  # Number of output classes
                "activation": "sigmoid",  # Activation function
                "in_channels": 3,  # Number of input channels
            },
            encoding_combination=EncodingCombination.MULTIPLICATION,
        ),
        device: str = "cpu",
        input_size: Tuple[int, int] = (512, 512),
    ):
        self.device = (
            device if torch.cuda.is_available() and ("cuda" in device) else "cpu"
        )
        if not is_state_dict:
            self.model = torch.load(
                model_path,
            )
        else:
            self.model = CustomUnet(**model_args)
            self.model.load_state_dict(
                torch.load(
                    model_path,
                ),
            )
        self.model.to(self.device)
        self.model.eval()
        self.query = None
        self.input_size = input_size
        self.query_embeddings = None

    def set_query(self, query: np.ndarray, frame: np.ndarray):
        # Create a black image of the same size as the original
        nquery = np.zeros_like(frame)
        h, w = query.shape[:2]

        # Calculate center positions to place the cropped region
        centered_x = (nquery.shape[1] - w) // 2
        centered_y = (nquery.shape[0] - h) // 2
        nquery[centered_y : centered_y + h, centered_x : centered_x + w] = query
        self.query = cv2.resize(nquery, self.input_size)

        norm_query = normalization(self.query)
        norm_query_tensor = (
            torch.from_numpy(norm_query).permute(2, 0, 1).to(torch.float32)
        )
        self.query_embeddings = self.model.query_encoder(
            norm_query_tensor.unsqueeze(0).to(self.device)
        )

    @torch.no_grad()
    def fast_predict(self, image: np.ndarray):
        if self.query_embeddings is None:
            raise ValueError("Query embeddings not set. Please set the query first.")
        image = cv2.resize(image, self.input_size)
        self.image = image
        norm_image = normalization(self.image)
        norm_image_tensor = (
            torch.from_numpy(norm_image).permute(2, 0, 1).to(torch.float32)
        )
        image_embeddings = self.model.image_encoder(
            norm_image_tensor.unsqueeze(0).to(self.device)
        )
        combined_features = self.model.combine_features(
            image_embeddings, self.query_embeddings
        )
        masks = self.model.get_masks(combined_features)
        return masks.cpu().squeeze(0, 1).numpy()

    def predict(self, image: np.ndarray):
        with torch.no_grad():
            image = cv2.resize(image, self.input_size)
            self.image = image
            image_query = np.concatenate([image, self.query]).reshape(2, *image.shape)
            image_query = normalization(image_query)
            image_query_tensor = (
                torch.from_numpy(image_query).permute(0, 3, 1, 2).to(torch.float32)
            )
            image_tensor = image_query_tensor.unsqueeze(0).to(self.device)
            output = self.model(image_tensor)
            # TODO: fix this else batch prediction wont work
            return output.cpu().squeeze(0, 1).numpy()

    def post_process(self, output: np.ndarray):
        output = output > 0.5
        return output.astype(np.uint8)

    def overlay_mask(self, image: np.ndarray, mask: np.ndarray, color=[0, 255, 0]):
        # Resize the mask to match the image size
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

        # Ensure the mask is binary (0 or 1)
        mask = (mask > 0).astype(np.uint8)

        # Convert the mask to a 3-channel mask
        mask_3d = np.stack([mask] * 3, axis=-1)

        # Create a colored version of the mask
        colored_mask = mask_3d * np.array(color, dtype=np.uint8)

        # Copy the original image to overlay the mask
        overlayed = image.copy()

        # Apply cv2.addWeighted only to the masked region
        overlayed_mask_region = cv2.addWeighted(image, 0.5, colored_mask, 0.5, 0)

        # Assign the blended region back to the masked area of the image
        overlayed[mask_3d == 1] = overlayed_mask_region[mask_3d == 1]

        return overlayed

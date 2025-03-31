import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Optional
import sys

sys.path.append(r"D:\MSc Works\temp_matching")
# else cant load model
from temp_matching.model import CustomUnet, EncodingCombination
from pydantic import BaseModel
import time
import psutil


class ModelPrediction(BaseModel):
    out: np.ndarray
    memory: float
    time: float
    gpu_memory: float

    class Config:
        arbitrary_types_allowed = True


def normalization(x):
    return x / 255


def denormalization(x):
    return x * 255


def get_sift_match_mask(original_image, template_query, min_matches=4):
    """
    Matches a template to an original image using SIFT and returns a mask of the matched region.

    Parameters:
        original_image (numpy.ndarray): The original image where the template is to be matched.
        template_query (numpy.ndarray): The template image to be matched.
        min_matches (int): Minimum number of matches required to consider it a valid match.
        threshold (float): Threshold for considering a match in template matching (default is 0.8).

    Returns:
        numpy.ndarray: A binary mask of the same size as the original image, with the matched region marked as 1.
                      If no match is found, returns a zeroed mask.
    """
    # Convert images to grayscale if necessary
    gray_original = (
        cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        if len(original_image.shape) == 3
        else original_image
    )
    gray_template = (
        cv2.cvtColor(template_query, cv2.COLOR_BGR2GRAY)
        if len(template_query.shape) == 3
        else template_query
    )

    # Initialize the SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    kp1, des1 = sift.detectAndCompute(gray_original, None)
    kp2, des2 = sift.detectAndCompute(gray_template, None)

    if des1 is None or des2 is None:
        # print("No keypoints detected in one of the images!")
        return np.zeros(
            (original_image.shape[0], original_image.shape[1]), dtype=np.uint8
        )

    # Use FLANN-based matcher
    index_params = dict(algorithm=1, trees=5)  # KDTree algorithm for faster search
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Match descriptors using KNN
    matches = flann.knnMatch(des2, des1, k=2)

    # Apply Lowe's ratio test to find good matches
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good_matches) < min_matches:
        # print(f"Not enough good matches found: {len(good_matches)}")
        return np.zeros(
            (original_image.shape[0], original_image.shape[1]), dtype=np.uint8
        )

    # Get the coordinates of the matched points
    src_pts = np.float32([kp2[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Compute the homography using RANSAC
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if H is None:
        # print("Homography computation failed!")
        return np.zeros(
            (original_image.shape[0], original_image.shape[1]), dtype=np.uint8
        )

    # Get the height and width of the template
    h, w = template_query.shape[:2]
    template_mask = np.ones(
        (h, w), dtype=np.uint8
    )  # Create a mask the size of the template

    # Warp the template mask to the perspective of the original image
    warped_mask = cv2.warpPerspective(
        template_mask, H, (original_image.shape[1], original_image.shape[0])
    )

    # Create a final binary mask from the warped mask
    final_mask = np.uint8(warped_mask > 0) * 255

    return final_mask


class Evaluator:
    def __init__(
        self,
        model_path: Path = Path(r"train_res\2024-09-24\best_model_state_dict.pth"),
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
        device: str = "cuda",
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
        self.frame_embeddings = None

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

    def set_frame(self, frame: np.ndarray):
        self.frame = frame
        self.frame = cv2.resize(self.frame, self.input_size)
        norm_frame = normalization(self.frame)
        norm_frame_tensor = (
            torch.from_numpy(norm_frame).permute(2, 0, 1).to(torch.float32)
        )
        self.image_embeddings = self.model.image_encoder(
            norm_frame_tensor.unsqueeze(0).to(self.device)
        )

    @torch.no_grad()
    def fast_predict(
        self, image: Optional[np.ndarray] = None, query: Optional[np.ndarray] = None
    ):
        if (image is None or query is None) and (
            self.query_embeddings is None or self.frame_embeddings is None
        ):
            raise ValueError(
                "Query/Frame embeddings not set. Please set the query/frame first."
            )
        if image is not None:
            self.set_frame(image)
        if query is not None:
            self.set_query(query, image)

        combined_features = self.model.combine_features(
            self.image_embeddings, self.query_embeddings
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

    def batch_predict(self, images: np.ndarray, queries: np.ndarray):
        with torch.no_grad():
            image_queries = np.array(
                [
                    np.concatenate(
                        (
                            normalization(cv2.resize(image, self.input_size)),
                            normalization(cv2.resize(query, self.input_size)),
                        )
                    ).reshape(2, *self.input_size, 3)
                    for image, query in zip(images, queries)
                ]
            )
            image_queries_tensor = (
                torch.from_numpy(image_queries).permute(0, 1, 4, 2, 3).to(torch.float32)
            )
            image_tensors = image_queries_tensor.to(self.device)
            t0 = time.perf_counter()
            output = self.model(image_tensors)
            memory = torch.cuda.memory_allocated("cuda") / 1e6
            t1 = time.perf_counter()
            ram_usage = psutil.virtual_memory()

            return ModelPrediction(
                out=output.cpu().squeeze(1).numpy(),
                time=t1 - t0,
                memory=ram_usage.used / 1e6,
                gpu_memory=memory,
            )

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


if __name__ == "__main__":
    from temp_matching.val_data_loader import CustomCocoDataset, DataConfig

    from tqdm import tqdm

    val_loader = CustomCocoDataset(DataConfig(max_data=-100))

    evaluator = Evaluator()
    frame = val_loader[1][0][0]
    query = val_loader[1][0][1]
    mask = val_loader[1][0][2]
    evaluator.set_query(query, frame)
    evaluator.set_frame(frame)

    out = evaluator.fast_predict(frame, query)
    out = evaluator.post_process(out)
    out2 = evaluator.post_process(evaluator.predict(frame))

    assert np.array_equal(out, out2)

    batch_size = 32
    images = []
    image_names = []
    cropped_regions = []
    queries = []
    masks = []
    lbls = []
    bboxes = []
    ann_ids = []
    results_dir = Path(r"D:\MSc Works\temp_matching\assets\model_results")
    out_csv = results_dir / "results.csv"
    if not results_dir.exists():
        results_dir.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w") as f:
        # 'image_name', 'label', 'bbox', 'iou', 'ann_id', 'model_time', 'model_gpu_memory', 'model_memory', 'sift_iou', 'sift_time'
        f.write(
            "image_name,label,bbox,model_iou,ann_id,model_time,model_gpu_memory,model_memory,sift_iou,sift_time\n"
        )

    pbar = tqdm(total=len(val_loader))
    for samples in val_loader:
        pbar.update(1)
        for sample in samples:
            image, query, mask, cropped_region, lbl, img_name, bbox, ann_id = sample
            pbar.set_description(f"{img_name}")

            images.append(image)
            cropped_regions.append(cropped_region)
            masks.append(mask)
            lbls.append(lbl)
            queries.append(query)
            image_names.append(img_name)
            bboxes.append(bbox)
            ann_ids.append(ann_id)

            if len(images) >= batch_size:
                model_pred = evaluator.batch_predict(images, queries)

                for i, out in enumerate(model_pred.out):
                    out = evaluator.post_process(out)
                    mask = cv2.resize(
                        masks[i],
                        out.shape[:2],
                    )
                    bmask = mask > 0
                    bout = out > 0
                    intersection = np.logical_and(bout, bmask).sum()
                    union = np.logical_or(bout, bmask).sum()
                    iou = intersection / union if union > 0 else 0
                    iou = round(iou, 3)
                    out = out.astype(np.uint8) * 255
                    # image_query_mask_out_iou = [
                    #     cv2.resize(images[i], out.shape[:2]),
                    #     cv2.resize(queries[i], out.shape[:2]),
                    #     cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
                    #     cv2.cvtColor(out * 255, cv2.COLOR_GRAY2BGR),
                    #     iou,
                    # ]

                    fname = (
                        results_dir
                        / f"{Path(image_names[i]).stem}_{ann_ids[i]}_model.npy"
                    )
                    np.save(fname, out)
                    t0 = time.perf_counter()
                    try:
                        sift_out = get_sift_match_mask(images[i], cropped_regions[i])
                    except:
                        sift_out = np.zeros_like(out)

                    sift_out = cv2.resize(sift_out, out.shape) > 0
                    smask = cv2.resize(masks[i], out.shape) > 0

                    t1 = time.perf_counter()
                    sift_time = t1 - t0
                    intersection = np.logical_and(sift_out, smask).sum()
                    union = np.logical_or(sift_out, smask).sum()
                    sift_iou = round(intersection / union, 3) if union > 0 else 0
                    fname = (
                        results_dir
                        / f"{Path(image_names[i]).stem}_{ann_ids[i]}_sift.npy"
                    )
                    # cv2.imwrite(
                    #     "res.png",
                    #     np.hstack([cv2.resize(sift_out, out.shape), out]),
                    # )
                    np.save(fname, sift_out)
                    with open(out_csv, "a") as f:
                        f.write(
                            f"{image_names[i]},{lbls[i]},{bboxes[i]},{iou},{ann_ids[i]},{model_pred.time/len(masks)},{model_pred.gpu_memory/len(masks)},{model_pred.memory/len(masks)},{sift_iou},{sift_time}\n"
                        )

                #     cv2.imwrite(
                #         "res.png",
                #         np.hstack(
                #             [
                #                 cv2.resize(images[i], out.shape[:2]),
                #                 cv2.resize(queries[i], out.shape[:2]),
                #                 cv2.resize(
                #                     cv2.cvtColor(masks[i], cv2.COLOR_GRAY2BGR),
                #                     out.shape[:2],
                #                 ),
                #                 cv2.cvtColor(out * 255, cv2.COLOR_GRAY2BGR),
                #             ]
                #         ),
                #     )
                #     print(i)
                # images = []
                cropped_regions = []
                queries = []
                masks = []
                lbls = []
                image_names = []
                bboxes = []
                ann_ids = []
    pbar.close()

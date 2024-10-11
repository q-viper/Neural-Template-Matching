import json
import os
import numpy as np
from vis import subplot_images
import datetime
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch.utils as smp_utils
from pathlib import Path
from model import CustomUnet, EncodingCombination
from data_handler import CustomCocoDataset, DataConfig, DataType
from dotenv import load_dotenv

if __name__ == "__main__":
    print("Running the trainer")
    load_dotenv(".env")
    TRAIN_DIR = os.environ.get("TRAIN_DIR")
    TRAIN_ANNOTATION_DIR = os.environ.get("TRAIN_ANNOTATION_DIR")
    VAL_DIR = os.environ.get("VAL_DIR")
    VAL_ANNOTATION_DIR = os.environ.get("VAL_ANNOTATION_DIR")
    OLD_MODEL_PATH = os.environ.get("OLD_MODEL_PATH")
    OLD_OPTIMIZER_PATH = os.environ.get("OLD_OPTIMIZER_PATH")
    READ_ALL_DATA = False

    BATCH_SIZE = 1
    INPUT_SIZE = (512, 512)
    MAX_TRAIN_DATA = 100
    MAX_VALID_DATA = 50
    VERBOSE = True
    EPOCHS = 300
    LOG_IMAGES_EVERY = 10
    OUT_DIR = "./train_res"
    ENCODER_NAME = "resnet152"
    UNET_ARGS = {
        "encoder_name": ENCODER_NAME,
        "classes": 1,  # Number of output classes
        "activation": "sigmoid",  # Activation function
        "in_channels": 3,  # Number of input channels
    }
    print(f"Training on {ENCODER_NAME} encoder")
    if TRAIN_DIR is None:
        raise ValueError("TRAIN_DIR is not set")

    train_dataset = CustomCocoDataset(
        config=DataConfig(
            data_root=TRAIN_DIR,
            annotation_path=TRAIN_ANNOTATION_DIR,
            read_all_data=READ_ALL_DATA,
            image_size=INPUT_SIZE,
            min_query_hw=32,
            aug_rate=0.5,
            rotate_anlge=(-30, 30),
            flip_rate=0.5,
            random_seed=1000,
            max_data=MAX_TRAIN_DATA,
            wrong_query_rate=0.2,
            train_size=1,
        )
    )
    valid_dataset = CustomCocoDataset(
        config=DataConfig(
            data_root=VAL_DIR,
            annotation_path=VAL_ANNOTATION_DIR,
            read_all_data=READ_ALL_DATA,
            image_size=INPUT_SIZE,
            min_query_hw=32,
            aug_rate=0.5,
            rotate_anlge=(-30, 30),
            flip_rate=0.5,
            random_seed=1000,
            max_data=MAX_VALID_DATA,
            wrong_query_rate=0.2,
            # train_size=0.9,
            data_type=DataType.VALID,
            train_size=1,
        )
    )
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    model = CustomUnet(
        unet_args=UNET_ARGS,
        encoding_combination=EncodingCombination.MULTIPLICATION,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using Device: {device}")

    loss = smp_utils.losses.DiceLoss()
    metrics = [
        smp_utils.metrics.IoU(threshold=0.5),
    ]

    optimizer = torch.optim.Adam(
        [
            dict(params=model.parameters(), lr=0.0001),
        ]
    )
    # create epoch runners
    # it is a simple loop of iterating over dataloader`s samples
    train_epoch = smp_utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=device,
        verbose=VERBOSE,
    )

    valid_epoch = smp_utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=device,
        verbose=VERBOSE,
    )

    max_score = 0
    out_dir = Path(OUT_DIR)
    expt_name = f"{ENCODER_NAME}_{str(datetime.datetime.now().date())}"
    out_dir = out_dir / expt_name

    print(f"Loding model from: {out_dir}")
    if Path(OLD_MODEL_PATH).exists():
        model_state = torch.load(OLD_MODEL_PATH)
        model.load_state_dict(model_state)
        model.to(device)
        print("Model loaded!")
    print(f"Loding optimizer from: {out_dir}")
    if Path(OLD_OPTIMIZER_PATH).exists():
        optimizer.load_state_dict(OLD_OPTIMIZER_PATH)
        print("Optimizer loaded!")

    logs = {}

    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    # write this file to the output directory
    with open(out_dir / "trainer.py", "w") as f:
        f.write(open(os.path.abspath(__file__)).read())

    for i in range(0, EPOCHS):
        print("\nEpoch: {}".format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        logs[i] = {f"train_{k}": v for k, v in train_logs.items()}
        logs[i].update({f"valid_{k}": v for k, v in valid_logs.items()})

        with open(out_dir / "logs.json", "w") as f:
            json.dump(logs, f, indent=4)

        print(logs[i])

        torch.save(model.state_dict(), str(out_dir / "last_model_state_dict.pth"))
        torch.save(optimizer.state_dict(), str(out_dir / "last_optimizer.pth"))

        if max_score < valid_logs["iou_score"]:
            max_score = valid_logs["iou_score"]
            torch.save(model, str(out_dir / "best_model.pth"))
            torch.save(model.state_dict(), str(out_dir / "best_model_state_dict.pth"))
            torch.save(optimizer.state_dict(), str(out_dir / "best_optimizer.pth"))
            print("Model saved!")

        if i == 25:
            optimizer.param_groups[0]["lr"] = 1e-5
            print("Decrease decoder learning rate to 1e-5!")
        if i % LOG_IMAGES_EVERY == 0 or i == EPOCHS - 1:
            idxs = np.random.choice(len(valid_dataset), 5)
            res = []
            titles = []
            for idx in idxs:
                imq, lbl = valid_dataset[idx]
                with torch.no_grad():
                    pred = model(imq.unsqueeze(0).to(device))
                    iou = smp_utils.metrics.IoU(threshold=0.5)
                    iou_score = iou(pred, lbl.unsqueeze(0).to(device)).item()

                    image = imq[0].permute(1, 2, 0).cpu().numpy()
                    query = imq[1].permute(1, 2, 0).cpu().numpy()
                    mask = lbl.squeeze(0, 1).cpu().numpy()
                    pred = pred.squeeze(0, 1).cpu().numpy()
                    image = valid_dataset.denormalization(image).astype(np.uint8)
                    query = valid_dataset.denormalization(query).astype(np.uint8)
                    mask = (mask * 255).astype(np.uint8)
                    pred = ((pred > 0.5) * 255).astype(np.uint8)
                    true_position = image.copy()
                    true_position[mask != 255] = 0
                    pred_position = image.copy()
                    pred_position[pred != 255] = 0
                    res.extend([image, query, true_position, pred_position])
                    titles.extend(
                        [
                            "Original Image",
                            "Centered Query",
                            "True Loc",
                            f"Pred. IoU: {iou_score:0.3f}",
                        ]
                    )
                # break
                subplot_images(
                    res, titles=titles, order=(-1, 4), axis=False, fig_size=(8, 10)
                ).savefig(str(out_dir / f"epoch_{i}.png"))
            # plt.show()

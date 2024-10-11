import matplotlib
import cv2
from temp_matching.vis import subplot_images, figure_to_array
from temp_matching.evaluator import Evaluator
from pathlib import Path
from typing import Tuple
import numpy as np
from tqdm import tqdm

# had to import this to load the model
from temp_matching.model import CustomUnet, EncodingCombination

# input_file: Path = Path("assets/pen.mp4")
input_file: Path = Path("assets/desk.mp4")
out_size: Tuple[int, int] = (1280 // 2, 720 // 2)
input_size: Tuple[int, int] = (512, 512)
device: str = "cuda"
is_state_dict = True

if is_state_dict:
    model_path: Path = Path(r"train_res\2024-09-24\best_model_state_dict.pth")
else:
    model_path: Path = Path(r"train_res\2024-09-24\best_model.pth")

output_file: Path = (
    Path(input_file.parent) / model_path.parent.name / f"{input_file.stem}_output.mp4"
)
if not output_file.parent.exists():
    output_file.parent.mkdir(parents=True, exist_ok=True)
cap = cv2.VideoCapture(str(input_file))
writer = cv2.VideoWriter(
    str(output_file),
    cv2.VideoWriter_fourcc(*"mp4v"),
    30,
    out_size,
)
matplotlib.use("Agg")  # Use the non-interactive Agg backend
frame = cv2.imread("assets/desk.png")
# frame = cv2.imread("assets/pen.png")
# frame = cv2.imread("assets/pen-nobg.png")

query = frame[650:1000, 200:450]
# query = frame[125:1050, 1045:1550]
# query = frame[39:370, 360:520]
query = cv2.cvtColor(query, cv2.COLOR_BGR2RGB)
evaluator = Evaluator(
    model_path=model_path,
    is_state_dict=is_state_dict,
    device=device,
    input_size=input_size,
)
evaluator.set_query(query, frame)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

start_frame = 0  # 99
frame_cnt = start_frame
end_frame = -102
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
while True:
    ret, frame = cap.read()
    if (frame_cnt > end_frame and end_frame > 0) or not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # output = evaluator.predict(frame)
    output = evaluator.fast_predict(frame)

    output = evaluator.post_process(output)
    overlayed = evaluator.overlay_mask(frame, output)
    fig = subplot_images(
        [
            evaluator.query,
            evaluator.image,
            cv2.cvtColor(output, cv2.COLOR_GRAY2BGR) * 255,
            overlayed,
        ],
        order=(2, -1),
        titles=[
            "Resized Centered Query",
            f"Resized Frame: {frame_cnt}/{total_frames}",
            "Output Mask",
            "Overlayed on Real",
        ],
        show=False,
    )
    fig_img = figure_to_array(fig)
    cv2.imshow("frame", cv2.cvtColor(fig_img, cv2.COLOR_RGB2BGR))

    fig_img = cv2.resize(fig_img, out_size)
    writer.write(cv2.cvtColor(fig_img, cv2.COLOR_RGB2BGR))
    # cv2.imwrite(
    #     f"assets/frame_{frame_cnt}_out.png", cv2.cvtColor(fig_img, cv2.COLOR_RGB2BGR)
    # )
    frame_cnt += 1
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()

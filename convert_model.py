import torch
from pathlib import Path
from temp_matching.model import *

def load_model(checkpoint_path: str, device: str = "cpu"):
    """Load your trained CustomUnet with multiplication."""
    # Define your Unet args exactly as during training!
    # Example: adjust encoder_name, classes, etc. to match your trained model
    unet_args = {
        "encoder_name": "resnet34",  # REPLACE with your encoder (e.g., 'resnet50', 'efficientnet-b0')
        "encoder_weights": None,     # No pretrained since you're loading checkpoint
        "in_channels": 3,
        "classes": 1,                # REPLACE with your num classes
        "activation": None,
    }
    
    
    
    model = torch.load(checkpoint_path, map_location=device)
    
    
    model.to(device)
    model.eval()
    return model


def export_to_onnx(
    checkpoint_path: str=r'F:\MSc Works\temp_matching\train_res\2024-09-24\best_model.pth',
    onnx_path: str = "model.onnx",
    input_size=(1,2, 3, 512, 512),
    opset_version: int = 17,
    dynamic_batch: bool = True,
    device: str = "cpu",
):
    model = load_model(checkpoint_path, device=device)
    
    dummy_input = torch.randn(input_size, device=device, dtype=torch.float32)
    
    input_names = ["combined_input"]   # Single input tensor
    output_names = ["masks"]
    
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {"combined_input": {0: "batch_size"}, "masks": {0: "batch_size"}}
    
    Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        verbose=False,
    )
    print(f"✅ Exported to {onnx_path}")
    print("Verify with: onnx.checker.check_model(onnx.load(onnx_path))")


if __name__ == "__main__":
    export_to_onnx()
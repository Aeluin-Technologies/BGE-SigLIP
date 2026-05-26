import torch
from transformers.models.siglip2.modeling_siglip2 import Siglip2VisionEmbeddings
from src.model import SiglipBgeAligner, SiglipONNXWrapper
from src.utils import patched_resize_positional_embeddings
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.transformers.optimizer import optimize_model


def export():
    Siglip2VisionEmbeddings.resize_positional_embeddings = (
        patched_resize_positional_embeddings
    )

    model = SiglipBgeAligner()
    model.load_state_dict(torch.load("model_checkpoint.pt"))
    onnx_model = SiglipONNXWrapper(model).eval()

    dummy_inputs = (
        torch.randn(1, 3, 224, 224),
        torch.ones(1, 196),
        torch.tensor([[14, 14]]),
    )
    torch.onnx.export(
        onnx_model,
        dummy_inputs,
        "model.onnx",
        opset_version=18,
        input_names=["pixel_values", "attention_mask", "spatial_shapes"],
        output_names=["image_embedding"],
    )

    optimized = optimize_model(
        "model.onnx", model_type="bert", num_heads=16, hidden_size=1152
    )
    optimized.save_model_to_file("model_fp32.onnx")

    quantize_dynamic(
        "model_fp32.onnx", "model_int8.onnx", weight_type=QuantType.QInt8
    )
    print("Export and quantization finished.")


if __name__ == "__main__":
    export()

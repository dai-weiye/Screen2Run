"""Independent raw-image CLIP cosine diagnostics; no canvas alignment.

DCGen's released whole-image evaluator preprocesses each image separately with
OpenAI ViT-B/32 (quickgelu), unlike this project's legacy resize-to-reference
helper. This adapter uses the locally cached Hugging Face copy of that model.
Backend equivalence to the released open_clip implementation remains explicit
and pending; these reports cannot silently replace a published experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

MODEL = "openai/clip-vit-base-patch32"
REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
PROTOCOL = "clip_vit_b32_raw_independent_rgb_hf_v1"
# Design2Code's published CLIP step is not computed on the raw screenshots. Their README:
# "To rule out the texts in the screenshots, we use the OpenCV inpainting to mask all
# detected text boxes using their bounding box coordinates." Reporting an unmasked cosine
# under the same column name is therefore not their formula. That is why this module now
# carries both protocols and every report says which one produced its number.
PROTOCOL_TEXT_MASKED = "clip_vit_b32_text_inpainted_independent_rgb_hf_v1"
# Their ocr_free_utils recovers the boxes from the reference HTML source, which Android
# does not have. The boxes here come from the same macOS Vision pass the block metrics use,
# on both sides and for every method -- the declared Android operationalisation.
INPAINT_RADIUS = 3.0


def unit_embedding(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] != 1 or not np.isfinite(result).all():
        raise ValueError("Expected one finite image embedding")
    norm = float(np.linalg.norm(result))
    if not norm:
        raise ValueError("Zero image embedding")
    return result[0] / norm


def inpaint_text_boxes(image: Image.Image, boxes: list[list[float]]) -> Image.Image:
    """Inpaint every detected text box, in the image's own pixels.

    ``boxes`` are the stored normalised ``[x, y, w, h]`` with a top-left origin, which is
    the convention both sides of every pair already share.
    """
    import cv2

    array = np.array(image.convert("RGB"))
    height, width = array.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    for box in boxes:
        x, y, bw, bh = (float(v) for v in box)
        x0, y0 = max(0, int(round(x * width))), max(0, int(round(y * height)))
        x1 = min(width, int(round((x + bw) * width)))
        y1 = min(height, int(round((y + bh) * height)))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 255
    if not mask.any():
        return image
    return Image.fromarray(cv2.inpaint(array, mask, INPAINT_RADIUS, cv2.INPAINT_TELEA))


def score_pair(model, processor, reference: Image.Image, generated: Image.Image) -> float:
    import torch
    embeddings = []
    with torch.no_grad():
        for image in (reference, generated):
            # Keep each source's own dimensions. The model's canonical resize
            # and center crop happen inside the identical processor on each side.
            inputs = processor(images=image.convert("RGB"), return_tensors="pt")
            embeddings.append(unit_embedding(model.get_image_features(**inputs)))
    return float(np.dot(*embeddings))


def score_pair_text_masked(model, processor, reference: Image.Image, generated: Image.Image,
                           reference_boxes: list[list[float]],
                           generated_boxes: list[list[float]]) -> float:
    """The published CLIP step: inpaint both sides' text boxes, then embed."""
    return score_pair(model, processor,
                      inpaint_text_boxes(reference, reference_boxes),
                      inpaint_text_boxes(generated, generated_boxes))


def sha_file(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def load_local_model():
    from huggingface_hub import snapshot_download
    from transformers import CLIPImageProcessor, CLIPModel
    snapshot = Path(snapshot_download(MODEL, revision=REVISION, local_files_only=True))
    config = json.loads((snapshot / "config.json").read_bytes())
    vision = config.get("vision_config", {})
    if (vision.get("patch_size"), vision.get("image_size"), vision.get("hidden_act"),
            config.get("projection_dim")) != (32, 224, "quick_gelu", 512):
        raise ValueError("Cached CLIP architecture is not the declared OpenAI ViT-B/32")
    weights = snapshot / "pytorch_model.bin"
    provenance = {"model": MODEL, "revision": REVISION, "backend": "transformers CPU float32",
                  "weights_sha256": sha_file(weights),
                  "config_sha256": sha_file(snapshot / "config.json"),
                  "preprocessor_sha256": sha_file(snapshot / "preprocessor_config.json"),
                  "versions": {name: importlib.metadata.version(name)
                               for name in ("torch", "transformers", "Pillow", "numpy")}}
    model = CLIPModel.from_pretrained(snapshot, local_files_only=True).float().eval()
    processor = CLIPImageProcessor.from_pretrained(snapshot, local_files_only=True)
    return model, processor, provenance


def openclip_vision_state(hf_state):
    """Lossless tensor renaming for the OpenAI ViT-B/32 visual tower only."""
    import torch
    prefix = "vision_model."
    names = {"class_embedding": "embeddings.class_embedding",
             "positional_embedding": "embeddings.position_embedding.weight",
             "conv1.weight": "embeddings.patch_embedding.weight",
             "ln_pre.weight": "pre_layrnorm.weight", "ln_pre.bias": "pre_layrnorm.bias",
             "ln_post.weight": "post_layernorm.weight", "ln_post.bias": "post_layernorm.bias"}
    result = {out: hf_state[prefix + source] for out, source in names.items()}
    result["proj"] = hf_state["visual_projection.weight"].T.contiguous()
    for i in range(12):
        source, target = f"{prefix}encoder.layers.{i}.", f"transformer.resblocks.{i}."
        for suffix in ("weight", "bias"):
            result[target + "attn.in_proj_" + suffix] = torch.cat([
                hf_state[source + f"self_attn.{letter}_proj.{suffix}"] for letter in "qkv"], dim=0)
            for out, original in (("attn.out_proj", "self_attn.out_proj"),
                                  ("ln_1", "layer_norm1"), ("ln_2", "layer_norm2"),
                                  ("mlp.c_fc", "mlp.fc1"), ("mlp.c_proj", "mlp.fc2")):
                result[target + out + "." + suffix] = hf_state[source + original + "." + suffix]
    return result


def stable_preprocessing_description(processor) -> str:
    """Keep transform parameters, not Python's process-specific addresses.

    A callable's repr otherwise changes the cache/protocol identity on every
    restart despite identical preprocessing. This does not alter any pixels.
    """
    return re.sub(r"(<function [^<>]+?) at 0x[0-9a-fA-F]+>", r"\1>", str(processor))


def load_published_model():
    """Use DCGen's released open_clip backend with cached, pinned OpenAI weights.

    No text tower is scored and no new model download is allowed. The conversion
    changes names/concatenation only; strict loading rejects missing tensors.
    """
    import open_clip
    hf_model, hf_processor, provenance = load_local_model()
    model, _, processor = open_clip.create_model_and_transforms(
        "ViT-B-32-quickgelu", pretrained=None, device="cpu", precision="fp32")
    model.visual.load_state_dict(openclip_vision_state(hf_model.state_dict()), strict=True)
    model.eval()
    provenance.update(backend="open_clip CPU float32", open_clip_version=open_clip.__version__,
                      conversion="HF visual tensor names + QKV concatenation; strict load",
                      preprocessing=stable_preprocessing_description(processor),
                      canvas_alignment=False, text_masking=False)
    return model, processor, provenance, hf_model, hf_processor


def published_embedding(model, processor, image: Image.Image) -> np.ndarray:
    import torch
    with torch.no_grad():
        tensor = processor(image.convert("RGB")).unsqueeze(0)
        return unit_embedding(model.encode_image(tensor))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--generated", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite CLIP report")
    import torch
    torch.set_num_threads(2)
    model, processor, provenance = load_local_model()
    input_hashes = [sha_file(path) for path in (args.reference, args.generated)]
    with Image.open(args.reference) as reference, Image.open(args.generated) as generated:
        if reference.getexif().get(274, 1) != 1 or generated.getexif().get(274, 1) != 1:
            raise ValueError("Image orientation must be normalized explicitly before evaluation")
        result = score_pair(model, processor, reference, generated)
        sizes = [reference.size, generated.size]
    if input_hashes != [sha_file(path) for path in (args.reference, args.generated)]:
        raise ValueError("Input image changed during evaluation")
    report = {"diagnostic_only": True, "protocol": PROTOCOL,
              "protocol_status": "pending_backend_equivalence_check",
              "clip_cosine": result, "model_provenance": provenance,
              "raw_image_sizes": sizes, "input_sha256": input_hashes,
              "canvas_alignment": False, "text_masking": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"clip_cosine": result, "output": str(args.output),
                      "protocol_status": report["protocol_status"]}))


if __name__ == "__main__":
    main()

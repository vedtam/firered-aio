import os
import gc
import ctypes
import subprocess
import importlib.util

# Fix: In some RunPod/containerised environments the nvidia_uvm kernel module
# is loaded but its device file (/dev/nvidia-uvm) has not been created yet,
# causing PyTorch's cudaGetDeviceCount() to return "CUDA unknown error" and
# silently fall back to CPU.
#
# Loading libcuda.so via ctypes and calling cuInit(0) forces the CUDA driver
# to fully initialise (including creating /dev/nvidia-uvm) before PyTorch
# tries to enumerate devices.  This must run BEFORE `import torch`.
try:
    _libcuda = ctypes.CDLL("libcuda.so.1", mode=ctypes.RTLD_GLOBAL)
    _libcuda.cuInit(0)
except (OSError, AttributeError):
    # libcuda not present (CPU-only machine) or cuInit not found — ignore.
    pass
# Also run nvidia-smi to wake the driver daemon (harmless if no GPU).
subprocess.run(["nvidia-smi"], capture_output=True)

# Fix: an empty string for CUDA_VISIBLE_DEVICES masks all GPUs and causes
# "CUDA unknown error" inside PyTorch. Unset it so the driver enumerates
# devices normally. Must happen BEFORE torch is imported.
if os.environ.get("CUDA_VISIBLE_DEVICES", None) == "":
    del os.environ["CUDA_VISIBLE_DEVICES"]

import gradio as gr
import numpy as np
import spaces
import torch
import random
from PIL import Image
from typing import Iterable
from gradio.themes import Soft
from gradio.themes.utils import colors, fonts, sizes

if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") == "1" and importlib.util.find_spec("hf_transfer") is None:
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

colors.orange_red = colors.Color(
    name="orange_red",
    c50="#FFF0E5",
    c100="#FFE0CC",
    c200="#FFC299",
    c300="#FFA366",
    c400="#FF8533",
    c500="#FF4500",
    c600="#E63E00",
    c700="#CC3700",
    c800="#B33000",
    c900="#992900",
    c950="#802200",
)

class OrangeRedTheme(Soft):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.gray,
        secondary_hue: colors.Color | str = colors.orange_red,
        neutral_hue: colors.Color | str = colors.slate,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Outfit"), "Arial", "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            background_fill_primary="*primary_50",
            background_fill_primary_dark="*primary_900",
            body_background_fill="linear-gradient(135deg, *primary_200, *primary_100)",
            body_background_fill_dark="linear-gradient(135deg, *primary_900, *primary_800)",
            button_primary_text_color="white",
            button_primary_text_color_hover="white",
            button_primary_background_fill="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_primary_background_fill_hover="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_dark="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_hover_dark="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_secondary_text_color="black",
            button_secondary_text_color_hover="white",
            button_secondary_background_fill="linear-gradient(90deg, *primary_300, *primary_300)",
            button_secondary_background_fill_hover="linear-gradient(90deg, *primary_400, *primary_400)",
            button_secondary_background_fill_dark="linear-gradient(90deg, *primary_500, *primary_600)",
            button_secondary_background_fill_hover_dark="linear-gradient(90deg, *primary_500, *primary_500)",
            slider_color="*secondary_500",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_primary_shadow="*shadow_drop_lg",
            button_large_padding="11px",
            color_accent_soft="*primary_100",
            block_label_background_fill="*primary_200",
        )

orange_red_theme = OrangeRedTheme()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.__version__ =", torch.__version__)
print("Using device:", device)

from diffusers import FlowMatchEulerDiscreteScheduler
from qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
from qwenimage.transformer_qwenimage import QwenImageTransformer2DModel
from qwenimage.qwen_fa3_processor import QwenDoubleStreamAttnProcessorFA3

dtype = torch.bfloat16
total_vram_gib = 0
if device.type == "cuda":
    total_vram_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

execution_profile = os.environ.get("FIRERED_EXECUTION_PROFILE", "auto").lower()
profile_offload_map = {
    "low_mem": "sequential",
    "balanced": "partial",
    "gpu": "none",
}
default_offload_mode = "none"
if device.type == "cuda":
    if total_vram_gib >= 40:
        default_offload_mode = "partial"
    else:
        default_offload_mode = "model"

if execution_profile == "auto":
    selected_offload_mode = default_offload_mode
else:
    selected_offload_mode = profile_offload_map.get(execution_profile, execution_profile)

offload_mode = os.environ.get("FIRERED_OFFLOAD_MODE", selected_offload_mode).lower()

transformer_load_kwargs = {
    "torch_dtype": dtype,
}
if device.type == "cuda" and offload_mode == "partial":
    transformer_load_kwargs["device_map"] = "cuda"

pipe = QwenImageEditPlusPipeline.from_pretrained(
    "FireRedTeam/FireRed-Image-Edit-1.1", # ---> Prev: FireRedTeam/FireRed-Image-Edit-1.0
    transformer=QwenImageTransformer2DModel.from_pretrained(
        "prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V19",
        **transformer_load_kwargs,
    ),
    torch_dtype=dtype
)

if device.type == "cuda":
    if offload_mode == "sequential":
        pipe.enable_sequential_cpu_offload()
        generator_device = "cpu"
    elif offload_mode == "model":
        pipe.enable_model_cpu_offload()
        generator_device = "cpu"
    elif offload_mode == "partial":
        pipe._runtime_device_override = device
        generator_device = "cpu"
    else:
        pipe = pipe.to(device)
        generator_device = device.type
else:
    pipe = pipe.to(device)
    generator_device = device.type

print("Offload mode:", offload_mode)
print("Execution profile:", execution_profile)
if total_vram_gib:
    print(f"GPU VRAM: {total_vram_gib:.2f} GiB")

if device.type == "cuda":
    try:
        pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())
        print("Flash Attention 3 Processor set successfully.")
    except Exception as e:
        print(f"Warning: Could not set FA3 processor, using default: {e}")
else:
    print("Skipping FA3 processor (not on CUDA).")

MAX_SEED = np.iinfo(np.int32).max

def snap_dimension(value):
    return max(256, (int(value) // 8) * 8)

def update_dimensions_on_upload(images):
    if not images:
        return gr.update(value=1024), gr.update(value=1024)

    first_image = images[0]
    if isinstance(first_image, (tuple, list)):
        first_image = first_image[0]

    if isinstance(first_image, str):
        original_width, original_height = Image.open(first_image).size
    elif isinstance(first_image, Image.Image):
        original_width, original_height = first_image.size
    else:
        original_width, original_height = Image.open(first_image.name).size

    if original_width > original_height:
        new_width = 1024
        aspect_ratio = original_height / original_width
        new_height = int(new_width * aspect_ratio)
    else:
        new_height = 1024
        aspect_ratio = original_width / original_height
        new_width = int(new_height * aspect_ratio)

    new_width = snap_dimension(new_width)
    new_height = snap_dimension(new_height)

    return gr.update(value=new_width), gr.update(value=new_height)

@spaces.GPU
def infer(
    images,
    prompt,
    negative_prompt,
    seed,
    randomize_seed,
    guidance_scale,
    width,
    height,
    steps,
    progress=gr.Progress(track_tqdm=True)
):
    gc.collect()
    torch.cuda.empty_cache()

    if not images:
        raise gr.Error("Please upload at least one image to edit.")

    pil_images = []
    if images is not None:
        for item in images:
            try:
                if isinstance(item, tuple) or isinstance(item, list):
                    path_or_img = item[0]
                else:
                    path_or_img = item

                if isinstance(path_or_img, str):
                    pil_images.append(Image.open(path_or_img).convert("RGB"))
                elif isinstance(path_or_img, Image.Image):
                    pil_images.append(path_or_img.convert("RGB"))
                else:
                    pil_images.append(Image.open(path_or_img.name).convert("RGB"))
            except Exception as e:
                print(f"Skipping invalid image item: {e}")
                continue

    if not pil_images:
        raise gr.Error("Could not process uploaded images.")

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    generator = torch.Generator(device=generator_device).manual_seed(seed)
    negative_prompt = (negative_prompt or "").strip() or None
    if guidance_scale > 1 and negative_prompt is None:
        negative_prompt = (
            "worst quality, low quality, bad anatomy, bad hands, text, error, missing fingers, "
            "extra digit, fewer digits, cropped, jpeg artifacts, signature, watermark, username, blurry"
        )

    width = snap_dimension(width)
    height = snap_dimension(height)

    try:
        result_image = pipe(
            image=pil_images,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            generator=generator,
            true_cfg_scale=guidance_scale,
        ).images[0]

        return result_image, seed

    except Exception as e:
        raise e
    finally:
        gc.collect()
        torch.cuda.empty_cache()

css = """
#col-container {
    margin: 0 auto;
    max-width: 1000px;
}
#main-title h1 {font-size: 2.4em !important;}
"""

with gr.Blocks() as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown("# **FireRed-Image-Edit-1.0-Fast - [v@1.1](https://huggingface.co/FireRedTeam/FireRed-Image-Edit-1.1)**", elem_id="main-title")
        gr.Markdown("Perform image edits using [FireRed-Image-Edit-1.0](https://huggingface.co/FireRedTeam/FireRed-Image-Edit-1.0) with 4-step fast inference. Open on [GitHub](https://github.com/PRITHIVSAKTHIUR/FireRed-Image-Edit-1.0-Fast)")

        with gr.Row(equal_height=True):
            with gr.Column():
                images = gr.Gallery(
                    label="Upload Images",
                    #sources=["upload", "clipboard"],
                    type="filepath",
                    columns=2,
                    rows=1,
                    height=300,
                    allow_preview=True
                )

                prompt = gr.Text(
                    label="Edit Prompt",
                    show_label=True,
                    max_lines=2,
                    placeholder="e.g., transform into anime, upscale, change lighting...",
                )

                run_button = gr.Button("Edit Image", variant="primary")

            with gr.Column():
                output_image = gr.Image(label="Output Image", interactive=False, format="png", height=395)

                with gr.Accordion("Advanced Settings", open=False, visible=True):
                    seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0)
                    randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)
                    guidance_scale = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=10.0, step=0.1, value=1.0)
                    negative_prompt_input = gr.Text(
                        label="Negative Prompt",
                        show_label=True,
                        max_lines=3,
                        placeholder="Optional. Leave blank to disable or use the built-in default when guidance > 1.",
                        value="",
                    )
                    width = gr.Slider(label="Width", minimum=256, maximum=2048, step=8, value=1024)
                    height = gr.Slider(label="Height", minimum=256, maximum=2048, step=8, value=1024)
                    steps = gr.Slider(label="Inference Steps", minimum=1, maximum=50, step=1, value=4)

        gr.Markdown("[*](https://huggingface.co/FireRedTeam/FireRed-Image-Edit-1.0)This is still an experimental Space for FireRed-Image-Edit-1.0.")

    images.change(
        fn=update_dimensions_on_upload,
        inputs=[images],
        outputs=[width, height]
    )

    run_button.click(
        fn=infer,
        inputs=[images, prompt, negative_prompt_input, seed, randomize_seed, guidance_scale, width, height, steps],
        outputs=[output_image, seed]
    )

if __name__ == "__main__":
    demo.queue(max_size=30).launch(css=css, theme=orange_red_theme, mcp_server=True, ssr_mode=False, show_error=True)

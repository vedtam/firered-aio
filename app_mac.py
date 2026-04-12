"""
app_mac.py — M1/Apple Silicon-optimized version of app.py.

Run locally on macOS with Apple Silicon (MPS) or CPU.
Does NOT require HuggingFace Spaces / ZeroGPU.

Usage:
    python app_mac.py              # load both FireRed base + AIO transformer
    python app_mac.py --aio-only   # skip FireRed base, load only AIO transformer
"""

import argparse
import gc
import os
import time
import urllib.parse as _urlparse

import gradio as gr
import numpy as np
import random
import torch
from PIL import Image, ImageDraw
from typing import Iterable
from gradio.themes import Soft
from gradio.themes.utils import colors, fonts, sizes

# ---------------------------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------------------------
_parser = argparse.ArgumentParser(description="FireRed AIO — macOS/MPS edition")
_parser.add_argument(
    "--aio-only",
    action="store_true",
    default=False,
    help=(
        "Skip loading the FireRed base pipeline and run only the "
        "AIO transformer (prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V19). "
        "Useful when memory is tight or you want faster startup."
    ),
)
_args, _unknown = _parser.parse_known_args()
AIO_ONLY: bool = _args.aio_only

# ---------------------------------------------------------------------------
# Device detection: CUDA → MPS → CPU
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"[app_mac] Using device: {device}")

# bfloat16 is not supported on MPS; use float16 there.
dtype = torch.float16 if device.type == "mps" else torch.bfloat16

# torch.Generator does not support MPS — always use CPU for RNG on MPS/CPU.
generator_device = "cpu" if device.type in ("mps", "cpu") else device.type


def _empty_cache():
    """Call torch.cuda.empty_cache() only when running on CUDA."""
    if device.type == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# CUDA-specific environment tuning (skip on MPS/CPU)
# ---------------------------------------------------------------------------
if device.type == "cuda":
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    total_vram_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"[app_mac] GPU VRAM: {total_vram_gib:.2f} GiB")
else:
    total_vram_gib = 0

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
from diffusers import FlowMatchEulerDiscreteScheduler
from qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
from qwenimage.transformer_qwenimage import QwenImageTransformer2DModel

if AIO_ONLY:
    print("[app_mac] Running in AIO-only mode — FireRed base model not loaded.")

    # Build a minimal pipeline that only wraps the AIO transformer so that
    # the infer() function can call pipe(image=..., prompt=..., ...) unchanged.
    class _AIOOnlyPipeline:
        """Thin wrapper: loads the AIO transformer and delegates to it directly."""

        def __init__(self):
            print("[app_mac] Loading AIO transformer …")
            self._transformer = QwenImageTransformer2DModel.from_pretrained(
                "prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V19",
                torch_dtype=dtype,
            )
            # Load the full pipeline using the AIO transformer but without the
            # heavy FireRed base weights — we still need a valid pipeline object
            # that can call .to(device) / offload helpers and run inference.
            self._pipe = QwenImageEditPlusPipeline.from_pretrained(
                "prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V19",
                transformer=self._transformer,
                torch_dtype=dtype,
            )
            self._apply_offload()
            print("[app_mac] AIO-only pipeline ready.")

        def _apply_offload(self):
            if device.type == "mps":
                self._pipe.enable_model_cpu_offload()
            elif device.type == "cuda":
                self._pipe.enable_model_cpu_offload()
            else:
                self._pipe = self._pipe.to(device)

        def __call__(self, **kwargs):
            return self._pipe(**kwargs)

    pipe = _AIOOnlyPipeline()

else:
    # -----------------------------------------------------------------------
    # Normal mode: load FireRed base + AIO transformer, same as app.py
    # -----------------------------------------------------------------------

    # Offload-mode selection
    execution_profile = os.environ.get("FIRERED_EXECUTION_PROFILE", "auto").lower()
    profile_offload_map = {
        "low_mem":  "sequential",
        "balanced": "partial",
        "gpu":      "none",
    }

    if device.type == "cuda":
        if total_vram_gib < 24:
            default_offload_mode = "sequential"
        elif total_vram_gib < 48:
            default_offload_mode = "model"
        elif total_vram_gib < 72:
            default_offload_mode = "partial"
        else:
            default_offload_mode = "none"
    else:
        # MPS / CPU: use model-level offload as a safe default for large models
        default_offload_mode = "model"

    if execution_profile == "auto":
        selected_offload_mode = default_offload_mode
    else:
        selected_offload_mode = profile_offload_map.get(execution_profile, execution_profile)

    offload_mode = os.environ.get("FIRERED_OFFLOAD_MODE", selected_offload_mode).lower()

    transformer_load_kwargs: dict = {"torch_dtype": dtype}
    if device.type == "cuda" and offload_mode == "partial":
        transformer_load_kwargs["device_map"] = "cuda"

    print("[app_mac] Loading FireRed base pipeline + AIO transformer …")
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        "FireRedTeam/FireRed-Image-Edit-1.1",
        transformer=QwenImageTransformer2DModel.from_pretrained(
            "prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V19",
            **transformer_load_kwargs,
        ),
        torch_dtype=dtype,
    )

    if device.type == "cuda":
        if offload_mode == "sequential":
            pipe.enable_sequential_cpu_offload()
        elif offload_mode == "model":
            pipe.enable_model_cpu_offload()
        elif offload_mode == "partial":
            pipe._runtime_device_override = device
        else:
            pipe = pipe.to(device)
    else:
        # MPS / CPU
        if offload_mode == "model":
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(device)

    print(f"[app_mac] Offload mode: {offload_mode}")
    print(f"[app_mac] Execution profile: {execution_profile}")

    # Flash Attention 3 is CUDA-only
    if device.type == "cuda":
        from qwenimage.qwen_fa3_processor import QwenDoubleStreamAttnProcessorFA3

        major, minor = torch.cuda.get_device_capability(0)
        sm = major * 10 + minor
        if sm > 90:
            print(
                f"[app_mac] Skipping FA3 processor: GPU sm_{sm} not supported by precompiled "
                "vllm-flash-attn3 kernel (requires sm_80 or sm_90). Using PyTorch SDPA fallback."
            )
        else:
            try:
                pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())
                print("[app_mac] Flash Attention 3 processor set successfully.")
            except Exception as exc:
                print(f"[app_mac] Warning: Could not set FA3 processor, using default: {exc}")
    else:
        print("[app_mac] Skipping FA3 processor (not on CUDA) — SDPA fallback will be used.")

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.environ.get(
    "FIRERED_OUTPUT_DIR",
    os.path.expanduser("~/firered_outputs"),
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------
MAX_SEED = 2**31 - 1


def snap_dimension(value):
    return max(256, (int(value) // 8) * 8)


PRESET_DIMS = {
    "SD":      (640, 360),
    "HD":      (1280, 720),
    "Full HD": (1920, 1080),
    "2K":      (2048, 1152),
}


def resolve_dimensions(preset, orientation):
    w, h = PRESET_DIMS.get(preset, (1280, 720))
    if orientation == "Portrait":
        w, h = h, w
    return w, h


def _build_thumb_html(paths):
    if not paths:
        return ""
    items = "".join(
        f'<div style="position:relative;display:inline-block;border-radius:8px;overflow:hidden;'
        f'width:110px;height:110px;flex-shrink:0">'
        f'<img src="/gradio_api/file={_urlparse.quote(p, safe="/")}" '
        f'style="width:110px;height:110px;object-fit:cover;display:block">'
        f'<button class="thumb-remove" data-idx="{i}" '
        f'style="position:absolute;top:4px;right:4px;background:rgba(0,0,0,0.65);'
        f'color:white;border:none;border-radius:50%;width:22px;height:22px;'
        f'cursor:pointer;font-size:16px;line-height:22px;padding:0;text-align:center">&#x2715;</button>'
        f'</div>'
        for i, p in enumerate(paths)
    )
    return (
        f'<div style="background:white;border:3px solid #e2e8f0;border-radius:4px;'
        f'box-shadow:0 2px 5px 0 rgba(0,0,0,0.1);padding:10px 12px;'
        f'width:100%;box-sizing:border-box;">'
        f'<div style="display:flex;flex-wrap:wrap;gap:8px">{items}</div>'
        f'</div>'
    )


def _files_changed(files):
    paths = files or []
    return paths, _build_thumb_html(paths)


def _remove_image(paths, idx):
    idx = int(idx) if idx is not None else -1
    if idx < 0:
        return paths, _build_thumb_html(paths), gr.update(), -1
    paths = list(paths)
    if 0 <= idx < len(paths):
        paths.pop(idx)
    return paths, _build_thumb_html(paths), paths or None, -1


# ---------------------------------------------------------------------------
# Inference  (plain function — no @spaces.GPU needed for local use)
# ---------------------------------------------------------------------------
def infer(
    images,
    prompt,
    negative_prompt,
    guidance_scale,
    width,
    height,
    steps,
    progress=gr.Progress(track_tqdm=True),
):
    gc.collect()
    _empty_cache()

    if not images:
        raise gr.Error("Please upload at least one image to edit.")

    pil_images = []
    if images is not None:
        for path in images:
            try:
                pil_images.append(Image.open(path).convert("RGB"))
            except Exception as e:
                print(f"Skipping invalid image: {e}")
                continue

    if not pil_images:
        raise gr.Error("Could not process uploaded images.")

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

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(OUTPUT_DIR, f"{timestamp}_seed{seed}.png")
        result_image.save(output_path)
        print(f"[app_mac] Saved output to {output_path}")

        return result_image

    except Exception as e:
        raise e
    finally:
        gc.collect()
        _empty_cache()


# ---------------------------------------------------------------------------
# CSS / UI
# ---------------------------------------------------------------------------
css = """
#col-container {
    margin: 0 auto;
    max-width: 1000px;
}
#prompt-main textarea {
    min-height: 300px !important;
}
#prompt-negative textarea {
    min-height: 200px !important;
}
#remove-trigger, div:has(> #remove-trigger) {
    display: none !important;
}
#thumbnails-html .html-container {
    padding: 0 !important;
}
"""

with gr.Blocks() as demo:
    with gr.Column(elem_id="col-container"):
        with gr.Row():
            with gr.Column():
                images = gr.File(
                    label="Upload Images",
                    file_count="multiple",
                    file_types=["image"],
                    type="filepath",
                    height=200,
                )

                file_paths = gr.State([])
                remove_trigger = gr.Number(value=-1, visible=True, elem_id="remove-trigger", label="", precision=0)
                thumbnails_html = gr.HTML(value="", elem_id="thumbnails-html")

                prompt = gr.Text(
                    label="Edit Prompt",
                    show_label=True,
                    lines=12,
                    max_lines=50,
                    placeholder="e.g., transform into anime, upscale, change lighting...",
                    elem_id="prompt-main",
                )

                negative_prompt_input = gr.Text(
                    label="Negative Prompt",
                    show_label=True,
                    lines=8,
                    max_lines=50,
                    placeholder="Optional. Leave blank to disable or use the built-in default when guidance > 1.",
                    value="",
                    elem_id="prompt-negative",
                )

                run_button = gr.Button("Edit Image", variant="primary", size="sm")

            with gr.Column():
                output_image = gr.Image(label="Output Image", interactive=False, format="png", height=395)

                with gr.Accordion("Advanced Settings", open=False, visible=True):
                    with gr.Row():
                        preset_radio = gr.Radio(
                            choices=["SD", "HD", "Full HD", "2K"],
                            value="HD",
                            label="Resolution",
                            scale=3,
                        )
                        orientation_radio = gr.Radio(
                            choices=["Landscape", "Portrait"],
                            value="Landscape",
                            label="Orientation",
                            scale=2,
                        )

                    with gr.Row():
                        width = gr.Number(label="Width (px)", value=1280, minimum=256, maximum=4096, step=8, scale=1)
                        height = gr.Number(label="Height (px)", value=720, minimum=256, maximum=4096, step=8, scale=1)

                    seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0, visible=False)
                    randomize_seed = gr.Checkbox(label="Randomize Seed", value=True, visible=False)
                    guidance_scale = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=10.0, step=0.1, value=1.0)
                    steps = gr.Slider(label="Inference Steps", minimum=1, maximum=50, step=1, value=4)

images.change(
    fn=_files_changed,
    inputs=[images],
    outputs=[file_paths, thumbnails_html],
)

remove_trigger.input(
    fn=_remove_image,
    inputs=[file_paths, remove_trigger],
    outputs=[file_paths, thumbnails_html, images, remove_trigger],
)

preset_radio.change(
    fn=resolve_dimensions,
    inputs=[preset_radio, orientation_radio],
    outputs=[width, height],
)
orientation_radio.change(
    fn=resolve_dimensions,
    inputs=[preset_radio, orientation_radio],
    outputs=[width, height],
)

run_button.click(
    fn=infer,
    inputs=[file_paths, prompt, negative_prompt_input, guidance_scale, width, height, steps],
    outputs=[output_image],
)

demo.load(
    fn=None,
    js="""() => {
    document.addEventListener('click', function(e) {
        const btn = e.target.closest('.thumb-remove');
        if (!btn) return;
        e.preventDefault();
        e.stopPropagation();
        const idx = parseInt(btn.dataset.idx, 10);
        const container = document.getElementById('remove-trigger');
        if (!container) return;
        const inp = container.querySelector('input');
        if (!inp) return;
        inp.value = String(idx);
        inp.dispatchEvent(new Event('input', {bubbles: true}));
    });
}"""
)

if __name__ == "__main__":
    demo.queue(max_size=10).launch(css=css, theme=orange_red_theme, ssr_mode=False, show_error=True)

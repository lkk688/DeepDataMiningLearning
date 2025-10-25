"""
Three-Stage Image Editing Pipeline (Batch Mode)
-----------------------------------------------
Supports:
  - Single image input
  - Folder input: processes all images sequentially

Example:
python smart_image_edit_pipeline_batch.py \
    --input_img ./photos \
    --output_dir ./outputs \
    --task portrait_clean \
    --stages 2 3
"""

import os
import gc
import glob
import torch
import argparse
from PIL import Image
from diffusers import (
    QwenImageEditPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionUpscalePipeline,
)
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


# =========================================================
#  Utilities: Device & Memory
# =========================================================
def get_device():
    if torch.cuda.is_available():
        device, dtype = "cuda", torch.bfloat16 #torch.float16
    elif torch.backends.mps.is_available():
        device, dtype = "mps", torch.float16
    else:
        device, dtype = "cpu", torch.float32
    print(f"[Device] Using {device}, dtype={dtype}")
    return device, dtype


def clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("[Memory] GPU memory cleared.\n")


# =========================================================
#  Stage 1 – VLM Prompt Enhancement
# =========================================================
def enhance_prompt_vlm(image_path, raw_prompt, model_name):
    device, dtype = get_device()
    print(f"[Stage 1] Loading VLM: {model_name}")
    processor = AutoProcessor.from_pretrained(model_name)
    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device
    )

    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": f"Rewrite this editing instruction clearly and precisely for an image editing model: {raw_prompt}",
                },
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(device)

    print("[Stage 1] Generating enhanced prompt...")
    with torch.inference_mode():
        ids = vlm.generate(**inputs, max_new_tokens=256)
        enhanced_prompt = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

    print(f"[Stage 1] Enhanced prompt:\n{enhanced_prompt}\n")
    del vlm, processor
    clear_memory()
    return enhanced_prompt


# =========================================================
#  Stage 2 – Image Editing
# =========================================================
def run_image_edit(input_path, output_path, prompt, model_name):
    device, dtype = get_device()
    print(f"[Stage 2] Loading edit model: {model_name}")
    pipe = QwenImageEditPipeline.from_pretrained(model_name, torch_dtype=dtype).to(device)

    image = Image.open(input_path).convert("RGB")
    inputs = dict(
        image=image,
        prompt=prompt,
        negative_prompt="do not change the main person, remove background clutter",
        generator=torch.manual_seed(0),
        num_inference_steps=40,
        true_cfg_scale=4.0,
    )

    with torch.inference_mode():
        result = pipe(**inputs)
        out_img = result.images[0]
        out_img.save(output_path)
        print(f"[Stage 2] Edited image saved: {os.path.abspath(output_path)}\n")

    del pipe
    clear_memory()
    return output_path


# =========================================================
#  Stage 3 – Enhancement / Upscale
# =========================================================
def run_image_enhance(
    input_path: str,
    output_path: str,
    mode: str,
    img2img_model: str,
    upscale_model: str,
):
    """
    Stage 3 – Image Enhancement / Upscale / Restoration
    Supports: "upscale", "img2img", "realesrgan", "gfpgan"
    """
    device, dtype = get_device()
    image = Image.open(input_path).convert("RGB")

    # ====================================================
    # 1️⃣  Real-ESRGAN — general super-resolution
    # ====================================================
    if mode == "realesrgan":
        print("[Stage 3] Loading Real-ESRGAN for general enhancement")
        try:
            from realesrgan import RealESRGAN
        except ImportError:
            raise ImportError("Please install realesrgan: pip install realesrgan")

        scale = 2
        sr_model = RealESRGAN(device, scale=scale)
        sr_model.load_weights(f"RealESRGAN_x{scale}plus.pth")
        with torch.inference_mode():
            sr_image = sr_model.predict(image)
        sr_image.save(output_path)
        print(f"[Stage 3] Real-ESRGAN enhanced image saved: {os.path.abspath(output_path)}\n")

    # ====================================================
    # 2️⃣  GFPGAN — face restoration (optional ESRGAN combo)
    # ====================================================
    elif mode == "gfpgan":
        print("[Stage 3] Loading GFPGAN for face restoration")
        try:
            #pip install realesrgan gfpgan
            from gfpgan import GFPGANer
            from realesrgan import RealESRGAN
        except ImportError:
            raise ImportError("Please install gfpgan and realesrgan: pip install gfpgan realesrgan")

        # Optional Real-ESRGAN for background
        bg_upsampler = RealESRGAN(device, scale=2)
        bg_upsampler.load_weights("RealESRGAN_x2plus.pth")

        restorer = GFPGANer(
            model_path="GFPGANv1.4.pth",
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=bg_upsampler,
        )

        with torch.inference_mode():
            _, _, restored_img = restorer.enhance(
                np.array(image), has_aligned=False, only_center_face=False, paste_back=True
            )

        Image.fromarray(restored_img[:, :, ::-1]).save(output_path)
        print(f"[Stage 3] GFPGAN face-restored image saved: {os.path.abspath(output_path)}\n")

    # ====================================================
    # 3️⃣  Stable Diffusion Upscaler
    # ====================================================
    elif mode == "upscale":
        print(f"[Stage 3] Loading upscaler: {upscale_model}")
        pipe = StableDiffusionUpscalePipeline.from_pretrained(
            upscale_model, torch_dtype=dtype
        ).to(device)

        # To save memory, reduce image size first
        image.thumbnail((512, 512))
        prompt = "enhance clarity, preserve all persons, do not add anything new"

        with torch.inference_mode():
            result = pipe(prompt=prompt, image=image, guidance_scale=0.0)
            result.images[0].save(output_path)
        print(f"[Stage 3] Upscaled image saved: {os.path.abspath(output_path)}\n")

        del pipe

    # ====================================================
    # 4️⃣  Stable Diffusion Img2Img Beautifier
    # ====================================================
    elif mode == "img2img":
        print(f"[Stage 3] Loading Img2Img model: {img2img_model}")
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            img2img_model, torch_dtype=dtype
        ).to(device)

        prompt = (
            "Enhance lighting, skin tone, and texture; "
            "keep all persons unchanged; realistic photo style"
        )

        with torch.inference_mode():
            result = pipe(prompt=prompt, image=image, strength=0.25, guidance_scale=4.0)
            result.images[0].save(output_path)
        print(f"[Stage 3] Beautified image saved: {os.path.abspath(output_path)}\n")

        del pipe

    else:
        raise ValueError(
            f"Unknown enhance mode: {mode}. Use one of ['upscale', 'img2img', 'realesrgan', 'gfpgan']."
        )

    clear_memory()
    return output_path


# =========================================================
#  Predefined Prompts
# =========================================================
COMMON_PROMPTS = {
    "portrait_clean": "Remove background and clutter, keep main person unchanged, professional portrait style.",
    "remove_watermark": "Remove watermark and small artifacts, preserve main subject, realistic lighting.",
    "background_blur": "Blur the background, focus on the person, cinematic depth-of-field effect.",
    "beautify": "Enhance lighting and skin tone, make it a professional headshot photo.",
}


# =========================================================
#  Process All Images by Stage (optimized for folder mode)
# =========================================================
def process_all_images(args, img_list):
    """Process each stage sequentially across all images to avoid reloading models repeatedly."""

    # ===== Stage 1: VLM Prompt Enhancement =====
    if 1 in args.stages:
        device, dtype = get_device()
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        print(f"[Stage 1] Loading VLM: {args.model_vlm}")
        processor = AutoProcessor.from_pretrained(args.model_vlm)
        vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_vlm, torch_dtype=dtype, device_map=device
        )

        for img_path in img_list:
            base_prompt = args.prompt or COMMON_PROMPTS.get(args.task, "")
            prompt = enhance_prompt_vlm(img_path, base_prompt, args.model_vlm)
            # Save the prompt for later use
            with open(os.path.join(args.output_dir, os.path.basename(img_path) + ".prompt.txt"), "w") as f:
                f.write(prompt)
        del vlm, processor
        clear_memory()
    else:
        print("[Stage 1] Skipped VLM enhancement.\n")

    # ===== Stage 2: Image Editing =====
    if 2 in args.stages:
        device, dtype = get_device()
        print(f"[Stage 2] Loading edit model: {args.model_edit}")
        from diffusers import QwenImageEditPipeline
        pipe = QwenImageEditPipeline.from_pretrained(args.model_edit, torch_dtype=dtype).to(device)

        for img_path in img_list:
            # Use saved prompt if Stage 1 was run
            prompt_file = os.path.join(args.output_dir, os.path.basename(img_path) + ".prompt.txt")
            if os.path.exists(prompt_file):
                with open(prompt_file, "r") as f:
                    prompt_final = f.read().strip()
            else:
                prompt_final = args.prompt or COMMON_PROMPTS.get(args.task, "")
            edited_path = os.path.join(
                args.output_dir, os.path.splitext(os.path.basename(img_path))[0] + "_edited.png"
            )

            image = Image.open(img_path).convert("RGB")
            inputs = dict(
                image=image,
                prompt=prompt_final,
                negative_prompt="do not change the main person, remove background clutter",
                generator=torch.manual_seed(0),
                num_inference_steps=40,
                true_cfg_scale=4.0,
            )
            with torch.inference_mode():
                result = pipe(**inputs)
                result.images[0].save(edited_path)
            print(f"[Stage 2] Saved: {edited_path}")
        del pipe
        clear_memory()
    else:
        print("[Stage 2] Skipped image editing.\n")

    # ===== Stage 3: Enhancement =====
    if 3 in args.stages:
        print(f"[Stage 3] Starting enhancement mode: {args.enhance_mode}")
        from PIL import Image
        import numpy as np
        device, dtype = get_device()

        # ------------------------------
        # 🧩 Helper: find source & output paths
        # ------------------------------
        def resolve_paths(img_path):
            """Detect whether file is edited or raw; return (source_path, output_path, name)."""
            base = os.path.basename(img_path)
            name, _ = os.path.splitext(base)
            if "edited" in name.lower():
                src = img_path
                name_clean = name.replace("_edited", "")
            else:
                candidate = os.path.join(args.output_dir, f"{name}_edited.png")
                src = candidate if os.path.exists(candidate) else img_path
                name_clean = name
            out_path = os.path.join(args.output_dir, f"{name_clean}_enhanced.png")
            return src, out_path, name_clean

        # ------------------------------
        # 🧩 Helper: enhancement per mode
        # ------------------------------
        def enhance_image(image, mode):
            if mode == "realesrgan":
                from realesrgan import RealESRGAN
                sr = RealESRGAN(device, scale=2)
                sr.load_weights("RealESRGAN_x2plus.pth")
                with torch.inference_mode():
                    result = sr.predict(image)
                del sr
                return result

            elif mode == "gfpgan":
                from gfpgan import GFPGANer
                from realesrgan import RealESRGAN
                bg = RealESRGAN(device, scale=2)
                bg.load_weights("RealESRGAN_x2plus.pth")
                restorer = GFPGANer(
                    model_path="GFPGANv1.4.pth",
                    upscale=1,
                    arch="clean",
                    channel_multiplier=2,
                    bg_upsampler=bg,
                )
                with torch.inference_mode():
                    _, _, restored = restorer.enhance(
                        np.array(image), has_aligned=False, only_center_face=False, paste_back=True
                    )
                del restorer, bg
                return Image.fromarray(restored[:, :, ::-1])

            elif mode == "upscale":
                from diffusers import StableDiffusionUpscalePipeline
                pipe = StableDiffusionUpscalePipeline.from_pretrained(
                    args.model_upscale, torch_dtype=dtype
                ).to(device)
                image.thumbnail((512, 512))
                with torch.inference_mode():
                    result = pipe(
                        prompt="enhance clarity, preserve all persons, do not add anything new",
                        image=image,
                        guidance_scale=0.0,
                    )
                out = result.images[0]
                del pipe
                return out

            elif mode == "img2img":
                from diffusers import StableDiffusionImg2ImgPipeline
                pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                    args.model_enhance, torch_dtype=dtype
                ).to(device)
                prompt = "Enhance lighting, skin tone, and texture; keep all persons unchanged; realistic photo style"
                with torch.inference_mode():
                    result = pipe(prompt=prompt, image=image, strength=0.25, guidance_scale=4.0)
                out = result.images[0]
                del pipe
                return out

            else:
                raise ValueError(f"Unknown enhance_mode: {mode}")

        # ------------------------------
        # 🚀 Run enhancement for all images
        # ------------------------------
        for img_path in img_list:
            src_path, out_path, name = resolve_paths(img_path)
            if not os.path.exists(src_path):
                print(f"[Stage 3] ⚠️  Missing source image: {img_path}")
                continue

            image = Image.open(src_path).convert("RGB")
            print(f"[Stage 3] Enhancing {name} ({args.enhance_mode})...")
            try:
                enhanced_img = enhance_image(image, args.enhance_mode)
                enhanced_img.save(out_path)
                print(f"[Stage 3] ✅ Saved: {out_path}")
            except Exception as e:
                print(f"[Stage 3] ❌ Failed {name}: {e}")

        clear_memory()

    else:
        print("[Stage 3] Skipped enhancement.\n")


# =========================================================
#  Main CLI Entrypoint
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="Smart Image Editing Pipeline (Optimized Folder Mode)")
    parser.add_argument("--input_img", type=str, required=True, help="Path to image or folder")
    parser.add_argument("--output_dir", type=str, default="./outputs/genimages", help="Output folder")
    parser.add_argument("--task", type=str, choices=list(COMMON_PROMPTS.keys()), default="portrait_clean",
                        help="Predefined task (portrait_clean, beautify, etc.)")
    parser.add_argument("--prompt", type=str, default=None, help="Custom text prompt (overrides task)")
    parser.add_argument("--stages", type=int, nargs="+", default=[3],
                        help="Stages to run (1=VLM, 2=Edit, 3=Enhance)")
    parser.add_argument("--model_vlm", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--model_edit", type=str, default="Qwen/Qwen-Image-Edit-2509")
    parser.add_argument("--model_enhance", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--model_upscale", type=str, default="stabilityai/stable-diffusion-x4-upscaler")
    parser.add_argument("--enhance_mode", type=str, choices=["img2img", "upscale", "realesrgan", "gfpgan"], default="img2img")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Collect all images ---
    if os.path.isdir(args.input_img):
        import glob
        img_list = sorted(
            [p for p in glob.glob(os.path.join(args.input_img, "*"))
             if p.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
        )
        print(f"\n📂 Folder mode: Found {len(img_list)} images in {args.input_img}\n")
    else:
        img_list = [args.input_img]
        print("\n🖼️ Single image mode\n")

    process_all_images(args, img_list)
    print("✅ All tasks completed. Results saved in:", os.path.abspath(args.output_dir))


if __name__ == "__main__":
    main()
import argparse
import os
from PIL import Image

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor


def load_image(image_path: str) -> Image.Image:
    if image_path is None or not os.path.isfile(image_path):
        # 生成一张占位图像（白底）以便快速验证前向过程
        return Image.new("RGB", (512, 512), color=(255, 255, 255))
    return Image.open(image_path).convert("RGB")


def main():
    parser = argparse.ArgumentParser(description="Minimal OpenVLA forward test")
    parser.add_argument("--image", type=str, default=None, help="Path to an input image (optional)")
    parser.add_argument("--instruction", type=str, default="move the crackerbox to align with the gray virtual target pose",
                        help="High-level instruction text")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Device to run on, e.g., cuda:0 or cpu")
    parser.add_argument("--flash_attn", action="store_true", help="Enable flash_attention_2 (requires flash_attn)")
    args = parser.parse_args()

    device = args.device

    # 载入图像
    image = load_image(args.image)

    # 加载处理器与VLA模型
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)

    # 默认关闭 FlashAttention2，除非显式开启
    attn_impl = "flash_attention_2" if args.flash_attn else None
    vla = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        attn_implementation=attn_impl,
        torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        low_cpu_mem_usage=False,
        trust_remote_code=True,
    ).to(device)

    # 组织提示词
    prompt = f"In: What action should the robot take to {{{args.instruction}}}?\nOut:"

    # 前向推理
    inputs = processor(prompt, image).to(device, dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32)
    with torch.inference_mode():
        action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

    # 打印结果
    if isinstance(action, torch.Tensor):
        action_out = action.detach().float().cpu().numpy().reshape(-1).tolist()
    else:
        # 某些实现可能直接返回 ndarray/list
        try:
            action_out = list(action)
        except Exception:
            action_out = [action]

    print("OpenVLA action (7-DoF expected):", action_out)


if __name__ == "__main__":
    main()



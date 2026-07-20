"""
BF16/FP8 → ComfyUI 原生 INT8 ConvRot 量化保存节点
==================================================
完全独立，不 import INT8-Fast 任何模块。

将扩散模型权重一步完成：加载 → ConvRot 旋转 → Per-row INT8 量化 → 保存为原生格式。

内联自 INT8-Fast 的代码（MIT 兼容许可）:
  - convrot.py:  Regular Hadamard 矩阵构建 + 权重旋转
  - int8_quant.py: per-row INT8 量化
  - int8_unet_loader.py: 各模型类型的敏感层排除列表
"""

import os
import json
import math
import torch
import folder_paths
import comfy.utils
from safetensors.torch import save_file


# =============================================================================
# ConvRot 层: Hadamard 旋转（内联自 INT8-Fast convrot.py）
# 参考: QuaRot (2024) / ConvRot (2025)
# =============================================================================

CONVROT_GROUP_SIZE = 256  # 分组大小，必须是 4 的幂（4, 16, 64, 256, 1024...）

# 缓存已构建的 Hadamard 矩阵，避免重复计算
_HADAMARD_CACHE: dict = {}


def _build_hadamard(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """构建 Regular Hadamard 正交矩阵（Theorem 3.3）。
    
    与 Sylvester Hadamard 不同，Regular Hadamard 没有全 1 列，
    避免扩散模型中逐行异常值的放大效应。
    
    Args:
        size: 矩阵尺寸，必须是 4 的幂
        device: 目标设备
        dtype: 数据类型
    """
    cache_key = (size, str(device), str(dtype))
    if cache_key in _HADAMARD_CACHE:
        return _HADAMARD_CACHE[cache_key]

    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError(f"Regular Hadamard 尺寸必须为 4 的幂，收到 {size}")

    # 基础 H4: 每行每列和恰好为 2（Eq 9）
    H4 = torch.tensor(
        [[1, 1, 1, -1],
         [1, 1, -1, 1],
         [1, -1, 1, 1],
         [-1, 1, 1, 1]],
        dtype=dtype, device=device,
    )

    # Kronecker 递推: H_{4^{k+1}} = H_{4^k} ⊗ H_4
    H = H4
    current_size = 4
    while current_size < size:
        H = torch.kron(H, H4)
        current_size *= 4

    # 归一化为正交矩阵（1/√size）
    H = H / (size ** 0.5)
    _HADAMARD_CACHE[cache_key] = H
    return H


def _rotate_weight(
    weight: torch.Tensor, H: torch.Tensor, group_size: int
) -> torch.Tensor:
    """离线旋转 Linear 权重: W_rot = W @ H_block^T。
    
    对 Linear(out, in) 权重 (out, in):
    每行按 group_size 分组，每组右乘 H^T。
    
    Args:
        weight: (out_features, in_features)
        H: (group_size, group_size) 归一化 Hadamard 矩阵
        group_size: 分组大小
    Returns:
        旋转后权重，形状不变
    """
    out_f, in_f = weight.shape
    if in_f % group_size != 0:
        raise ValueError(f"in_features {in_f} 不能被 group_size {group_size} 整除")

    n_groups = in_f // group_size
    # (out, in) → (out, n_groups, group_size)
    W_grouped = weight.view(out_f, n_groups, group_size)
    H_t = H.T.to(dtype=weight.dtype, device=weight.device)
    W_rot = torch.matmul(W_grouped, H_t)
    return W_rot.reshape(out_f, in_f)


# =============================================================================
# INT8 量化层（内联自 INT8-Fast int8_quant.py）
# =============================================================================

def _quantize_int8_per_row(x: torch.Tensor):
    """Per-row INT8 量化（逐行 absmax → scale）。
    
    Args:
        x: (out_features, in_features) 浮点张量
    Returns:
        (q_weight, scale): q_weight 为 int8，scale 形状 (out_features, 1)
    """
    abs_max = x.abs().amax(dim=1, keepdim=True)  # (out, 1)
    scale = (abs_max.float() / 127.0).clamp(min=1e-30)
    q = x.float().div(scale).round_().clamp_(-128.0, 127.0).to(torch.int8)
    return q, scale


# =============================================================================
# 敏感层排除列表（内联自 INT8-Fast int8_unet_loader.py）
# 这些层量化后质量损失显著，保持 BF16/FP8 原样不量化
# =============================================================================

_EXCLUSION_PATTERNS = {
    "flux2": [
        "img_in", "time_in", "guidance_in", "txt_in",
        "double_stream_modulation_img", "double_stream_modulation_txt",
        "single_stream_modulation",
    ],
    "z-image": [
        "cap_embedder", "t_embedder", "x_embedder", "cap_pad_token",
        "context_refiner", "final_layer", "noise_refiner", "adaLN",
        "x_pad_token", "layers.0.",
    ],
    "chroma": [
        "distilled_guidance_layer", "final_layer", "img_in", "txt_in",
        "nerf_image_embedder", "nerf_blocks", "nerf_final_layer_conv",
        "__x0__",
    ],
    "qwen": [
        "time_text_embed", "img_in", "norm_out", "proj_out", "txt_in",
    ],
    "ernie": [
        "time", "x_embedder", "text_proj", "adaLN",
    ],
    "anima": [
        "embed", "llm", "adaln",
    ],
    "krea2": [
        "first", "last", "tmlp", "tproj", "txtfusion", "txtmlp",
    ],
    "hidream o1": [
        "embed", "language_model.layers.35.mlp",
    ],
    "boogu": [
        "embed", "refine", "norm_out",
    ],
    "ideogram4": [
        "embed_image_indicator", "t_embedding", "proj",
    ],
    "wan": [
        "patch_embedding", "text_embedding", "time_embedding",
        "time_projection", "head", "img_emb", "face_adapter",
        "face_encoder", "motion_encoder", "pose_patch_embedding",
    ],
    "ltx2": [
        "adaln", "embedding", "patchify", "to_gate_logits",
        "proj_out", "model.audio", "model.video", "model.av",
        "model.patch", "model.proj", "shift",
    ],
}


def _is_excluded(key: str, model_type: str) -> bool:
    """判断状态字典键是否属于敏感层（应跳过量化）。
    
    匹配逻辑: 去掉模型前缀和 .weight/.bias 后缀后，检查模块路径是否包含排除模式。
    
    Args:
        key: safetensors 键名，如 'model.diffusion_model.img_in.weight'
        model_type: 模型类型
    """
    patterns = _EXCLUSION_PATTERNS.get(model_type)
    if not patterns:
        return False

    # 去掉 ComfyUI 常见模型前缀
    module_path = key
    for prefix in ("model.diffusion_model.", "diffusion_model.", "model."):
        if module_path.startswith(prefix):
            module_path = module_path[len(prefix):]
            break

    # 去掉末尾参数名
    for suffix in (".weight", ".bias"):
        if module_path.endswith(suffix):
            module_path = module_path[: -len(suffix)]
            break

    # 子串匹配排除模式
    for pattern in patterns:
        if pattern in module_path:
            return True

    return False


# =============================================================================
# 主节点: BF16/FP8 → INT8 ConvRot 量化保存
# =============================================================================

class BF16ToINT8Native:
    """将 BF16/FP8 扩散模型量化为 ComfyUI 原生 INT8 ConvRot 格式。

    一步完成整套量化流程:
      加载 BF16/FP8 权重 → Hadamard 旋转(ConvRot) → Per-row INT8 量化 → 保存 safetensors
    
    输出文件可直接被 ComfyUI 原生 INT8 Loader 加载。
    完全独立，不依赖 INT8-Fast 的任何模块。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unet_name": (
                    folder_paths.get_filename_list("diffusion_models"),
                    {"tooltip": "选择 diffusion_models/ 下的模型文件。支持 bf16/fp16/fp32/fp8，自动转为 float32 量化"},
                ),
                "model_type": (
                    [
                        "flux2", "z-image", "ideogram4", "chroma", "krea2",
                        "wan", "ltx2", "qwen", "ernie", "anima",
                        "hidream o1", "boogu",
                    ],
                    {"default": "flux2", "tooltip": "模型类型，决定哪些敏感层（embedding/adaln 等）跳过量化"},
                ),
                "enable_convrot": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "启用 ConvRot Hadamard 旋转。牺牲 ~1.1x 推理速度，换取接近 GGUF Q8 的质量"},
                ),
                "output_name": (
                    "STRING",
                    {"default": "", "tooltip": "输出文件名（不含 .safetensors 扩展名）。留空则自动使用 {原文件名}_int8_native"},
                ),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "convert"
    OUTPUT_NODE = True
    CATEGORY = "loaders"
    DESCRIPTION = (
        "将 BF16/FP8 模型量化为 ComfyUI 原生 INT8 ConvRot 格式，一步完成。"
        "不依赖 INT8-Fast 任何模块，输出可直接被 ComfyUI 原生 INT8 Loader 加载。"
    )

    def convert(
        self,
        unet_name: str,
        model_type: str,
        enable_convrot: bool,
        output_name: str,
    ):
        # ── 路径解析 ────────────────────────────────────────────
        unet_path = folder_paths.get_full_path("diffusion_models", unet_name)
        if not unet_path or not os.path.exists(unet_path):
            raise FileNotFoundError(f"模型文件未找到: {unet_name}")

        # 输出到同目录，方便在 Loader 下拉列表中直接看到
        output_dir = os.path.dirname(unet_path)
        base_name = os.path.splitext(os.path.basename(unet_path))[0]

        if not output_name.strip():
            output_name = f"{base_name}_int8_native"

        output_path = os.path.join(output_dir, f"{output_name}.safetensors")

        # ── 加载源模型状态字典 ──────────────────────────────────
        print(f"[INT8 Native Save] 📂 加载模型: {unet_name}")
        sd, metadata = comfy.utils.load_torch_file(unet_path, return_metadata=True)

        # 安全检查: 避免重复量化已量化的模型
        has_comfy_quant = any(k.endswith(".comfy_quant") for k in sd)
        if has_comfy_quant:
            print("[INT8 Native Save] ⚠️ 检测到已有 comfy_quant 键，此模型似乎已经量化。")
            print("[INT8 Native Save] ⚠️ 跳过转换以避免破坏已有量化数据。")
            print("[INT8 Native Save] 💡 如需重新量化，请使用原始 BF16/FP8 模型。")
            return {}

        # ── 设备与精度 ──────────────────────────────────────────
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        print(f"[INT8 Native Save] 🖥️ 设备: {device}  |  模型: {unet_name}")
        print(f"[INT8 Native Save] 🔧 模型类型: {model_type}  |  ConvRot: {enable_convrot}")

        # ── 构建 Hadamard 矩阵（只构建一次，所有层复用） ────────
        hadamard_H = None
        if enable_convrot:
            try:
                hadamard_H = _build_hadamard(CONVROT_GROUP_SIZE, device, torch.float32)
                print(f"[INT8 Native Save] 🔢 Hadamard 矩阵: {CONVROT_GROUP_SIZE}×{CONVROT_GROUP_SIZE}")
            except Exception as e:
                print(f"[INT8 Native Save] ⚠️ Hadamard 构建失败 ({e})，回退到纯 per-row INT8")
                enable_convrot = False

        # ── 识别需要量化的层 ────────────────────────────────────
        # 判断标准: 2D 张量 + 键名以 .weight 结尾 → 大概率是 Linear 层权重
        weight_keys = [k for k, v in sd.items() if k.endswith(".weight") and v.ndim == 2]

        total_weights = len(weight_keys)
        if total_weights == 0:
            print("[INT8 Native Save] ⚠️ 未找到任何 2D 权重张量，可能不是扩散模型。")
            print("[INT8 Native Save] ⚠️ 跳过转换，原样保存。")
            save_file(sd, output_path, metadata=metadata if isinstance(metadata, dict) else None)
            return {}

        print(f"[INT8 Native Save] 📊 发现 {total_weights} 个 Linear 权重层")

        # ── 逐层量化 ────────────────────────────────────────────
        new_sd = {}          # 新的状态字典
        quantized_count = 0  # 已量化层计数
        skipped_excluded = 0 # 排除的敏感层
        skipped_nondiv = 0   # 不能被 group_size 整除的层（跳过 ConvRot 但仍量化）

        for key, tensor in sd.items():
            # ── 2D weight: 量化候选 ─────────────────────────────
            if key in weight_keys:
                out_f, in_f = tensor.shape

                # 敏感层排除
                if _is_excluded(key, model_type):
                    new_sd[key] = tensor
                    skipped_excluded += 1
                    continue

                # 移到 GPU → 统一转 float32 计算（兼容 bf16/fp16/fp32/fp8 所有格式）
                w = tensor.to(device=device).float()

                # ── ConvRot: Hadamard 旋转 ──────────────────────
                use_convrot = False
                can_convrot = enable_convrot and (in_f % CONVROT_GROUP_SIZE == 0)
                if can_convrot and hadamard_H is not None:
                    try:
                        w = _rotate_weight(w, hadamard_H, CONVROT_GROUP_SIZE)
                        use_convrot = True
                    except Exception as e:
                        print(f"  ⚠️ {key}: ConvRot 旋转异常 ({e})，跳过")
                elif enable_convrot and not can_convrot:
                    skipped_nondiv += 1

                # ── Per-row INT8 量化 ───────────────────────────
                q_weight, q_scale = _quantize_int8_per_row(w)

                # 写回 CPU
                new_sd[key] = q_weight.cpu()

                # 写入 weight_scale（ComfyUI 原生格式必要元数据）
                scale_key = key.replace(".weight", ".weight_scale")
                new_sd[scale_key] = q_scale.cpu()

                # 写入 comfy_quant 元数据（ComfyUI 原生格式）
                quant_key = key.replace(".weight", ".comfy_quant")
                quant_conf = {"format": "int8_tensorwise"}  # ComfyUI 官方量化算法标识
                if use_convrot:
                    quant_conf["convrot"] = True
                    quant_conf["convrot_groupsize"] = CONVROT_GROUP_SIZE
                else:
                    quant_conf["convrot"] = False
                quant_bytes = json.dumps(quant_conf, separators=(",", ":")).encode("utf-8")
                new_sd[quant_key] = torch.tensor(list(quant_bytes), dtype=torch.uint8)

                quantized_count += 1

            # ── 旧的 comfy_quant / weight_scale: 丢弃（避免与新元数据冲突） ──
            elif key.endswith(".comfy_quant") or key.endswith(".weight_scale"):
                continue

            # ── 其他张量: 原样保留（bias / norm / embedding 等） ──
            else:
                new_sd[key] = tensor

        # ── 输出统计 ────────────────────────────────────────────
        print(f"\n[INT8 Native Save] ✅ 量化完成!")
        print(f"  🔹 量化层数:     {quantized_count}")
        print(f"  🔸 排除(敏感层): {skipped_excluded}")
        if skipped_nondiv > 0:
            print(f"  🔸 非整除层:     {skipped_nondiv}（跳过 ConvRot，仍做 per-row INT8）")
        print(f"  📦 总张量数:     {len(new_sd)}")

        # ── 保存 safetensors ────────────────────────────────────
        save_metadata = {}
        if isinstance(metadata, dict):
            save_metadata.update(metadata)

        print(f"\n[INT8 Native Save] 💾 保存到: {output_path}")
        save_file(new_sd, output_path, metadata=save_metadata if save_metadata else None)

        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"[INT8 Native Save] 📏 文件大小: {file_size_mb:.1f} MB")
        print(f"[INT8 Native Save] 🎉 完成！可在 ComfyUI 原生 INT8 Loader 中直接加载。")

        return {}

"""
Comfyui-int8-native-save
========================
将 BF16/FP8 扩散模型一步量化为 ComfyUI 原生 INT8 ConvRot 格式。

完全独立，零外部依赖（除 ComfyUI 和 PyTorch 自带模块）。
不 import INT8-Fast 的任何代码。
"""

from .bf16_to_int8_native import BF16ToINT8Native

# ── ComfyUI 节点注册 ────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "BF16ToINT8Native": BF16ToINT8Native,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BF16ToINT8Native": "Quantize BF16 → INT8 ConvRot (Native)",
}

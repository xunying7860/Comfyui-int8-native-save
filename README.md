# ComfyUI INT8 Native Save

将 BF16/FP16/FP32/FP8 扩散模型**一步量化**为 ComfyUI 原生 INT8 ConvRot 格式。

> ⚡ 一步完成：加载 → ConvRot 旋转 → Per-row INT8 量化 → 保存。零外部依赖。

## 为什么需要这个节点

ComfyUI 新版已[原生支持 INT8 量化](https://github.com/Comfy-Org/ComfyUI/commit/1a510f04234e5a213d3985a1a54f65652623f4bc)，但缺少一个**独立、简单的 BF16/FP8 → INT8 量化保存工具**。

| | INT8-Fast 节点 | 本节点 |
|---|---|---|
| 外部依赖 | Triton + 自身模块 | **零**（仅 PyTorch + safetensors） |
| 量化步骤 | Load OTF → Save 旧格式 → 命令行转换 | **一步** |
| 输出格式 | 旧格式（需 convert_to_comfy.py） | **ComfyUI 原生格式** |
| 源格式支持 | BF16/FP16/FP32 | BF16/FP16/FP32/**FP8** |
| 独立运行 | 依赖 INT8-Fast 注册的 ops | **完全独立** |

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/xunying7860/Comfyui-int8-native-save.git
```

或通过 ComfyUI Manager 搜索 `int8-native-save`。

## 使用方法

在 ComfyUI 节点列表中找到 **"Quantize BF16 → INT8 ConvRot (Native)"**（分类: loaders）。

### 参数

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `unet_name` | 下拉 | — | 选择 `diffusion_models/` 下的模型 |
| `weight_dtype` | 下拉 | `bf16` | 源权重精度（仅信息标识，所有格式统一转 float32 计算） |
| `model_type` | 下拉 | `flux2` | 模型类型，决定哪些敏感层跳过量化 |
| `enable_convrot` | 布尔 | `true` | 启用 ConvRot 旋转（~1.1× 推理开销，接近 GGUF Q8 质量） |
| `output_name` | 字符串 | 自动生成 | 输出文件名（留空 = `{原名}_int8_native`） |

### 工作流

```
BF16/FP8 模型 ──→ [Quantize BF16 → INT8 ConvRot] ──→ 原生 INT8 模型
```

输出文件保存到源模型同目录，可直接被 ComfyUI 原生 INT8 Loader 加载。

## 支持模型

| 模型类型 | 量化策略 |
|---------|---------|
| **flux2** | 排除 `img_in`、`txt_in`、`guidance_in`、调制层 |
| **anima** | 排除 `embed`、`llm`、`adaln` |
| **qwen** | 排除 `time_text_embed`、`img_in`、`proj_out` 等 |
| **z-image** | 排除 `cap_embedder`、`final_layer`、`noise_refiner` 等 |
| **chroma** | 排除 `distilled_guidance_layer`、`nerf_blocks` 等 |
| **krea2** | 排除 `first`、`last`、`tmlp`、`tproj` 等 |
| **wan** | 排除 `patch_embedding`、`head`、`face_adapter` 等 |
| **ltx2** | 排除 `adaln`、`patchify`、`proj_out` 等 |
| **ernie** | 排除 `time`、`x_embedder`、`adaLN` |
| **ideogram4** | 排除 `embed_image_indicator`、`t_embedding` 等 |
| **hidream o1** | 排除 `embed`、LLM 层 |
| **boogu** | 排除 `embed`、`refine`、`norm_out` |

### 量化精度

| 模型 | 格式 | Cos-sim vs BF16 | MSE ↓ |
|------|------|:---:|:---:|
| Anima | INT8 ConvRot | 0.9922 | 0.0075 |
| Flux2 Klein 9B | INT8 ConvRot | 0.9871 | 0.0220 |
| Qwen Image 2512 | INT8 ConvRot | 0.9725 | 0.0089 |

> 数据来源: [INT8-Fast Metrics](https://github.com/BobJohnson24/ComfyUI-INT8-Fast/blob/main/Metrics.md)

### 文件体积

- BF16 (2 字节/参数) → INT8 (1 字节/参数)：体积减半
- 加上 `weight_scale`（float32/行）和 `comfy_quant` 元数据，实际体积约为 BF16 的 **55-60%**

## 依赖

- PyTorch ≥ 2.1（FP8 需 ≥ 2.1）
- safetensors
- ComfyUI（folder_paths、comfy.utils）

## 致谢

本节点的 ConvRot 实现、量化逻辑、模型排除列表借鉴自 [BobJohnson24/ComfyUI-INT8-Fast](https://github.com/BobJohnson24/ComfyUI-INT8-Fast)，已内联为自包含代码，不产生运行时依赖。

## License

MIT

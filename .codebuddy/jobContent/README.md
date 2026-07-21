# 工作内容归档索引

本目录用于归档 AI 协助生成的"工作内容类" Markdown 文档（需求分析、技术方案、汇报材料、排查记录等），按创建日期分文件夹存放。

## 文档列表

| 日期 | 文档 | 主题 | 关联资料 |
| --- | --- | --- | --- |
| 2026-07-08 | [./2026-07-08/Triton入门与AWQ融合算子教学文档.md](./2026-07-08/Triton入门与AWQ融合算子教学文档.md) | Triton 零基础入门教学（6 个递进练习），目标是独立写出 AWQ fused dequant+GEMM kernel | `layers/attention.py::store_kvcache_kernel`、`layers/awq_gemm.py`、Triton 官方教程 |
| 2026-07-08 | [./2026-07-08/AWQ反量化实现方案.md](./2026-07-08/AWQ反量化实现方案.md) | AWQ dequantize 函数的向量化实现方案（Phase A 正确性版本） | `layers/awq_gemm.py`、AutoAWQ 打包格式 |
| 2026-07-08 | [./2026-07-08/AWQ集成代码Bug清单.md](./2026-07-08/AWQ集成代码Bug清单.md) | Phase 1~2 代码审查 Bug 清单（13 个 Bug + 3 个结构遗漏） | `config.py`、`loader.py`、`awq_linear.py`、`awq_gemm.py`、`qwen3.py` |
| 2026-07-07 | [./2026-07-07/AWQ量化接入nano-kvllm实现方案.md](./2026-07-07/AWQ量化接入nano-kvllm实现方案.md) | AWQ 量化原理调研 + 接入 nano-kvllm 的分阶段实现方案 | `nanokvllm/layers/linear.py`、`nanokvllm/utils/loader.py`、`nanokvllm/config.py`、AutoAWQ/vLLM 社区实现 |
| 2026-07-07 | [./2026-07-07/nanokvllm-窗口化周期性压缩算法分析.md](./2026-07-07/nanokvllm-窗口化周期性压缩算法分析.md) | nano-kvllm 窗口化+周期性 KV Cache 压缩机制代码分析 | `nanokvllm/` 目录、`README.md`、`CHANGE_LOG.md` |

## 维护记录

| 日期 | 变更 |
| --- | --- |
| 2026-07-08 | 新增 Triton 入门教学文档（6 个递进练习：向量加法→2D tile→GEMM→int4 解包→AWQ fused kernel 骨架→调优基础）。 |
| 2026-07-08 | 新增 AWQ 反量化实现方案文档（向量化 dequantize，含 AutoAWQ 打包格式分析、验证方法）。 |
| 2026-07-08 | 新增 AWQ 集成代码 Bug 清单（13 Bug + 3 遗漏，按严重程度分级）。 |
| 2026-07-07 | 新增 AWQ 量化接入实现方案文档（含 AWQ 原理调研、AutoAWQ/vLLM 实现调研、分阶段落地路线图）。 |
| 2026-07-07 | 建立本索引，归档 nanokvllm 窗口化+周期性压缩算法分析文档。 |

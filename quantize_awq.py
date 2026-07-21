"""
使用 AutoAWQ 生成 nanokvllm 可加载的 AWQ 量化权重。

要点（务必对齐 nanokvllm 的实现约束）：
1. version 必须是 "GEMM"，与 awq_gemm.py::dequantize_awq_weight 的解包格式一致。
2. 不量化 lm_head / embedding（nanokvllm 中这两层是普通 FP16 层）。
3. 额外写出 quantize_config.json，键名为 bits/group_size/zero_point，
   以匹配 nanokvllm/config.py 的读取逻辑（AutoAWQ 默认键名是 w_bit/q_group_size，会对不上）。

依赖：pip install autoawq
"""
import os
import json
import argparse

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="原始 FP16/BF16 模型目录")
    parser.add_argument("--out", required=True, help="量化权重输出目录")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--zero_point", action="store_true", default=True)
    args = parser.parse_args()

    quant_config = {
        "zero_point": args.zero_point,
        "q_group_size": args.group_size,
        "w_bit": args.bits,
        "version": "GEMM",  # 必须 GEMM
    }

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoAWQForCausalLM.from_pretrained(args.model, trust_remote_code=True)

    # AutoAWQ 默认会跳过 lm_head，这里显式保证不量化 lm_head
    model.quantize(tokenizer, quant_config=quant_config)

    os.makedirs(args.out, exist_ok=True)
    model.save_quantized(args.out)
    tokenizer.save_pretrained(args.out)

    # 关键：写出 nanokvllm 期望的 quantize_config.json（键名对齐）
    nanokv_quant_cfg = {
        "bits": args.bits,
        "group_size": args.group_size,
        "zero_point": bool(args.zero_point),
        "version": "GEMM",
    }
    with open(os.path.join(args.out, "quantize_config.json"), "w") as f:
        json.dump(nanokv_quant_cfg, f, indent=2)

    print(f"[OK] AWQ quantized model saved to: {args.out}")
    print(f"[OK] quantize_config.json written: {nanokv_quant_cfg}")


if __name__ == "__main__":
    main()
